"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""

import argparse
import json
import os

from PIL import Image

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (NUM_CLASSES, add_dict_to_argparser,
                                          args_to_dict, classifier_defaults,
                                          create_classifier,
                                          create_model_and_diffusion,
                                          model_and_diffusion_defaults,
                                          sag_defaults,)

import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
import yaml

import datetime

MAX_GPU_NUM = 8

def get_datetime():
    UTC = datetime.timezone(datetime.timedelta(hours=+9))
    date = datetime.datetime.now(UTC).strftime('%Y-%m-%d_%H-%M-%S')
    return date

def seed_everything(seed):
    def set_seed(seed):
        import random
        random.seed(seed)
        np.random.seed(seed)
        th.manual_seed(seed)
        th.cuda.manual_seed(seed)
        th.backends.cudnn.deterministic = True
        th.backends.cudnn.benchmark = False
    set_seed(int(seed) + dist.get_rank())
    print("Set seed to ", int(seed) + dist.get_rank(), " on GPU ", dist.get_rank())

def get_short_layer_names_str(layers):
    def get_short_layer_name(layer_name):
        return layer_name.split('.')[0][0] + layer_name.split('.')[1] # e.g. input_blocks.14.1 -> i14
    drop_layer_names = [get_short_layer_name(layer_name) for layer_name in layers]
    drop_layer_names = sorted(drop_layer_names)
    drop_layer_names = ' '.join(drop_layer_names)
    return drop_layer_names

def sample_images(args, model, diffusion, classifier):
    import time
    start_time = time.time()

    with open(os.path.join(logger.get_dir(), 'config.yaml'), 'w') as f:
        yaml.dump(args.__dict__, f)

    def cond_fn(x, t, y=None):
        logger.log('Using classifier guidance')
        assert y is not None
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            return th.autograd.grad(selected.sum(), x_in)[0] * args.classifier_scale

    def model_fn(x, t, y=None, perturb=None):
        assert y is not None
        replace_attn_layers(model, perturb)
        model.register_extract_attention_hook() # register hook to extract attention maps (for SAG)
        return model(x, t, y if args.class_cond else None)

    def replace_attn_layers(model, perturb):
        from guided_diffusion.unet import QKVAttentionLegacy, QKVAttentionLegacyDrop, QKVAttentionLegacyNoise, QKVAttentionLegacyMask, QKVAttentionLegacyBlur, QKVAttentionLegacyAvgPool2D

        attn_blocks = [name_module for name_module in model.named_modules() if "AttentionBlock" in name_module[1].__class__.__name__]

        if args.drop_layer_randomly:
            import random
            attn_blocks_to_drop = random.choices(args.drop_layers, k=1)
            if perturb is not None:
                print(f"Randomly selected {attn_blocks_to_drop} to drop")
        else:
            attn_blocks_to_drop = args.drop_layers
        
        for name, module in attn_blocks:
            if name in attn_blocks_to_drop and perturb == 'attention_drop':
                # print(f"Replacing {name}'s attention processor with QKVAttentionLegacyDrop with drop_rate={args.noise_scale}")
                module.attention = QKVAttentionLegacyDrop(module.num_heads, drop_rate=args.noise_scale)
            elif name in attn_blocks_to_drop and perturb == 'attention_pass':
                # print(f"Replacing {name}'s attention processor with QKVAttentionLegacyPass with noise_scale={args.noise_scale}")
                module.attention = QKVAttentionLegacyPass(module.num_heads, drop_rate=args.noise_scale)
            elif name in attn_blocks_to_drop and perturb == 'attention_blur':
                # print(f"Replacing {name}'s attention processor with QKVAttentionLegacyBlur")
                module.attention = QKVAttentionLegacyBlur(module.num_heads, blur_kernel_size=5, blur_sigma=args.blur_sigma)
            elif name in attn_blocks_to_drop and perturb == 'attention_avgpool2d':
                # print(f"Replacing {name}'s attention processor with QKVAttentionLegacyAvgPool2D")
                module.attention = QKVAttentionLegacyAvgPool2D(module.num_heads)
            elif name in attn_blocks_to_drop and perturb == 'attention_noise':
                # print(f"Replacing {name}'s attention processor with QKVAttentionLegacyNoise with noise_scale={args.noise_scale}")
                module.attention = QKVAttentionLegacyNoise(module.num_heads, noise_scale=args.noise_scale)
            elif name in attn_blocks_to_drop and perturb == 'attention_mask':
                # print(f"Replacing {name}'s attention processor with QKVAttentionLegacyMask(retrain_diag=False) with noise_scale={args.noise_scale}")
                module.attention = QKVAttentionLegacyMask(module.num_heads, drop_rate=args.noise_scale, retain_diag=False)
            elif name in attn_blocks_to_drop and perturb == 'attention_mask_identity':
                # print(f"Replacing {name}'s attention processor with QKVAttentionLegacyMask(retrain_diag=True) with noise_scale={args.noise_scale}")
                module.attention = QKVAttentionLegacyMask(module.num_heads, drop_rate=args.noise_scale, retain_diag=True)
            else:
                # reset to original attention processor
                module.attention = QKVAttentionLegacy(module.num_heads)

    logger.log("sampling...")
    all_images = []
    all_labels = []
    shape_str = None
    guidance_kwargs = {}
    guidance_kwargs["guide_scale"] = args.guide_scale
    guidance_kwargs["guide_schedule"] = args.guide_schedule
    guidance_kwargs["blur_sigma"] = args.blur_sigma
    guidance_kwargs["guidance_strategies"] = args.guidance_strategies

    img_num = 0
    i = 0
    while len(all_images) * args.batch_size < args.num_samples:
        if args.seed_everything:
            seed_everything(int(args.seed) + i*MAX_GPU_NUM)
        logger.log("\n")
        logger.log(f"{datetime.datetime.now()}")
        iter_start_time = time.time()
        
        model_kwargs = {}
        logger.log("Sampling random classes")
        classes = th.randint(
            low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
        )
        t = 0

        model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        out = sample_fn(
            model_fn,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=None if not args.classifier_guidance else cond_fn,
            device=dist_util.dev(),
            guidance_kwargs=guidance_kwargs,
            progress=True,
            visualize=args.visualize,
        )
        sample = out["sample"][-1]
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        if args.visualize and i % args.visualize_interval == 0:
            logger.log("Visualizing samples")
            # Visualize the samples, pred_xstart_final, pred_xstart, pred_xstart_perturbed, delta as images
            col_imgs = []
            for samp, pred_xstart_final, pred_xstart, pred_xstart_perturbed, delta in zip(out["sample"], out["pred_xstart_final"], out["pred_xstart"], out["pred_xstart_perturbed"], out["delta"]):
                # Put all in the single column
                if pred_xstart_perturbed is None:
                    pred_xstart_perturbed = th.zeros_like(pred_xstart_final).to('cpu')
                if delta is None:
                    delta = th.zeros_like(pred_xstart_final).to('cpu')
                samp, pred_xstart_final, pred_xstart, pred_xstart_perturbed, delta = samp.cpu(), pred_xstart_final.cpu(), pred_xstart.cpu(), pred_xstart_perturbed.cpu(), delta.cpu()
                col_img = th.cat([samp, pred_xstart_final, pred_xstart, pred_xstart_perturbed, delta], dim=2)
                col_img = ((col_img + 1) * 127.5).clamp(0, 255).to(th.uint8)
                col_img = col_img.permute(0, 2, 3, 1)
                col_img = col_img.contiguous()
                col_img = col_img.cpu().numpy()
                col_imgs.append(col_img)
            # Concatenate each column to the final image
            for img_num_ in range(args.batch_size):
                # init zero np array
                final_img = np.array([], dtype=np.uint8).reshape(5 * args.image_size, 0, 3)
                for col_index, col in enumerate(col_imgs):
                    if col_index % args.visualize_step_interval != 0:
                        continue
                    col = col[img_num_].squeeze()
                    final_img = np.concatenate((final_img, col), axis=1)
                img_path = os.path.join(logger.get_dir(), f"image_{i}_{img_num_}_gpu_{dist.get_rank()}_visualization.png")
                img_pil = Image.fromarray(final_img)
                img_pil.save(img_path)
                print("Saved image to ", img_path)
                
        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        gathered_labels = [th.zeros_like(classes) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_labels, classes)
        all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} / {args.num_samples} samples")

        if args.save_png:
            logger.log("Saving images as PNG files")
            # Save images as PNG files
            for idx, (col_img, label) in enumerate(zip(sample, classes)):
                if not os.path.exists(os.path.join(logger.get_dir(), 'png')):
                    os.makedirs(os.path.join(logger.get_dir(), 'png'), exist_ok=True)
                img_path = os.path.join(logger.get_dir(), 'png', f"image_{i}_{idx}_class_{label}_gpu_{dist.get_rank()}.png")
                img_pil = Image.fromarray(col_img.cpu().numpy())
                img_pil.save(img_path)

        img_num += args.batch_size * t
        i += 1
        
        iter_elapsed_time = time.time() - iter_start_time
        logger.log(f"Elapsed time for this iteration: {iter_elapsed_time:.2f}s")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    label_arr = np.concatenate(all_labels, axis=0)
    label_arr = label_arr[: args.num_samples]

    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr, label_arr)

    dist.barrier()
    logger.log("sampling complete")

    elapsed_time = time.time() - start_time
    logger.log(f"Elapsed time: {elapsed_time:.2f}s")

    import sys
    with open(os.path.join(logger.get_dir(), 'command.txt'), 'w') as f:
        f.write('python ' + ' '.join(sys.argv))

def main():
    args = create_argparser().parse_args()
    results_dir = f'{args.results_root_dir}/{args.results_dir}'

    # set gpu device
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        logger.info(f"Using GPU {args.gpu}")
        dist_util.setup_dist(gpu_index=args.gpu)
    else:
        dist_util.setup_dist(gpu_offset=args.gpu_offset)

    if args.seed is not None and args.seed_everything:
        seed_everything(int(args.seed))

    if args.log_datetime:
        base_logger_dir = f"{results_dir}/{args.note}@{get_datetime()}"
    else:
        base_logger_dir = f"{results_dir}/{args.note}"
    logger.configure(dir=base_logger_dir)

    with open(os.path.join(logger.get_dir(), 'config.yaml'), 'w') as f:
        yaml.dump(args.__dict__, f)

    # write command to a file
    import sys
    with open(os.path.join(logger.get_dir(), 'command.txt'), 'w') as f:
        f.write('python ' + ' '.join(sys.argv))

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        sel_attn_depth=args.sel_attn_depth,
        sel_attn_block=args.sel_attn_block,
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    if args.class_cond:
        logger.log("loading classifier...")
        classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
        classifier.load_state_dict(
            dist_util.load_state_dict(args.classifier_path, map_location="cpu")
        )
        classifier.to(dist_util.dev())
        if args.classifier_use_fp16:
            classifier.convert_to_fp16()
        classifier.eval()
    else:
        classifier = None
        logger.log("not using classifier")

    sample_images(args, model, diffusion, classifier)

def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        classifier_path="",
        classifier_scale=1.0,
        sample_class=-1,
        gpu=None,
        gpu_offset=0,
        note="",
        visualize=False,
        visualize_interval=1,
        visualize_step_interval=1,
        verbose=True,
        results_dir="default",
        results_root_dir="RESULTS",
        save_png=True,
        log_datetime=True,
        seed=0,
        seed_everything=True,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    defaults.update(sag_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    parser.add_argument("--drop_layers", nargs="+", default=['input_blocks.14.1', 'input_blocks.16.1', 'input_blocks.17.1', 'middle_block.1'])
    parser.add_argument('--guidance_strategies', type=json.loads)
    return parser


if __name__ == "__main__":
    main()

    '''
    ImageNet 128x128
    ['input_blocks.7.1', 'input_blocks.8.1', 'input_blocks.10.1', 'input_blocks.11.1', 'input_blocks.13.1', 'input_blocks.14.1', 
    'middle_block.1', 
    'output_blocks.0.1', 'output_blocks.1.1', 'output_blocks.2.1', 'output_blocks.3.1', 'output_blocks.4.1', 'output_blocks.5.1', 'output_blocks.6.1', 'output_blocks.7.1', 'output_blocks.8.1']

    ImageNet 256x256
    ['input_blocks.10.1', 'input_blocks.11.1', 'input_blocks.13.1', 'input_blocks.14.1', 'input_blocks.16.1', 'input_blocks.17.1',
    'middle_block.1',
    'output_blocks.0.1', 'output_blocks.1.1', 'output_blocks.2.1', 'output_blocks.3.1', 'output_blocks.4.1', 'output_blocks.5.1', 'output_blocks.6.1', 'output_blocks.7.1', 'output_blocks.8.1']
    '''