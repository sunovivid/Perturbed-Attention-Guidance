SAMPLE_FLAGS="--batch_size 5 --num_samples 5000 --timestep_respacing ddim25 --use_ddim True --classifier_guidance False --classifier_scale 0.0"
MODEL_FLAGS="--attention_resolutions 32,16,8 --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
UNCOND_FLAGS="--class_cond False  --model_path models/256x256_diffusion_uncond.pt"
COND_FLAGS="--class_cond True --model_path models/256x256_diffusion.pt --classifier_path models/256x256_classifier.pt"

GPU_COUNT=7 # number of GPUs to use
export NCCL_P2P_DISABLE=1 # for multi-node sampling
mpiexec -n $GPU_COUNT python classifier_sample.py \
    --guide_scale 4.0 \
    --guidance_strategies "{\"attention_mask_identity\":[25,0]}" \
    --drop_layers input_blocks.14.1 input_blocks.16.1 input_blocks.17.1 middle_block.1 \
    --gpu_offset 0 \
    --results_root_dir "RESULTS" \
    --results_dir "uncond_ddim25" \
    --seed 0 \
    --save_png True \
    $SAMPLE_FLAGS \
    $MODEL_FLAGS \
    $UNCOND_FLAGS