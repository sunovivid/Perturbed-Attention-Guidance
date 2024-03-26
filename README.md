# Perturbed-Attention Guidance
<a href="https://ku-cvlab.github.io/Perturbed-Attention-Guidance"><img src="https://img.shields.io/badge/Project%20Page-online-brightgreen"></a>

This is the official implementation of the paper "Self-Rectifying Diffusion Sampling with Perturbed-Attention Guidance" by Ahn et al.

![teaser](./imgs/teaser.png)

**Perturbed-Attention Guidance** significantly enhances the sample quality of diffusion models *without requiring external conditions*, such as class labels or text prompts, or *additional training*. This proves particularly valuable in unconditional generation settings, where classifier-free guidance (CFG) is inapplicable. Our guidance can be utilized to enhance performance in various downstream tasks that leverage unconditional diffusion models, including ControlNet with an empty prompt and image restoration tasks like super-resolution and inpainting.

For more information, check out [the project page](https://ku-cvlab.github.io/Perturbed-Attention-Guidance).

## Overview

This repository is based on [SusungHong/Self-Attention-Guidance](https://github.com/SusungHong/Self-Attention-Guidance), which is based on [openai/guided-diffusion](https://github.com/openai/guided-diffusion). The environment setup and the pretrained models are the same as the original repository. The main difference is that the sampling code is modified to support perturbed-attention guidance. Please refer to [Using PAG in Guided-Diffusion](#Using-PAG-in-Guided-Diffusion) for environment setup and sampling.

If you're interested in utilizing PAG with Stable Diffusion, we have made available a [🤗🧨diffusers community pipeline](https://huggingface.co/hyoungwoncho/sd_perturbed_attention_guidance) on the HuggingFace Hub. There's no need to download the entire source code; simply specifying the `custom_pipeline` argument to **hyoungwoncho/sd_perturbed_attention_guidance** with the latest diffusers library (v0.27) is all that's required. Example code is provided in `sd_pag_demo.ipynb`.

## Using PAG with Stable Diffusion

```
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    custom_pipeline="hyoungwoncho/sd_perturbed_attention_guidance",
    torch_dtype=torch.float16
)

device="cuda"
pipe = pipe.to(device)

output = pipe(
        prompts,
        width=512,
        height=512,
        num_inference_steps=50,
        guidance_scale=0.0,
        pag_scale=5.0,
        pag_applied_layers_index=['m0']
    ).images[0]
```
## Using PAG with Guided-Diffusion 
### Environment
The following commands are for setting up the environment using conda. 
- Python 3.9
- PyTorch 1.11.0, Torchvision 0.12.0
- NVIDIA RTX 3090
```
conda create -n pag python=3.9
conda activate pag

conda install gxx_linux-64 #
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```
If you have any issues with the environment setup, please refer to the original repository.

### Downloading Pretrained Diffusion Models (and Classifiers for CG)
Pretrained weights for ImageNet can be downloaded from [the repository](https://github.com/openai/guided-diffusion). Download and place them in the `./models/` directory.

### Sampling from Pretrained Diffusion Models
Run the baseline sampling code first to check if the environment is set up correctly.

### ImageNet 256 Unconditional Sampling (DDIM-25)

```
sh run/sample_uncond_ddim25_baseline.sh
```

The sampling code is modified to support perturbed-attention guidance. The following command samples from the pretrained model.


### ImageNet 256 Unconditional Sampling (DDPM-250)
```
sh run/sample_uncond_ddpm250.sh
```
### ImageNet 256 Conditional Sampling (DDPM-250)
```
sh run/sample_cond_ddpm250.sh
```

### ImageNet 256 Unconditional Sampling (DDIM-25)
```
sh run/sample_uncond_ddim25.sh
```
### ImageNet 256 Conditional Sampling (DDIM-25)
```
sh run/sample_cond_ddim25.sh
```

## Multi-GPU Sampling
If mpiexec is installed, you can use the following command to sample from multiple GPUs.

```
sh run/sample_uncond_ddim25@multigpu.sh
```
it is same with `run/sample_uncond_ddim25.sh` except for the following part.

```
GPU_COUNT=8 # number of GPUs to use
export NCCL_P2P_DISABLE=1 # for multi-node sampling
mpiexec -n $GPU_COUNT 
    ~ same code ~
    --gpu_offset 0  # change --gpu to --gpu_offset
    ~ same code ~
```
