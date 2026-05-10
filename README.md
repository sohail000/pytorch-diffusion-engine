# Stable Diffusion - From Scratch in PyTorch

A complete, from-scratch implementation of [Stable Diffusion v1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5) in pure PyTorch. Every component - CLIP text encoder, VAE encoder/decoder, U-Net with cross-attention, and noise schedulers - is implemented manually with no dependency on `diffusers` or `stable-diffusion` libraries.

Built for **deep understanding**, not just usage. Every tensor shape is annotated. Every architectural decision is documented.

---

## Architecture

```
                    ┌─────────────────────────────────────────────────────┐
                    │                   CLIP Text Encoder                 │
                    │  "a photo of     ┌──────────┐   (1, 77, 768)       │
                    │   an astronaut"──│ 12 Layers │──── context ─────────┤
                    │                  │ (Causal   │                      │
                    │                  │  Self-Attn│                      │
                    │                  └──────────┘                      │
                    └───────────────────────┬─────────────────────────────┘
                                            │
                    ┌───────────────────────▼─────────────────────────────┐
                    │                   U-Net Diffusion                   │
                    │                                                     │
                    │  Noisy Latent ──► Encoder ──► Bottleneck ──► Decoder│
                    │  (1,4,64,64)     │         │           │           │
                    │                  │ Cross   │ Cross     │ Cross     │
                    │                  │ Attn    │ Attn      │ Attn      │
                    │                  │ + Self  │ + Self    │ + Self    │
                    │                  │ Attn    │ Attn      │ Attn      │
                    │                  │         │           │           │
                    │  Time Embed ─────┤─────────┤───────────┤           │
                    │  (sinusoidal)    │         │           │           │
                    │                  └─── Skip Connections ┘           │
                    │                                                     │
                    │  Output: Predicted Noise (1, 4, 64, 64)            │
                    └───────────────────────┬─────────────────────────────┘
                                            │ × N steps (reverse diffusion)
                    ┌───────────────────────▼─────────────────────────────┐
                    │                   VAE Decoder                       │
                    │  Denoised Latent ──► Upsample 8× ──► RGB Image     │
                    │  (1, 4, 64, 64)                      (1, 3, 512, 512)│
                    └─────────────────────────────────────────────────────┘
```

## Features

| Feature | Status |
|---------|--------|
| Text-to-Image generation | ✅ |
| Image-to-Image (img2img) | ✅ |
| Classifier-Free Guidance | ✅ |
| DDPM Sampler | ✅ |
| **DDIM Sampler** (deterministic, 10-50x fewer steps) | ✅ NEW |
| **Euler / Euler Ancestral Sampler** | ✅ NEW |
| **Flash Attention** (PyTorch 2.0+ auto-detected) | ✅ NEW |
| **Prompt Interpolation** (slerp in CLIP space) | ✅ NEW |
| **GIF/Animation export** | ✅ NEW |
| **Grid generation** (side-by-side comparisons) | ✅ NEW |
| **Progress callbacks** | ✅ NEW |
| Fine-tuned model support (any SD v1.5 ckpt) | ✅ |

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download model weights

Download from HuggingFace and place in a `data/` folder:

```bash
mkdir data

# Tokenizer files
wget -P data/ https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/tokenizer/vocab.json
wget -P data/ https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/tokenizer/merges.txt

# Model weights (~4GB)
wget -P data/ https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt
```

### 3. Generate images

```python
from pipeline import generate
from model_loader import preload_models_from_standard_weights
from transformers import CLIPTokenizer

# Load
device = "cuda"  # or "mps" for Apple Silicon, "cpu" for CPU
tokenizer = CLIPTokenizer("data/vocab.json", merges_file="data/merges.txt")
models = preload_models_from_standard_weights("data/v1-5-pruned-emaonly.ckpt", device)

# Generate - using DDIM (fast, 20 steps)
image = generate(
    prompt="a beautiful sunset over mountains, oil painting, highly detailed",
    uncond_prompt="ugly, blurry, low quality",
    sampler_name="ddim",
    n_inference_steps=20,
    cfg_scale=7.5,
    seed=42,
    models=models,
    device=device,
    idle_device="cpu",
    tokenizer=tokenizer,
)

from PIL import Image
Image.fromarray(image).save("output.png")
```

### 4. Image-to-Image

```python
from PIL import Image

input_img = Image.open("input.jpg")
result = generate(
    prompt="turn this into a watercolor painting",
    uncond_prompt="photo, realistic",
    input_image=input_img,
    strength=0.7,        # 0.3 = subtle change, 0.9 = major change
    sampler_name="euler",
    n_inference_steps=25,
    models=models,
    device=device,
    idle_device="cpu",
    tokenizer=tokenizer,
)
Image.fromarray(result).save("img2img_output.png")
```

### 5. Prompt Interpolation → Animated GIF

```python
from pipeline import interpolate_prompts, save_animation

frames = interpolate_prompts(
    prompt_a="a serene lake at dawn",
    prompt_b="a volcanic eruption at night",
    num_steps=30,
    models=models,
    tokenizer=tokenizer,
    device=device,
    idle_device="cpu",
    seed=42,
    n_inference_steps=20,
)
save_animation(frames, "morph.gif", fps=10)
```

### 6. Sampler Comparison Grid

```python
from pipeline import generate
from PIL import Image

prompt = "a cyberpunk cityscape at night, neon lights"
results = {}
for sampler in ["ddpm", "ddim", "euler", "euler_a"]:
    steps = 50 if sampler == "ddpm" else 20
    img = generate(
        prompt=prompt,
        sampler_name=sampler,
        n_inference_steps=steps,
        seed=42,
        models=models, device=device,
        idle_device="cpu", tokenizer=tokenizer,
    )
    results[sampler] = Image.fromarray(img)

# Stitch into 2x2 grid
grid = Image.new("RGB", (1024, 1024))
for i, (name, img) in enumerate(results.items()):
    grid.paste(img, ((i % 2) * 512, (i // 2) * 512))
grid.save("sampler_comparison.png")
```

## Samplers Explained

| Sampler | Steps | Deterministic | Best For |
|---------|-------|--------------|----------|
| **DDPM** | 50-1000 | No | Baseline, academic reference |
| **DDIM** | 10-50 | Yes (eta=0) | Fast generation, reproducibility |
| **Euler** | 15-30 | Yes | General purpose, good quality/speed |
| **Euler Ancestral** | 15-30 | No | Creative/varied outputs |

## Tested Fine-Tuned Models

Any Stable Diffusion v1.5 compatible checkpoint works:

- [InkPunk Diffusion](https://huggingface.co/Envvi/Inkpunk-Diffusion/tree/main)
- [Illustration Diffusion](https://huggingface.co/ogkalu/Illustration-Diffusion/tree/main)
- [Anything V3](https://huggingface.co/Linaqruf/anything-v3.0) (anime, use `clip_skip=1`)
- [Deliberate](https://huggingface.co/XpucT/Deliberate) (photorealistic)
- [DreamShaper](https://huggingface.co/Lykon/DreamShaper) (artistic)

## File Structure

```
├── pipeline.py          # Main generation pipeline (txt2img, img2img, interpolation)
├── attention.py         # Self-Attention & Cross-Attention (with Flash Attention)
├── clip.py              # CLIP text encoder (12-layer transformer)
├── encoder.py           # VAE encoder (image → latent space)
├── decoder.py           # VAE decoder (latent space → image)
├── diffusion.py         # U-Net with time & text conditioning
├── ddpm.py              # DDPM sampler (original)
├── ddim.py              # DDIM sampler (fast, deterministic)
├── euler.py             # Euler & Euler Ancestral samplers
├── model_loader.py      # Weight loading utility
├── model_converter.py   # Convert official SD weights to our format
├── demo.ipynb           # Interactive demo notebook
└── add_noise.ipynb      # Educational: visualize the forward diffusion process
```

## Performance

| Device | Sampler | Steps | Time |
|--------|---------|-------|------|
| RTX 4090 | DDIM | 20 | ~3s |
| RTX 3080 | DDIM | 20 | ~6s |
| M2 Pro (MPS) | DDIM | 20 | ~15s |
| CPU (i7) | DDIM | 20 | ~5min |

Flash Attention (PyTorch 2.0+) is auto-detected and provides ~30% speedup with ~50% less VRAM.

## How It Works (The Short Version)

1. **CLIP** encodes your text prompt into a sequence of 77 embeddings (768-dim each)
2. The **U-Net** starts from pure noise and iteratively denoises it, guided by the text embeddings via cross-attention
3. At each step, the U-Net predicts the noise to remove. The **sampler** uses this prediction to compute the next (less noisy) state
4. **Classifier-Free Guidance** runs the U-Net twice (with and without text) and amplifies the difference - this is why `cfg_scale` controls how strongly the image follows your prompt
5. The final denoised latent is decoded by the **VAE Decoder** back into a 512×512 RGB image

## Acknowledgments

- [CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion/)
- [divamgupta/stable-diffusion-tensorflow](https://github.com/divamgupta/stable-diffusion-tensorflow)
- [kjsman/stable-diffusion-pytorch](https://github.com/kjsman/stable-diffusion-pytorch)
- [huggingface/diffusers](https://github.com/huggingface/diffusers/)

## License

See [license.txt](license.txt).
