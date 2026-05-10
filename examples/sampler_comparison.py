"""
Example: Text-to-Image generation with different samplers.
Generates the same prompt with DDPM, DDIM, Euler, and Euler Ancestral,
then stitches them into a comparison grid.
"""
from pipeline import generate
from model_loader import preload_models_from_standard_weights
from transformers import CLIPTokenizer
from PIL import Image, ImageDraw, ImageFont
import time

# ── Config ──
DEVICE = "cuda"  # "mps" for Mac, "cpu" for no GPU
CKPT = "data/v1-5-pruned-emaonly.ckpt"
PROMPT = "a serene japanese garden with cherry blossoms, watercolor painting, highly detailed"
NEGATIVE = "ugly, blurry, low quality, distorted"
SEED = 42

# ── Load ──
print("Loading models...")
tokenizer = CLIPTokenizer("data/vocab.json", merges_file="data/merges.txt")
models = preload_models_from_standard_weights(CKPT, DEVICE)

# ── Generate with each sampler ──
samplers = {
    "DDPM (50 steps)": ("ddpm", 50),
    "DDIM (20 steps)": ("ddim", 20),
    "Euler (20 steps)": ("euler", 20),
    "Euler A (20 steps)": ("euler_a", 20),
}

results = {}
for label, (sampler, steps) in samplers.items():
    print(f"\nGenerating with {label}...")
    start = time.time()
    img = generate(
        prompt=PROMPT,
        uncond_prompt=NEGATIVE,
        sampler_name=sampler,
        n_inference_steps=steps,
        cfg_scale=7.5,
        seed=SEED,
        models=models,
        device=DEVICE,
        idle_device="cpu",
        tokenizer=tokenizer,
    )
    elapsed = time.time() - start
    print(f"  Done in {elapsed:.1f}s")
    results[label] = Image.fromarray(img)

# ── Stitch grid ──
grid = Image.new("RGB", (1024, 1024), (20, 20, 20))
for i, (label, img) in enumerate(results.items()):
    x, y = (i % 2) * 512, (i // 2) * 512
    grid.paste(img, (x, y))
    # Add label
    draw = ImageDraw.Draw(grid)
    draw.rectangle([x, y + 480, x + 512, y + 512], fill=(0, 0, 0, 180))
    draw.text((x + 10, y + 488), label, fill="white")

grid.save("sampler_comparison.png")
print(f"\nSaved sampler_comparison.png")
