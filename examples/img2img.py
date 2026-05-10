"""
Example: Image-to-Image transformation.
Takes an existing image and transforms it using a text prompt.
"""
from pipeline import generate
from model_loader import preload_models_from_standard_weights
from transformers import CLIPTokenizer
from PIL import Image

# ── Config ──
DEVICE = "cuda"
CKPT = "data/v1-5-pruned-emaonly.ckpt"
INPUT_IMAGE = "output.png"  # Use any image you have

# ── Load ──
tokenizer = CLIPTokenizer("data/vocab.json", merges_file="data/merges.txt")
models = preload_models_from_standard_weights(CKPT, DEVICE)

input_img = Image.open(INPUT_IMAGE).convert("RGB")

# ── Different strengths ──
strengths = [0.3, 0.5, 0.7, 0.9]
prompt = "studio ghibli anime style, vibrant colors, fantasy landscape"

for s in strengths:
    print(f"Generating with strength={s}...")
    img = generate(
        prompt=prompt,
        uncond_prompt="photo, realistic, ugly, blurry",
        input_image=input_img,
        strength=s,
        sampler_name="ddim",
        n_inference_steps=25,
        seed=42,
        models=models,
        device=DEVICE,
        idle_device="cpu",
        tokenizer=tokenizer,
    )
    Image.fromarray(img).save(f"img2img_strength_{s}.png")
    print(f"  Saved img2img_strength_{s}.png")

print("Done! Compare the outputs to see how strength affects the result.")
