from pipeline import generate
from model_loader import preload_models_from_standard_weights
from transformers import CLIPTokenizer
from PIL import Image

device = "cuda"
tokenizer = CLIPTokenizer("data/vocab.json", merges_file="data/merges.txt")
models = preload_models_from_standard_weights("data/v1-5-pruned-emaonly.ckpt", device)

img = generate(
    prompt="a castle on a mountain, oil painting",
    uncond_prompt="ugly, blurry",
    sampler_name="ddim",
    n_inference_steps=20,
    seed=42,
    models=models,
    device=device,
    idle_device="cpu",
    tokenizer=tokenizer,
)
Image.fromarray(img).save("output.png")
print("Done! Check output.png")