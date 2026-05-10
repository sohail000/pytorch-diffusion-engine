"""
Example: Prompt Interpolation → Animated GIF
Smoothly morphs between two concepts using spherical interpolation
in CLIP embedding space.
"""
from pipeline import interpolate_prompts, save_animation
from model_loader import preload_models_from_standard_weights
from transformers import CLIPTokenizer

# ── Config ──
DEVICE = "cuda"
CKPT = "data/v1-5-pruned-emaonly.ckpt"

PROMPT_A = "a peaceful snowy village at night, cozy lights"
PROMPT_B = "a tropical beach at sunset, palm trees, golden hour"
NUM_FRAMES = 24
FPS = 8

# ── Load ──
print("Loading models...")
tokenizer = CLIPTokenizer("data/vocab.json", merges_file="data/merges.txt")
models = preload_models_from_standard_weights(CKPT, DEVICE)

# ── Interpolate ──
print(f"Generating {NUM_FRAMES} frames...")
print(f'  "{PROMPT_A}"')
print(f'  → "{PROMPT_B}"')

frames = interpolate_prompts(
    prompt_a=PROMPT_A,
    prompt_b=PROMPT_B,
    num_steps=NUM_FRAMES,
    models=models,
    tokenizer=tokenizer,
    device=DEVICE,
    idle_device="cpu",
    seed=42,
    n_inference_steps=20,
)

save_animation(frames, "prompt_morph.gif", fps=FPS)
print("Done! Open prompt_morph.gif to see the result.")
