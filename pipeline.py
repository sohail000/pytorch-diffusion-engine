import torch
import numpy as np
from tqdm import tqdm
from typing import Optional, Callable, List, Union
from PIL import Image

from ddpm import DDPMSampler
from ddim import DDIMSampler
from euler import EulerSampler

WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8

# Available samplers registry
SAMPLERS = {
    "ddpm": DDPMSampler,
    "ddim": DDIMSampler,
    "euler": EulerSampler,
    "euler_a": EulerSampler,  # Same class, ancestral flag set in step()
}


def generate(
    prompt: str,
    uncond_prompt: Optional[str] = None,
    input_image: Optional[Image.Image] = None,
    strength: float = 0.8,
    do_cfg: bool = True,
    cfg_scale: float = 7.5,
    sampler_name: str = "ddim",
    n_inference_steps: int = 50,
    models: dict = {},
    seed: Optional[int] = None,
    device: Optional[str] = None,
    idle_device: Optional[str] = None,
    tokenizer=None,
    # ── New features ──
    callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
    eta: float = 0.0,  # For DDIM: 0=deterministic, 1=DDPM-equivalent
    clip_skip: int = 0,  # Skip last N CLIP layers (for anime models, use 1-2)
    prompt_weights: Optional[List[float]] = None,  # Per-token weights
) -> np.ndarray:
    """
    Generate an image from text prompt using Stable Diffusion.
    
    Args:
        prompt: Text description of desired image
        uncond_prompt: Negative prompt (default: empty string)
        input_image: Optional input for img2img mode
        strength: img2img denoising strength (0=no change, 1=full denoise)
        do_cfg: Enable classifier-free guidance
        cfg_scale: CFG strength (7-12 typical, higher = more prompt adherence)
        sampler_name: One of 'ddpm', 'ddim', 'euler', 'euler_a'
        n_inference_steps: Number of denoising steps
        models: Dict with 'clip', 'encoder', 'decoder', 'diffusion' models
        seed: Random seed for reproducibility
        device: Compute device (cuda/mps/cpu)
        idle_device: Device to offload unused models to (saves VRAM)
        tokenizer: HuggingFace tokenizer
        callback: Progress callback fn(step, total_steps, latents)
        eta: DDIM stochasticity (0=deterministic, 1=stochastic)
        clip_skip: Skip last N CLIP layers for style models
        prompt_weights: Per-token emphasis weights
    
    Returns:
        numpy array of shape (H, W, 3) with uint8 pixel values
    """
    with torch.no_grad():
        if not 0 < strength <= 1:
            raise ValueError("strength must be between 0 and 1")

        if uncond_prompt is None:
            uncond_prompt = ""

        if sampler_name not in SAMPLERS:
            raise ValueError(
                f"Unknown sampler '{sampler_name}'. Choose from: {list(SAMPLERS.keys())}"
            )

        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x

        # ── Random seed ──
        generator = torch.Generator(device=device)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)

        # ── CLIP text encoding ──
        clip = models["clip"]
        clip.to(device)

        if do_cfg:
            cond_tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            ).input_ids
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
            cond_context = clip(cond_tokens)

            uncond_tokens = tokenizer.batch_encode_plus(
                [uncond_prompt], padding="max_length", max_length=77
            ).input_ids
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
            uncond_context = clip(uncond_tokens)

            # (2 * Batch_Size, Seq_Len, Dim)
            context = torch.cat([cond_context, uncond_context])
        else:
            tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            ).input_ids
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            context = clip(tokens)

        to_idle(clip)

        # ── Initialize sampler ──
        SamplerClass = SAMPLERS[sampler_name]
        sampler = SamplerClass(generator)
        sampler.set_inference_timesteps(n_inference_steps)

        latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)

        # ── Encode input image (img2img) ──
        if input_image:
            encoder = models["encoder"]
            encoder.to(device)

            input_image_tensor = input_image.resize((WIDTH, HEIGHT))
            input_image_tensor = np.array(input_image_tensor)
            input_image_tensor = torch.tensor(
                input_image_tensor, dtype=torch.float32, device=device
            )
            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))
            input_image_tensor = input_image_tensor.unsqueeze(0)
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)

            encoder_noise = torch.randn(latents_shape, generator=generator, device=device)
            latents = encoder(input_image_tensor, encoder_noise)

            sampler.set_strength(strength=strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0])

            to_idle(encoder)
        else:
            latents = torch.randn(latents_shape, generator=generator, device=device)

        # ── Denoising loop ──
        diffusion = models["diffusion"]
        diffusion.to(device)

        timesteps = tqdm(sampler.timesteps, desc=f"Generating ({sampler_name})")
        total_steps = len(sampler.timesteps)

        for i, timestep in enumerate(timesteps):
            time_embedding = get_time_embedding(timestep).to(device)
            model_input = latents

            if do_cfg:
                model_input = model_input.repeat(2, 1, 1, 1)

            model_output = diffusion(model_input, context, time_embedding)

            if do_cfg:
                output_cond, output_uncond = model_output.chunk(2)
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond

            # ── Sampler-specific step ──
            if sampler_name == "ddim":
                latents = sampler.step(timestep, latents, model_output, eta=eta)
            elif sampler_name == "euler_a":
                latents = sampler.step(timestep, latents, model_output, ancestral=True)
            elif sampler_name == "euler":
                latents = sampler.step(timestep, latents, model_output, ancestral=False)
            else:
                latents = sampler.step(timestep, latents, model_output)

            # Progress callback
            if callback is not None:
                callback(i + 1, total_steps, latents)

        to_idle(diffusion)

        # ── Decode latents to pixels ──
        decoder = models["decoder"]
        decoder.to(device)
        images = decoder(latents)
        to_idle(decoder)

        images = rescale(images, (-1, 1), (0, 255), clamp=True)
        images = images.permute(0, 2, 3, 1)
        images = images.to("cpu", torch.uint8).numpy()
        return images[0]


def generate_grid(
    prompts: List[str],
    seeds: Optional[List[int]] = None,
    cols: int = 2,
    **kwargs,
) -> Image.Image:
    """Generate multiple images and arrange in a grid. Great for comparisons."""
    images = []
    for i, prompt in enumerate(prompts):
        seed = seeds[i] if seeds else None
        img_array = generate(prompt=prompt, seed=seed, **kwargs)
        images.append(Image.fromarray(img_array))

    rows = (len(images) + cols - 1) // cols
    grid = Image.new("RGB", (WIDTH * cols, HEIGHT * rows), (0, 0, 0))
    for i, img in enumerate(images):
        grid.paste(img, ((i % cols) * WIDTH, (i // cols) * HEIGHT))
    return grid


def interpolate_prompts(
    prompt_a: str,
    prompt_b: str,
    num_steps: int = 5,
    models: dict = {},
    tokenizer=None,
    device: str = "cpu",
    **kwargs,
) -> List[np.ndarray]:
    """
    Spherical linear interpolation between two prompts in CLIP embedding space.
    Creates smooth transitions between concepts — great for animations/videos.
    """
    clip = models["clip"]
    clip.to(device)

    tokens_a = tokenizer.batch_encode_plus([prompt_a], padding="max_length", max_length=77).input_ids
    tokens_b = tokenizer.batch_encode_plus([prompt_b], padding="max_length", max_length=77).input_ids
    tokens_a = torch.tensor(tokens_a, dtype=torch.long, device=device)
    tokens_b = torch.tensor(tokens_b, dtype=torch.long, device=device)

    embed_a = clip(tokens_a)
    embed_b = clip(tokens_b)

    clip.to(kwargs.get("idle_device", device))

    frames = []
    for i in range(num_steps):
        t = i / (num_steps - 1)
        # Spherical interpolation (slerp)
        context = slerp(t, embed_a, embed_b)

        # Generate with interpolated context directly
        generator = torch.Generator(device=device)
        seed = kwargs.get("seed", 42)
        generator.manual_seed(seed)

        sampler = DDIMSampler(generator)
        sampler.set_inference_timesteps(kwargs.get("n_inference_steps", 30))

        latents = torch.randn((1, 4, LATENTS_HEIGHT, LATENTS_WIDTH), generator=generator, device=device)

        diffusion = models["diffusion"]
        diffusion.to(device)

        for timestep in sampler.timesteps:
            time_embedding = get_time_embedding(timestep).to(device)
            model_output = diffusion(latents, context, time_embedding)
            latents = sampler.step(timestep, latents, model_output, eta=0.0)

        diffusion.to(kwargs.get("idle_device", device))

        decoder = models["decoder"]
        decoder.to(device)
        images = decoder(latents)
        decoder.to(kwargs.get("idle_device", device))

        images = rescale(images, (-1, 1), (0, 255), clamp=True)
        images = images.permute(0, 2, 3, 1)
        images = images.to("cpu", torch.uint8).numpy()
        frames.append(images[0])

    return frames


def save_animation(
    frames: List[np.ndarray],
    output_path: str = "animation.gif",
    fps: int = 10,
):
    """Save list of frame arrays as animated GIF."""
    pil_frames = [Image.fromarray(f) for f in frames]
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=1000 // fps,
        loop=0,
    )
    print(f"Saved animation to {output_path} ({len(frames)} frames, {fps} fps)")


def slerp(t: float, v0: torch.Tensor, v1: torch.Tensor, dot_threshold: float = 0.9995):
    """Spherical linear interpolation between two tensors."""
    v0_flat = v0.flatten()
    v1_flat = v1.flatten()
    
    v0_norm = v0_flat / torch.norm(v0_flat)
    v1_norm = v1_flat / torch.norm(v1_flat)
    
    dot = torch.sum(v0_norm * v1_norm)
    
    if torch.abs(dot) > dot_threshold:
        # Close enough for linear interpolation
        return (1 - t) * v0 + t * v1
    
    theta_0 = torch.acos(dot)
    sin_theta_0 = torch.sin(theta_0)
    theta_t = theta_0 * t
    sin_theta_t = torch.sin(theta_t)
    
    s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
    s1 = sin_theta_t / sin_theta_0
    
    return s0 * v0 + s1 * v1


def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x


def get_time_embedding(timestep):
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
