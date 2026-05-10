import torch
import numpy as np


class EulerSampler:
    """
    Euler & Euler Ancestral sampler for Stable Diffusion.
    
    The Euler method treats the reverse diffusion as an ODE and solves it
    using first-order Euler steps in the "sigma" (noise level) space.
    This is the default sampler in many modern SD interfaces (Automatic1111, ComfyUI).
    
    Advantages over DDPM/DDIM:
      - Better quality at low step counts (15-25 steps)
      - More natural color distribution
      - Euler Ancestral adds controlled stochasticity for more creative outputs
    
    Reference: Karras et al., "Elucidating the Design Space of Diffusion-Based 
    Generative Models" (https://arxiv.org/abs/2206.00364)
    """

    def __init__(
        self,
        generator: torch.Generator,
        num_training_steps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.0120,
    ):
        self.betas = torch.linspace(
            beta_start ** 0.5, beta_end ** 0.5, num_training_steps, dtype=torch.float32
        ) ** 2
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.one = torch.tensor(1.0)

        # Convert to sigma space: σ = sqrt((1 - α_cumprod) / α_cumprod)
        self.sigmas = ((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5

        self.generator = generator
        self.num_train_timesteps = num_training_steps
        self.timesteps = torch.from_numpy(np.arange(0, num_training_steps)[::-1].copy())

    def set_inference_timesteps(self, num_inference_steps: int = 50):
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_train_timesteps // self.num_inference_steps
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps)

        # Build sigma schedule for inference
        sigmas = []
        for t in self.timesteps:
            sigmas.append(self.sigmas[t])
        sigmas.append(torch.tensor(0.0))  # σ_0 = 0 (final step)
        self.inference_sigmas = torch.stack(sigmas)

    def _get_previous_timestep(self, timestep: int) -> int:
        return timestep - self.num_train_timesteps // self.num_inference_steps

    def set_strength(self, strength: float = 1.0):
        start_step = self.num_inference_steps - int(self.num_inference_steps * strength)
        self.timesteps = self.timesteps[start_step:]
        self.start_step = start_step
        # Rebuild sigmas for truncated schedule
        sigmas = []
        for t in self.timesteps:
            sigmas.append(self.sigmas[t])
        sigmas.append(torch.tensor(0.0))
        self.inference_sigmas = torch.stack(sigmas)

    def step(
        self,
        timestep: int,
        latents: torch.Tensor,
        model_output: torch.Tensor,
        ancestral: bool = False,
    ) -> torch.Tensor:
        """
        Euler step in sigma space.
        
        If ancestral=True, uses Euler Ancestral which adds noise at each step
        for more creative/varied outputs.
        """
        # Find current step index
        step_index = (self.timesteps == timestep).nonzero(as_tuple=True)[0].item()
        sigma = self.inference_sigmas[step_index]
        sigma_next = self.inference_sigmas[step_index + 1]

        # Convert model output (noise prediction) to "denoised" prediction
        # x_0 = (x_t - σ * ε_θ) — in the v-prediction parameterization
        pred_original = latents - sigma * model_output

        # Derivative (direction) in sigma space: d = (x - x_0) / σ
        derivative = (latents - pred_original) / sigma

        if ancestral and sigma_next > 0:
            # Euler Ancestral: split into deterministic + stochastic parts
            sigma_up = (sigma_next ** 2 * (sigma ** 2 - sigma_next ** 2) / sigma ** 2) ** 0.5
            sigma_down = (sigma_next ** 2 - sigma_up ** 2) ** 0.5

            # Deterministic step
            dt = sigma_down - sigma
            prev_sample = latents + derivative * dt

            # Stochastic step
            noise = torch.randn(
                model_output.shape,
                generator=self.generator,
                device=model_output.device,
                dtype=model_output.dtype,
            )
            prev_sample = prev_sample + noise * sigma_up
        else:
            # Standard Euler: deterministic step
            dt = sigma_next - sigma
            prev_sample = latents + derivative * dt

        return prev_sample

    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        alphas_cumprod = self.alphas_cumprod.to(
            device=original_samples.device, dtype=original_samples.dtype
        )
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noise = torch.randn(
            original_samples.shape,
            generator=self.generator,
            device=original_samples.device,
            dtype=original_samples.dtype,
        )
        return sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
