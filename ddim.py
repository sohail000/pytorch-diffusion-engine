import torch
import numpy as np


class DDIMSampler:
    """
    Denoising Diffusion Implicit Models (DDIM) Sampler.
    
    DDIM is a deterministic sampler that allows for much fewer inference steps
    than DDPM while maintaining comparable quality. It interprets the diffusion
    process as a non-Markovian process, enabling:
      - 10-50x fewer steps than DDPM (e.g., 20 steps vs 1000)
      - Deterministic generation (same seed = same output)
      - Controllable stochasticity via eta parameter
    
    Reference: https://arxiv.org/abs/2010.02502 (Song et al., 2020)
    """

    def __init__(
        self,
        generator: torch.Generator,
        num_training_steps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.0120,
    ):
        # Scaled linear schedule (same as SD v1.5)
        self.betas = torch.linspace(
            beta_start ** 0.5, beta_end ** 0.5, num_training_steps, dtype=torch.float32
        ) ** 2
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.one = torch.tensor(1.0)

        self.generator = generator
        self.num_train_timesteps = num_training_steps
        self.timesteps = torch.from_numpy(np.arange(0, num_training_steps)[::-1].copy())

    def set_inference_timesteps(self, num_inference_steps: int = 50):
        self.num_inference_steps = num_inference_steps
        # Uniformly spaced timesteps in reverse
        step_ratio = self.num_train_timesteps // self.num_inference_steps
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps)

    def _get_previous_timestep(self, timestep: int) -> int:
        return timestep - self.num_train_timesteps // self.num_inference_steps

    def set_strength(self, strength: float = 1.0):
        start_step = self.num_inference_steps - int(self.num_inference_steps * strength)
        self.timesteps = self.timesteps[start_step:]
        self.start_step = start_step

    def step(
        self,
        timestep: int,
        latents: torch.Tensor,
        model_output: torch.Tensor,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """
        DDIM step. When eta=0, fully deterministic. When eta=1, equivalent to DDPM.
        
        Implements Eq. 12 from the DDIM paper:
        x_{t-1} = sqrt(α_{t-1}) * predicted_x0 + sqrt(1 - α_{t-1} - σ²) * ε_θ + σ * noise
        """
        t = timestep
        prev_t = self._get_previous_timestep(t)

        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one

        # Predict x_0 from the noise prediction
        pred_original_sample = (latents - (1 - alpha_prod_t) ** 0.5 * model_output) / alpha_prod_t ** 0.5

        # Compute sigma for stochastic component (Eq. 16)
        sigma = eta * (
            (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        ) ** 0.5

        # Direction pointing to x_t (Eq. 12 middle term)
        pred_direction = (1 - alpha_prod_t_prev - sigma ** 2) ** 0.5 * model_output

        # Combine
        pred_prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_direction

        # Add noise only if eta > 0 and not final step
        if eta > 0 and t > 0:
            noise = torch.randn(
                model_output.shape,
                generator=self.generator,
                device=model_output.device,
                dtype=model_output.dtype,
            )
            pred_prev_sample += sigma * noise

        return pred_prev_sample

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
