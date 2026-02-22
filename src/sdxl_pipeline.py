"""SDXL Pipeline with fairness-aware embedding support."""

from typing import Any, List, Optional, Union
from pathlib import Path
import torch
from diffusers import StableDiffusionXLPipeline
from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipelineOutput
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
    retrieve_timesteps,
)

class UserStableDiffusionXLPipeline(StableDiffusionXLPipeline):
    """Stable Diffusion XL pipeline with fairness-aware embedding modifications."""

    @torch.no_grad()
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: str = "pil",
        **kwargs,
    ):
        # 0. Setup
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        device = self._execution_device
        
        # Guidance scale > 1.0 triggers Classifier-Free Guidance (CFG)
        do_classifier_free_guidance = guidance_scale > 1.0

        # 1. Encode prompt
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=kwargs.get("negative_prompt", ""),
        )

        # FIX: Ensure we have paired tensors if CFG is active to avoid 'expected 2, got 1'
        if do_classifier_free_guidance and prompt_embeds.shape[0] == 1:
            prompt_embeds = torch.cat([torch.zeros_like(prompt_embeds), prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([torch.zeros_like(pooled_prompt_embeds), pooled_prompt_embeds], dim=0)

        # 2. Fairness Modification (Extraction & Debias)
        current_exp_dir = getattr(self, "exp_dir", "output/experiments")
        
        if hasattr(self, "processor"):
            # EXTRACTION MODE: Capture embeddings for calibration
            if self.usermode.get("extract"):
                # Use the 'positive' part of the embedding for the bias map
                p_pool = pooled_prompt_embeds.chunk(2)[1] if do_classifier_free_guidance else pooled_prompt_embeds
                
                self.processor.extract_embedding(
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=p_pool,
                    usermode=self.usermode,
                    exp_dir=current_exp_dir,
                    protect=kwargs.get("protect", "gender"),
                    cat=kwargs.get("cat", "default")
                )

            # DEBIASING MODE: Apply FairPCA transformation
            if hasattr(self.processor, "modify_embedding") and self.usermode.get("proc") == "fpca":
                if do_classifier_free_guidance:
                    neg_p, pos_p = prompt_embeds.chunk(2)
                    neg_pool, pos_pool = pooled_prompt_embeds.chunk(2)
                    
                    # Transform only the positive embedding
                    pos_p, pos_pool = self.processor.modify_embedding(
                        self, pos_p, pos_pool, usermode=self.usermode, exp_dir=current_exp_dir
                    )
                    
                    prompt_embeds = torch.cat([neg_p, pos_p], dim=0)
                    pooled_prompt_embeds = torch.cat([neg_pool, pos_pool], dim=0)
                else:
                    prompt_embeds, pooled_prompt_embeds = self.processor.modify_embedding(
                        self, prompt_embeds, pooled_prompt_embeds, 
                        usermode=self.usermode, exp_dir=current_exp_dir
                    )

        # 3. Timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, None
        )

        # 4. Prepare Latents
        # Adjust latent count based on whether we are using CFG
        batch_size = prompt_embeds.shape[0] // (2 if do_classifier_free_guidance else 1)
        latents = self.prepare_latents(
            batch_size, self.unet.config.in_channels, height, width,
            prompt_embeds.dtype, device, generator, None
        )

        # 5. Prepare Time IDs (Required for SDXL)
        add_time_ids = self._get_add_time_ids(
            (height, width), (0, 0), (height, width),
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=self.text_encoder_2.config.projection_dim
        ).to(device)

        if do_classifier_free_guidance:
            add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)

        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # Expand latents for CFG
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # Predict noise
                noise_pred = self.unet(
                    latent_model_input, t,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs={"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids},
                    return_dict=False,
                )[0]

                # Perform guidance math
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # Step the scheduler
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                progress_bar.update()

        # 7. Decode Latents to Image
        if output_type != "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)
        else:
            image = latents

        return StableDiffusionXLPipelineOutput(images=image)
