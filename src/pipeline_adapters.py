"""Pipeline Adapters for different models to support fairness processors.

This module contains adapters for FLUX, PixArt-Sigma, and Würstchen models.
For SDXL, see sdxl_pipeline.py. For SD3, see sdpipline.py.
"""

import torch
from diffusers import DiffusionPipeline
from typing import Optional, Union, List


class UserFluxPipeline(DiffusionPipeline):
    """FLUX Pipeline with fairness support."""

    @torch.no_grad()
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 4,  # FLUX schnell uses 1-4 steps
        guidance_scale: float = 0.0,  # FLUX often uses no guidance
        num_images_per_prompt: Optional[int] = 1,
        **kwargs,
    ):
        """Generate images with FLUX model using fairness-aware processing."""
        device = (
            self._execution_device if hasattr(self, "_execution_device") else "cuda"
        )

        # FLUX specific prompt encoding
        if hasattr(self, "encode_prompt"):
            prompt_embeds = self.encode_prompt(
                prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
            )
        else:
            text_encoder_output = self.text_encoder(prompt, return_dict=True)
            prompt_embeds = text_encoder_output.last_hidden_state
            pooled_prompt_embeds = (
                text_encoder_output.pooler_output
                if hasattr(text_encoder_output, "pooler_output")
                else None
            )

        # Fairness Integration Point 1: Extract features
        if hasattr(self, "usermode") and "extract" in self.usermode:
            pooled = (
                pooled_prompt_embeds
                if "pooled_prompt_embeds" in locals()
                else prompt_embeds.mean(dim=1)
            )
            self.processor.extract_embedding(
                prompt_embeds,
                pooled,
                self.processor,
                usermode=self.usermode,
                imagemodel="flux",
                exp_dir=self.exp_dir,
                **kwargs,
            )
            return None

        # Fairness Integration Point 2: Modify embeddings
        if hasattr(self, "processor"):
            if "pooled_prompt_embeds" in locals():
                prompt_embeds, pooled_prompt_embeds = self.processor.modify_embedding(
                    self,
                    prompt_embeds,
                    pooled_prompt_embeds,
                    usermode=self.usermode,
                    exp_dir=self.exp_dir,
                )
            else:
                modified_embeds, _ = self.processor.modify_embedding(
                    self,
                    prompt_embeds,
                    prompt_embeds.mean(dim=1),  # Create pseudo pooled embeddings
                    usermode=self.usermode,
                    exp_dir=self.exp_dir,
                )
                prompt_embeds = modified_embeds

        # Continue with FLUX generation
        kwargs["prompt_embeds"] = prompt_embeds

        return super().__call__(
            prompt=None,  # We already have embeddings
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            **kwargs,
        )


class UserPixArtSigmaPipeline(DiffusionPipeline):
    """PixArt-Sigma Pipeline with fairness support."""

    @torch.no_grad()
    def __call__(self, prompt: Optional[Union[str, List[str]]] = None, **kwargs):
        """Generate images with PixArt-Sigma model using fairness-aware processing."""
        device = (
            self._execution_device if hasattr(self, "_execution_device") else "cuda"
        )

        # Get embeddings
        if hasattr(self, "encode_prompt"):
            (
                prompt_embeds,
                _prompt_attention_mask,
                _negative_prompt_embeds,
                _negative_prompt_attention_mask,
            ) = self.encode_prompt(
                prompt=prompt,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                device=device,
            )
        else:
            prompt_embeds = self.text_encoder(prompt)[0]

        # Fairness Integration Point 1: Extract features
        if hasattr(self, "usermode") and "extract" in self.usermode:
            pooled = prompt_embeds.mean(dim=1)
            self.processor.extract_embedding(
                prompt_embeds,
                pooled,
                self.processor,
                usermode=self.usermode,
                imagemodel="pixart",
                exp_dir=self.exp_dir,
                **kwargs,
            )
            return None

        # Fairness Integration Point 2: Modify embeddings
        if hasattr(self, "processor"):
            prompt_embeds, _ = self.processor.modify_embedding(
                self,
                prompt_embeds,
                prompt_embeds.mean(dim=1),
                usermode=self.usermode,
                exp_dir=self.exp_dir,
            )

        kwargs["prompt_embeds"] = prompt_embeds
        return super().__call__(prompt=None, **kwargs)


class UserWuerstchenPipeline:
    """Würstchen Pipeline with fairness support."""

    def __init__(self, base_pipeline) -> None:
        """Initialize Würstchen pipeline wrapper with fairness support."""
        self.base_pipeline = base_pipeline
        for attr in dir(base_pipeline):
            if not attr.startswith("_") and not hasattr(self, attr):
                setattr(self, attr, getattr(base_pipeline, attr))

    @torch.no_grad()
    def __call__(self, prompt=None, **kwargs):
        """Generate images with Würstchen model using fairness-aware processing."""
        if hasattr(self, "encode_prompt"):
            text_embeddings = self.encode_prompt(prompt)
        else:
            text_embeddings = self.prior_text_encoder(prompt)

        if hasattr(self, "usermode") and "extract" in self.usermode:
            pooled = text_embeddings.mean(dim=1)
            self.processor.extract_embedding(
                text_embeddings,
                pooled,
                self.processor,
                usermode=self.usermode,
                imagemodel="wurstchen",
                exp_dir=self.exp_dir,
                **kwargs,
            )
            return None

        if hasattr(self, "processor"):
            text_embeddings, _ = self.processor.modify_embedding(
                self,
                text_embeddings,
                text_embeddings.mean(dim=1),
                usermode=self.usermode,
                exp_dir=self.exp_dir,
            )

        kwargs["prompt_embeds"] = text_embeddings
        return self.base_pipeline(prompt=None, **kwargs)


def create_adapted_pipeline(base_pipeline, processor, usermode, exp_dir):
    """Inject processor and usermode into any pipeline."""
    base_pipeline.processor = processor
    base_pipeline.usermode = usermode
    base_pipeline.exp_dir = exp_dir

    original_call = base_pipeline.__call__

    @torch.no_grad()
    def adapted_call(prompt=None, **kwargs):
        if prompt is not None and hasattr(base_pipeline, "encode_prompt"):
            encode_result = base_pipeline.encode_prompt(prompt)

            if isinstance(encode_result, tuple):
                prompt_embeds = encode_result[0]
                pooled_embeds = (
                    encode_result[1]
                    if len(encode_result) > 1
                    else prompt_embeds.mean(dim=1)
                )
            else:
                prompt_embeds = encode_result
                pooled_embeds = prompt_embeds.mean(dim=1)

            if "extract" in usermode:
                processor.extract_embedding(
                    prompt_embeds,
                    pooled_embeds,
                    processor,
                    usermode=usermode,
                    imagemodel=base_pipeline.__class__.__name__,
                    exp_dir=exp_dir,
                    **kwargs,
                )
                return None

            prompt_embeds, pooled_embeds = processor.modify_embedding(
                base_pipeline,
                prompt_embeds,
                pooled_embeds,
                usermode=usermode,
                exp_dir=exp_dir,
            )

            kwargs["prompt_embeds"] = prompt_embeds
            if "pooled_prompt_embeds" in kwargs:
                kwargs["pooled_prompt_embeds"] = pooled_embeds

            prompt = None

        return original_call(prompt=prompt, **kwargs)

    base_pipeline.__call__ = adapted_call
    return base_pipeline
