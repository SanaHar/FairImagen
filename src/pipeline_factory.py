import torch
from diffusers import AutoPipelineForText2Image, EulerDiscreteScheduler
from sdxl_pipeline import UserStableDiffusionXLPipeline

def create_pipeline(pipename):
    """
    Factory to create the appropriate diffusion pipeline.
    Optimized for Tesla P100 stability and SDXL-Lightning quality.
    """
    if pipename == "sdxl-lightning":
        # Using a stable SDXL checkpoint that avoids 404 errors
        model_id = "Lykon/dreamshaper-xl-v2-turbo"
        
        # Load using your custom class for FairPCA support
        pipe = UserStableDiffusionXLPipeline.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            variant="fp16"
        )
        
        # CRITICAL: Fixes the gray/hazy look in fast-distilled models
        pipe.scheduler = EulerDiscreteScheduler.from_config(
            pipe.scheduler.config, 
            timestep_spacing="trailing",
            prediction_type="epsilon"
        )
        
    elif pipename == "sd15":
        from diffusers import StableDiffusionPipeline
        model_id = "runwayml/stable-diffusion-v1-5"
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16
        )
    else:
        raise ValueError(f"Unknown pipeline: {pipename}")
        
    return pipe
