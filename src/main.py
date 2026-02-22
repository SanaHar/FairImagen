import json
import torch
import tqdm
import fire
import hyperparse
from pathlib import Path
from base_processor import BaseProcessor
from fairpca_processor import FairPCAProcessor
from pipeline_factory import create_pipeline


# ------------------------------------------------------------
# Helper to extract occupation name from prompt
# ------------------------------------------------------------
def extract_occupation(prompt: str) -> str:
    prompt = prompt.lower()

    prefixes = [
        "generate an image of a ",
        "generate an image of an ",
        "generate a photo of a ",
        "generate a photo of an ",
        "a photo of a ",
        "a photo of an "
    ]

    for p in prefixes:
        if prompt.startswith(p):
            prompt = prompt[len(p):]

    return prompt.strip().replace(" ", "_")


# ------------------------------------------------------------
# Main run loop
# ------------------------------------------------------------
def run(pipe, usermode) -> None:
    data_path = Path(f"data/{usermode['data']}.json")

    with data_path.open() as f:
        data = json.load(f)

    pipe.enable_attention_slicing()

    proc_type = usermode.get("proc", "base")
    steps = 20
    guidance = 7.5

    protect_str = str(usermode.get("protect", "attr")) \
        .replace("[", "").replace("]", "") \
        .replace("'", "").replace(",", "_").replace(" ", "")

    method_folder = (
        "base"
        if proc_type == "base"
        else f"debiased_{protect_str}"
    )

    for item in tqdm.tqdm(data, desc=f"Running {method_folder}"):

        # Handle both string prompts and dict prompts
        if isinstance(item, str):
            raw_prompt = item
        else:
            raw_prompt = item["prompt"]

        occupation_name = extract_occupation(raw_prompt)

        final_dir = Path("output/results") / method_folder / occupation_name
        final_dir.mkdir(parents=True, exist_ok=True)

        with torch.inference_mode():

            iters = int(usermode.get("num_images", 5))

            for i in range(iters):
                generator = torch.Generator("cuda").manual_seed(
                    int(usermode.get("seed", 42)) + i
                )

                output = pipe(
                    prompt=raw_prompt,
                    negative_prompt="",
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    generator=generator,
                    protect=item.get("protect", "gender") if isinstance(item, dict) else "gender",
                    cat=item.get("cat", "default") if isinstance(item, dict) else "default"
                )

                output.images[0].save(
                    final_dir / f"image_{i+1}.jpeg"
                )


# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------
def main(usermode_str="data=test,num_images=10", extramode_str="") -> None:
    usermode = hyperparse.parse_string(usermode_str)
    extramode = hyperparse.parse_string(extramode_str)
    usermode.update(extramode)

    pipe = create_pipeline("sdxl-lightning")
    pipe.to("cuda")

    pipe.exp_dir = "output/experiments"
    pipe.usermode = usermode

    pipe.processor = (
        FairPCAProcessor()
        if usermode.get("proc") == "fpca"
        else BaseProcessor()
    )

    run(pipe, usermode)


if __name__ == "__main__":
    fire.Fire(main)
