import os
import glob
from concat_generate_and_evaluate import process_paths_and_predict, setup_paths

BASE_RESULTS_DIR = "output/results"
METRICS_BASE_DIR = "metrics_output"

setup_paths(METRICS_BASE_DIR)

folders = ["base", "debiased_gender", "debiased_race", "debiased_gender_race"]
jobs = ["doctor", "nurse", "scientist", "engineer"]

def collect_images(folder):
    return glob.glob(os.path.join(folder, "*.jpeg")) + glob.glob(os.path.join(folder, "*.png"))

for f in folders:
    for job in jobs:
        path = os.path.join(BASE_RESULTS_DIR, f, job)
        if not os.path.exists(path): continue
        img_paths = collect_images(path)
        if not img_paths: continue
        
        tag = f"{f}_{job}"
        print(f"\n>>> Running Evaluation for: {tag}")
        process_paths_and_predict(img_paths, tag)

print("\nAll metrics computed successfully.")
