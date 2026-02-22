import os
import pandas as pd
import numpy as np

metrics_dir = 'metrics_output'
methods = ["base", "debiased_gender", "debiased_race", "debiased_gender_race"]
jobs = ["doctor", "nurse", "scientist", "engineer"]

print(f"{'CATEGORY':<40} | {'BiasW':<7} | {'BiasP':<7} | {'KL':<7} | {'ENS':<7} | {'ICAD':<7}")
print("-" * 85)

for method in methods:
    for job in jobs:
        tag = f"{method}_{job}"
        
        def get_val(suffix, col, g_col='Attribute_Group', g_val='race+gender+age'):
            try:
                df = pd.read_csv(os.path.join(metrics_dir, f"{suffix}_{tag}.csv"))
                if g_col not in df.columns: g_col = 'Attribute'
                return f"{df[df[g_col] == g_val][col].values[0]:.4f}"
            except: return "N/A"

        # Special handling for BiasP (Mean)
        try:
            df_p = pd.read_csv(os.path.join(metrics_dir, f"bias_p_metrics_{tag}.csv"))
            bp = f"{df_p[df_p['Attribute_Group'] == 'race+gender+age']['Bias-P'].mean():.4f}"
        except: bp = "N/A"

        bw = get_val('bias_w_metrics', 'Bias-W')
        kl = get_val('kl_divergence_metrics', 'KL_Divergence')
        ens = get_val('ens', 'ENS')
        icad = get_val('icad_metrics', 'Stratified_ICAD')

        print(f"{tag:<40} | {bw:<7} | {bp:<7} | {kl:<7} | {ens:<7} | {icad:<7}")

print("-" * 85)
