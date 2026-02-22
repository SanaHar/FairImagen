from itertools import product
from pathlib import Path
import torch

from base_processor import BaseProcessor
from fair_PCA import FairPCA


class FairPCAProcessor(BaseProcessor):
    """Applies FairPCA debiasing to prompt embeddings."""

    def modify_embedding(
        self,
        pipe,
        prompt_embeds,
        pooled_prompt_embeds,
        usermode=None,
        exp_dir=None,
    ):
        if usermode is None:
            usermode = {}

        if exp_dir is None:
            exp_dir = getattr(pipe, "exp_dir", Path("output/experiments"))

        if usermode.get("proc") != "fpca" or "remove" not in usermode:
            return prompt_embeds, pooled_prompt_embeds

        print(f"\n[FairPCA] ACTION: Debiasing for {usermode.get('protect')}")

        if not hasattr(pipe, "fpcas"):
            pipe.fpcas = calc_projection_matrix(exp_dir, usermode)

            for fpca in pipe.fpcas:
                fpca.UUT = torch.tensor(
                    fpca.UUT,
                    dtype=pooled_prompt_embeds.dtype,
                    device=pooled_prompt_embeds.device,
                )

        for fpca in pipe.fpcas:

            if pooled_prompt_embeds.shape[-1] == fpca.UUT.shape[0]:
                pooled_prompt_embeds = fpca.transform(pooled_prompt_embeds)

            if prompt_embeds.shape[-1] == 2048:
                p1 = prompt_embeds[:, :, :768]
                p2 = prompt_embeds[:, :, 768:]

                if p1.shape[-1] == fpca.UUT.shape[0]:
                    p1 = fpca.transform(p1)
                if p2.shape[-1] == fpca.UUT.shape[0]:
                    p2 = fpca.transform(p2)

                prompt_embeds = torch.cat((p1, p2), dim=-1)

            else:
                if prompt_embeds.shape[-1] == fpca.UUT.shape[0]:
                    prompt_embeds = fpca.transform(prompt_embeds)

        print("[FairPCA] SUCCESS: Embeddings modified.")
        return prompt_embeds, pooled_prompt_embeds


# ============================================================
# Projection logic
# ============================================================

def calc_projection_matrix(exp_dir, usermode):
    protect_str = str(usermode.get("protect", "attr")) \
        .replace("[", "").replace("]", "") \
        .replace("'", "").replace(",", "_").replace(" ", "")

    feature_path = Path(exp_dir) / f"extracted_features_{protect_str}.pt"

    if not feature_path.exists():
        raise FileNotFoundError(f"Missing calibration file: {feature_path}")

    print(f"[FairPCA] Loading calibration file: {feature_path}")

    data = torch.load(feature_path, weights_only=True)

    # 🔥 Stronger debiasing
    hdim = int(usermode.get("hdim", 1800))
    tradeoff = 2.0

    f = FairPCA(
        target_dim=hdim,
        standardize=False,
        tradeoff_param=tradeoff
    )

    f.usermode = usermode

    if len(data.keys()) > 1:
        print("[FairPCA] INFO: INTERSECTIONAL mode")
        calc_projection_matrix_mgmd_cross(data, f)
    else:
        print("[FairPCA] INFO: SINGLE-ATTRIBUTE mode")
        protect = next(iter(data))

        if len(data[protect]) == 2:
            calc_projection_matrix_sg(data[protect], f)
        else:
            calc_projection_matrix_mg(data[protect], f)

    return [f]


def calc_projection_matrix_sg(data, fpca):
    X = torch.cat(list(data.values()), dim=0)
    keys = list(data.keys())

    z = torch.cat([
        torch.zeros(data[keys[0]].shape[0]),
        torch.ones(data[keys[1]].shape[0])
    ]).to(X.dtype)

    fpca.fit(X, z)


def calc_projection_matrix_mg(data, fpca):
    X = torch.cat(list(data.values()), dim=0)
    keys = list(data.keys())

    Z = torch.zeros(X.shape[0], len(keys)).type_as(X)

    start = 0
    for i, k in enumerate(keys):
        size = data[k].shape[0]
        Z[start:start + size, i] = 1.0
        start += size

    fpca.fit_mg(X, Z)


def calc_projection_matrix_mgmd_cross(data, fpca):
    Xs, Zs = [], []

    cross = list(product(*[data[p].keys() for p in data]))

    for gid, comb in enumerate(cross):
        xs = []

        for pi, protect in enumerate(data):
            xs.append(data[protect][comb[pi]])

        X_comb = torch.cat(xs, dim=0)

        Xs.append(X_comb)
        Zs.append(torch.full((X_comb.shape[0],), gid))

    X = torch.cat(Xs, dim=0)
    Zid = torch.cat(Zs).long()

    Z = torch.zeros(X.shape[0], len(cross)).type_as(X)
    Z[torch.arange(X.shape[0]), Zid] = 1.0

    fpca.fit_mg(X, Z)
