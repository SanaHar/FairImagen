import torch
import os
from pathlib import Path


class BaseProcessor:
    """
    Processor for capturing embeddings during calibration.
    Accumulates embeddings per (protect, category).
    """

    def __init__(self) -> None:
        self.storage = {}

    # ------------------------------------------------------------
    # Generate filename based on protect attribute
    # ------------------------------------------------------------
    def get_feature_filename(self, usermode) -> str:
        protect_str = str(usermode.get("protect", "default")) \
            .replace("[", "") \
            .replace("]", "") \
            .replace("'", "") \
            .replace(",", "_") \
            .replace(" ", "")

        return f"extracted_features_{protect_str}.pt"

    # ------------------------------------------------------------
    # Extraction logic (CALIBRATION MODE)
    # ------------------------------------------------------------
    def extract_embedding(
        self,
        prompt_embeds,
        pooled_prompt_embeds,
        usermode,
        exp_dir,
        **kwargs
    ):
        directory = Path(exp_dir)
        directory.mkdir(parents=True, exist_ok=True)

        filename = self.get_feature_filename(usermode)
        file_path = directory / filename

        # --------------------------------------------------------
        # Load existing calibration file (if exists)
        # --------------------------------------------------------
        if file_path.exists():
            try:
                self.storage = torch.load(file_path, weights_only=True)
            except Exception:
                self.storage = {}

        protect = kwargs.get("protect", "gender")
        cat = kwargs.get("cat", "default")

        # Ensure structure exists
        if protect not in self.storage:
            self.storage[protect] = {}

        # --------------------------------------------------------
        # If category already exists and is Tensor → convert to list
        # (for backward compatibility with old saved files)
        # --------------------------------------------------------
        if cat in self.storage[protect]:
            if isinstance(self.storage[protect][cat], torch.Tensor):
                self.storage[protect][cat] = [
                    self.storage[protect][cat]
                ]
        else:
            self.storage[protect][cat] = []

        # --------------------------------------------------------
        # Append new embedding (ACCUMULATION FIX)
        # --------------------------------------------------------
        self.storage[protect][cat].append(
            pooled_prompt_embeds.detach().cpu()
        )

        # --------------------------------------------------------
        # Convert lists → concatenated tensors before saving
        # --------------------------------------------------------
        for p in self.storage:
            for c in self.storage[p]:
                if isinstance(self.storage[p][c], list):
                    self.storage[p][c] = torch.cat(
                        self.storage[p][c],
                        dim=0
                    )

        torch.save(self.storage, file_path)

        print(f" >>> [CALIBRATION] Captured: {protect} | Category: {cat}")
        print(f" >>> [CALIBRATION] Samples: {self.storage[protect][cat].shape[0]}")
        print(f" >>> [CALIBRATION] File: {file_path}")
        print(f" >>> [CALIBRATION] File size: {os.path.getsize(file_path)} bytes")

    # ------------------------------------------------------------
    # Base processor does not modify embeddings
    # ------------------------------------------------------------
    def modify_embedding(
        self,
        pipe,
        prompt_embeds,
        pooled_prompt_embeds,
        usermode=None,
        exp_dir="."
    ):
        return prompt_embeds, pooled_prompt_embeds
