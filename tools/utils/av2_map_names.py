# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0
import os
from pathlib import Path

def collect_city_names(train_dir: str):
    train_dir = Path(train_dir)
    city_names = set()

    # recursively search for all *_ground_height_surface____*.npy files
    for npy_path in train_dir.rglob("*_ground_height_surface____*.npy"):
        fname = npy_path.name
        # split at the separator "____"
        parts = fname.split("____")
        if len(parts) < 2:
            continue
        # take everything after "____", remove extension
        city = parts[-1].replace(".npy", "")
        city_names.add(city)

    return sorted(city_names)


if __name__ == "__main__":
    train_root = "./data/av2/train"   # adjust if needed
    cities = collect_city_names(train_root)
    print("Found cities:", cities)