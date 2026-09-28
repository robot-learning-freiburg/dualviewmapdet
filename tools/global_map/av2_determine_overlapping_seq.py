# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Find AV2 sequences (scene_token) whose *entire* tile coverage is overlapped by other sequences.

Definition (per map_location):
  A sequence S qualifies iff for every tile T that contains at least one file from S,
  the same tile T also contains files from at least `min_other_sequences_per_tile`
  *distinct other* sequences.

Additionally:
  Writes all qualifying scene_tokens (unique across map locations) to a single .txt file,
  one token per line.

Tile layout expected:
  <root_dir>/<map_location>/<x0>_<y0>/<scene_token>_<timestamp>.npy
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import DefaultDict, Dict, List, Optional, Set, Tuple


TileKey = Tuple[str, str]        # (map_location, tile_folder_name)
SceneKey = Tuple[str, str]       # (map_location, scene_token)


@dataclass(frozen=True)
class SceneOverlapStats:
    map_location: str
    scene_token: str
    num_tiles: int
    min_other_overlap: int
    mean_other_overlap: float


def _iter_map_locations(root_dir: str, map_locations: Optional[List[str]]) -> List[str]:
    if map_locations is not None:
        return map_locations

    locs: List[str] = []
    with os.scandir(root_dir) as it:
        for e in it:
            if not e.is_dir():
                continue
            if e.name == "processed":
                continue
            locs.append(e.name)
    locs.sort()
    return locs


def _parse_scene_token_from_npy(filename: str) -> Optional[str]:
    if not filename.endswith(".npy"):
        return None
    stem = filename[:-4]
    if "_" not in stem:
        return stem if stem else None
    return stem.rsplit("_", 1)[0] or None


def build_tile_and_scene_indices(
    root_dir: str,
    map_locations: Optional[List[str]] = None,
) -> Tuple[Dict[TileKey, Set[str]], Dict[SceneKey, Set[TileKey]]]:
    root_dir = os.path.abspath(root_dir)
    locs = _iter_map_locations(root_dir, map_locations)

    tile_to_scenes: Dict[TileKey, Set[str]] = {}
    scene_to_tiles: DefaultDict[SceneKey, Set[TileKey]] = defaultdict(set)

    for loc in locs:
        loc_dir = os.path.join(root_dir, loc)
        if not os.path.isdir(loc_dir):
            continue

        with os.scandir(loc_dir) as tile_it:
            for tile_entry in tile_it:
                if not tile_entry.is_dir():
                    continue

                tile_name = tile_entry.name
                tile_key: TileKey = (loc, tile_name)

                scenes_in_tile: Set[str] = set()
                with os.scandir(tile_entry.path) as file_it:
                    for f in file_it:
                        if not f.is_file():
                            continue
                        scene_token = _parse_scene_token_from_npy(f.name)
                        if scene_token is None:
                            continue
                        scenes_in_tile.add(scene_token)

                if not scenes_in_tile:
                    continue

                tile_to_scenes[tile_key] = scenes_in_tile
                for s in scenes_in_tile:
                    scene_to_tiles[(loc, s)].add(tile_key)

    return tile_to_scenes, scene_to_tiles


def find_fully_overlapped_scenes(
    tile_to_scenes: Dict[TileKey, Set[str]],
    scene_to_tiles: Dict[SceneKey, Set[TileKey]],
    *,
    min_other_sequences_per_tile: int = 1,
) -> List[SceneOverlapStats]:
    if min_other_sequences_per_tile < 1:
        raise ValueError("min_other_sequences_per_tile must be >= 1")

    qualifying: List[SceneOverlapStats] = []

    for (loc, scene_token), tiles in scene_to_tiles.items():
        if not tiles:
            continue

        other_counts: List[int] = []
        ok = True
        for tile_key in tiles:
            scenes_here = tile_to_scenes.get(tile_key, set())
            other = scenes_here.difference({scene_token})
            c = len(other)
            other_counts.append(c)
            if c < min_other_sequences_per_tile:
                ok = False
                break

        if ok:
            min_other = min(other_counts) if other_counts else 0
            mean_other = (sum(other_counts) / len(other_counts)) if other_counts else 0.0
            qualifying.append(
                SceneOverlapStats(
                    map_location=loc,
                    scene_token=scene_token,
                    num_tiles=len(tiles),
                    min_other_overlap=min_other,
                    mean_other_overlap=mean_other,
                )
            )

    qualifying.sort(key=lambda x: (-x.num_tiles, -x.min_other_overlap, x.map_location, x.scene_token))
    return qualifying


def main() -> None:
    # ------------------ EDIT THESE ------------------
    root_dir = "./data/av2_gmaps/lidar_map"
    min_other_sequences_per_tile = 1

    map_locations = None  # e.g. ["ATX", "PAO"] or None for all

    save_json = True
    out_json_path = "./data/av2_gmaps/overlap_sequences.json"

    save_scene_txt = True
    out_scene_txt_path = "./data/av2_gmaps/overlap_scene_tokens.txt"
    # -----------------------------------------------

    print(f"[1/2] Scanning tiles under: {root_dir}")
    tile_to_scenes, scene_to_tiles = build_tile_and_scene_indices(root_dir, map_locations=map_locations)

    print(f"  Found tiles:  {len(tile_to_scenes)}")
    print(f"  Found scenes: {len(scene_to_tiles)}")

    print(f"[2/2] Evaluating fully-overlapped scenes (min_other_sequences_per_tile={min_other_sequences_per_tile})")
    qualifying = find_fully_overlapped_scenes(
        tile_to_scenes,
        scene_to_tiles,
        min_other_sequences_per_tile=min_other_sequences_per_tile,
    )

    # Print summary per map_location
    per_loc: DefaultDict[str, List[SceneOverlapStats]] = defaultdict(list)
    for s in qualifying:
        per_loc[s.map_location].append(s)

    print("\n=== RESULTS ===")
    for loc in sorted(per_loc.keys()):
        lst = per_loc[loc]
        print(f"{loc}: {len(lst)} qualifying scenes")
        for st in lst[:200]:
            print(
                f"  {st.scene_token} | tiles={st.num_tiles} | "
                f"min_other={st.min_other_overlap} | mean_other={st.mean_other_overlap:.2f}"
            )
        if len(lst) > 200:
            print(f"  ... ({len(lst) - 200} more)")

    print(f"\nTotal qualifying scenes: {len(qualifying)}")

    if save_json:
        payload = [
            {
                "map_location": s.map_location,
                "scene_token": s.scene_token,
                "num_tiles": s.num_tiles,
                "min_other_overlap": s.min_other_overlap,
                "mean_other_overlap": s.mean_other_overlap,
            }
            for s in qualifying
        ]
        os.makedirs(os.path.dirname(out_json_path), exist_ok=True)
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved JSON: {out_json_path}")

    if save_scene_txt:
        unique_scene_tokens: Set[str] = {s.scene_token for s in qualifying}
        os.makedirs(os.path.dirname(out_scene_txt_path), exist_ok=True)
        with open(out_scene_txt_path, "w", encoding="utf-8") as f:
            for tok in sorted(unique_scene_tokens):
                f.write(tok + "\n")
        print(f"Saved unique scene tokens TXT ({len(unique_scene_tokens)}): {out_scene_txt_path}")


if __name__ == "__main__":
    main()