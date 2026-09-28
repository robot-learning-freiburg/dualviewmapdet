# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0
import os
from collections import defaultdict, Counter
from itertools import combinations
from typing import Dict, Set, Tuple, List

def _iter_tile_dirs(root: str):
    """Yield absolute paths to tile dirs named like '<x>_<y>' under root."""
    if not os.path.isdir(root):
        return
    for name in os.listdir(root):
        p = os.path.join(root, name)
        if os.path.isdir(p) and "_" in name:
            yield p

def _parse_scene_from_filename(fname: str) -> str:
    """
    Extract scene_token from '<scene_token>_<timestamp>.npy'.
    Assumes scene_token has no underscores (true for nuScenes tokens).
    """
    base = os.path.splitext(os.path.basename(fname))[0]
    if "_" not in base:
        return ""
    scene_token, _ = base.rsplit("_", 1)
    return scene_token

def compute_scene_overlap_stats_hot_tiles_and_counts(
    save_dir: str,
    map_location: str,
) -> Tuple[
    Dict[str, Set[str]],              # overlaps: scene -> set(other scenes)
    Dict[int, int],                   # histogram: k -> #scenes with exactly k overlaps
    List[Tuple[str, int, List[str]]], # hot_tiles: (tile_dir, num_scenes_in_tile, sorted_scenes)
    Dict[str, int],                   # per_scene_overlap_counts: scene -> degree
    int,                              # total_tiles
    int                               # total_scenes
]:
    """
    Scan tiles under save_dir/map_location and compute:
      - Overlap graph and histogram (as before)
      - Tiles with maximum distinct scenes ("hot tiles")
      - Per-scene overlap counts
      - Totals: number of tiles and number of scenes
    """
    root = os.path.join(save_dir, map_location)

    # tile_dir -> set(scenes)
    tile_scenes: Dict[str, Set[str]] = defaultdict(set)
    all_scenes: Set[str] = set()

    for tile_dir in _iter_tile_dirs(root):
        scenes_here: Set[str] = set()
        for f in os.listdir(tile_dir):
            if f.endswith(".npy"):
                scene = _parse_scene_from_filename(f)
                if scene:
                    scenes_here.add(scene)
                    all_scenes.add(scene)
        if scenes_here:
            tile_scenes[tile_dir] |= scenes_here

    # Overlap graph
    overlaps: Dict[str, Set[str]] = {s: set() for s in all_scenes}
    for scenes in tile_scenes.values():
        if len(scenes) >= 2:
            for a, b in combinations(scenes, 2):
                overlaps[a].add(b)
                overlaps[b].add(a)

    # Per-scene overlap degrees and histogram
    per_scene_overlap_counts: Dict[str, int] = {s: len(nbrs) for s, nbrs in overlaps.items()}
    degree_counts = Counter(per_scene_overlap_counts.values())
    histogram: Dict[int, int] = dict(sorted(degree_counts.items()))

    # Hot tiles (max distinct scenes in a single tile)
    if tile_scenes:
        counts = {td: len(s) for td, s in tile_scenes.items()}
        max_overlap = max(counts.values())
        hot_tiles = [
            (td, counts[td], sorted(tile_scenes[td]))
            for td in counts if counts[td] == max_overlap
        ]
        hot_tiles.sort(key=lambda x: x[0])
    else:
        hot_tiles = []

    total_tiles = len(tile_scenes)
    total_scenes = len(all_scenes)

    return overlaps, histogram, hot_tiles, per_scene_overlap_counts, total_tiles, total_scenes

# ---- Example usage ----
if __name__ == "__main__":
    save_dir = "data/av2_gmaps/lidar_map"
    map_location = "DTW" # "singapore-hollandvillage" # "boston-seaport"  #"singapore-queenstown"

    overlaps, hist, hot_tiles, per_scene_counts, total_tiles, total_scenes = \
        compute_scene_overlap_stats_hot_tiles_and_counts(save_dir, map_location)

    print("Totals:")
    print(f"  tiles: {total_tiles}")
    print(f"  scenes: {total_scenes}")

    isolated = [s for s, nbrs in overlaps.items() if len(nbrs) == 0]
    print(f"\nIsolated scenes (no overlap): {len(isolated)}")

    print("\nOverlap histogram (k_overlaps -> num_scenes):")
    for k, n in hist.items():
        print(f"  {k} -> {n}")

    print("\nPer-scene overlap counts (scene -> degree):")
    for s, k in sorted(per_scene_counts.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {s} -> {k}")

    print("\nTiles with MOST overlap (tile_dir, num_scenes, scenes):")
    if not hot_tiles:
        print("  (none)")
    else:
        for td, nscenes, scenes in hot_tiles:
            print(f"  {td} | {nscenes} scenes | {scenes}")
