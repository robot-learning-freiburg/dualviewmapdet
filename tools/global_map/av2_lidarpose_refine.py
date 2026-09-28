# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Argoverse2 sequence-level global pose refinement from tile-based global maps.

Pipeline:
  1. Scan tiles in:  save_dir / map_location / "{x0}_{y0}" / "{scene_token}_{timestamp}.npy"
     - Group tiles by sequence (scene_token).
     - Record which tiles each sequence covers.
  2. Load original lidar2global (city_SE3_ego_lidar_t) from av2_train_infos.pkl.
     - For each sequence, collect all frame tokens (str(lidar_timestamp_ns)) and 4x4 lidar2global.
     - Use the first lidar2global of each sequence as the "root pose".
  3. Find candidate sequence pairs for KISS-Matcher:
     - A pair (A,B) is a candidate if they share at least `min_overlap_tiles` tiles.
     - Report sequences that have no candidate partner.
  4. For each candidate pair:
     - Load both sequences' accumulated static global point clouds by stacking all their tiles.
       * Tiles are already in the (old/original) global frame.
       * Filter static points by pcl[:, 7] < 0 (dynamic objects removed).
     - Run KISS-Matcher to estimate rigid transform T_BA (A -> B) in the old global frame.
       * Threshold and inlier count are parameters.
       * Optionally save a debug merged cloud aligned into B frame:
             N x 4 array with xyzc, where c=1 for A (transformed into B) and c=2 for B.
  5. Pose graph optimization over sequence roots:
     - One node per sequence: X_s in the refined global frame (*).
     - Cross-sequence BetweenFactorPose3 edges from KISS-Matcher results.
     - No intra-sequence edges (Argoverse2 intra alignment is already good).
     - For each connected component of this graph, add a PriorFactorPose3 that fixes one
       sequence root to its original old global pose (gauge fixing).
  6. Export:
     - Per-frame refined T_global_lidar_star:
         T_star_root_s = optimized pose of sequence root.
         T_old_root_s = old pose of sequence root.
         For any frame f in sequence s with old pose G_f_old:
             T_global_lidar_star[f] = T_star_root_s @ T_old_root_s^{-1} @ G_f_old
     - Concatenate all sequences' static point clouds, transformed into the refined global frame,
       and save as a single global map.
         global_pcl_star.npy
     - Save dict {frame_token -> 4x4} as T_global_lidar_star.npy
"""

import os
import os.path as osp
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
from tqdm import tqdm

import mmcv
import gtsam
from gtsam import Pose3, Rot3, Point3

import kiss_matcher

from kiss_icp.kiss_icp import KissICP
from kiss_icp.config.parser import load_config, KISSConfig


# --------------------------- CONFIG ---------------------------------


@dataclass
class AV2PoseGraphConfig:
    # Input tiles root (produced by GlobalMapGenerator.save_pointcloud_tiles)
    save_dir: str

    # City / map key (e.g. "PIT", "MIA", ...)
    map_location: str

    # Annotation file with AV2 infos (contains city_SE3_ego_lidar_t etc.)
    ann_files: List[str]

    # Output root directory
    out_dir: str

    # Minimum number of overlapping tiles to consider two sequences a candidate pair
    min_overlap_tiles: int = 1

    # Use whole sequence point clouds for KISS-Matcher (instead of only overlapping tiles)
    use_whole_sequence_for_kiss_matcher: bool = True

    # KISS-Matcher parameters
    kiss_matcher_voxel_size: float = 2.0  # parameter to KISSMatcherConfig
    kiss_min_inliers: int = 15        # minimum final inlier count to accept registration

    # Debug: save merged clouds from KISS-Matcher (A aligned to B)
    debug_save_matches: bool = True

    # GTSAM noise (std-dev) for priors and pairwise constraints
    prior_sigmas_rpy_txyz: Tuple[float, float, float, float, float, float] = (
        np.deg2rad(1.0),
        np.deg2rad(1.0),
        np.deg2rad(1.0),
        10.0,
        10.0,
        10.0,
    )
    pair_sigmas_rpy_txyz: Tuple[float, float, float, float, float, float] = (
        np.deg2rad(3.0),
        np.deg2rad(3.0),
        np.deg2rad(4.0),
        1.0,
        1.0,
        1.0,
    )


# --------------------------- UTILS ----------------------------------


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def to_pose3(T: np.ndarray) -> Pose3:
    R = T[:3, :3]
    t = T[:3, 3]
    return Pose3(Rot3(R), Point3(*t))


def pose3_to_matrix(P: Pose3) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = P.rotation().matrix()
    T[:3, 3] = np.asarray(P.translation())
    return T


def build_noise_diag(
    sigmas_rpy_txyz: Tuple[float, float, float, float, float, float]
) -> gtsam.noiseModel.Diagonal:
    rx, ry, rz, tx, ty, tz = sigmas_rpy_txyz
    return gtsam.noiseModel.Diagonal.Sigmas(
        np.array([rx, ry, rz, tx, ty, tz], dtype=float)
    )


def remove_nan_from_point_cloud(point_cloud: np.ndarray) -> np.ndarray:
    """Drop NaN/Inf values; ensure contiguous."""
    mask = np.isfinite(point_cloud).all(axis=1)
    return np.ascontiguousarray(point_cloud[mask])


def _iter_tile_dirs(root: str):
    """Yield (tile_dir_path, x0, y0) for subdirs named '<x0>_<y0>'."""
    if not osp.isdir(root):
        return
    for name in os.listdir(root):
        p = osp.join(root, name)
        if not osp.isdir(p):
            continue
        if "_" not in name:
            continue
        try:
            x_str, y_str = name.split("_", 1)
            x0 = float(x_str)
            y0 = float(y_str)
        except Exception:
            continue
        yield p, x0, y0

def trans_from_vec(t: np.ndarray) -> np.ndarray:
    """Homogeneous translation from 3-vector."""
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = np.asarray(t, dtype=np.float64)
    return T

class MyKISSICP:
    """
    Thin wrapper around KISS-ICP for two-frame refinement.

    Usage:
        icp = MyKISSICP(voxel_size=1.0, min_range=0.0, max_range=100.0)
        icp.register_first_frame(B_centered, 0.0)
        T_BA_refine_local = icp.register(A_centered, 1.0)  # B->A in centered coords
    """
    def __init__(
        self,
        voxel_size: float = 1.0,
        min_range: float = 0.0,
        max_range: float = 100.0,
    ):
        cfg = load_config(config_file=None, max_range=None)  # default config
        # Our scans are already motion-compensated / in global frame.
        cfg.data.deskew = False
        cfg.data.min_range = float(min_range)
        cfg.data.max_range = float(max_range)
        cfg.mapping.voxel_size = float(voxel_size)
        self.config: KISSConfig = cfg
        self.odom = KissICP(config=self.config)

    def register_first_frame(self, pts_xyz: np.ndarray, timestamp: float) -> None:
        A = remove_nan_from_point_cloud(pts_xyz)
        if A.shape[0] < 500:
            raise Exception(f"KISS-ICP: first frame too sparse: {A.shape[0]} points")
        tA = np.array(timestamp, dtype=np.float64)
        self.odom.register_frame(A, tA)

    def register(self, pts_xyz: np.ndarray, timestamp: float) -> np.ndarray:
        """
        Returns 4x4 T_AB (B -> A), where A is the last frame registered,
        and B is the current frame.
        """
        B = remove_nan_from_point_cloud(pts_xyz)
        if B.shape[0] < 500:
            raise Exception(f"KISS-ICP: frame too sparse: {B.shape[0]} points")
        tB = np.array(timestamp, dtype=np.float64)

        self.odom.register_frame(B, tB)
        T_AB = np.asarray(self.odom.last_delta, dtype=np.float64)

        if not np.isfinite(T_AB).all() or T_AB.shape != (4, 4):
            raise Exception("KISS-ICP: last_delta is not a valid 4x4 transform")
        return T_AB

# --------------------------- DATA STRUCTS ---------------------------


@dataclass
class SequenceData:
    scene_token: str
    # Tiles covered by this sequence (set of (x0, y0))
    tiles: Set[Tuple[float, float]]
    # Paths of tile .npy files for this sequence
    tile_paths: List[str]
    # Per-frame lidar2global poses (old/original) sorted by timestamp_ns
    frame_tokens: List[str]  # token = str(lidar_timestamp_ns)
    frame_timestamps_ns: List[int]
    frame_T_global_lidar: List[np.ndarray]  # list of 4x4


# --------------------------- STEP 1: LOAD INFOS & TILES -------------


def load_av2_infos(ann_files: List[str]) -> List[dict]:
    """Load av2_train_infos.pkl (mmcv format) and return the list of infos."""
    file_client = mmcv.FileClient(backend='disk')

    infos = []
    for ann_file in ann_files:
        # load annotations
        if hasattr(file_client, 'get_local_path'):
            with file_client.get_local_path(ann_file) as local_path:
                data = mmcv.load(open(local_path, 'rb'), file_format="pkl")
            data_infos = data["infos"]
            infos.extend(data_infos)
    return infos


def build_sequences_from_tiles_and_infos(
    cfg: AV2PoseGraphConfig,
) -> List[SequenceData]:
    """
    - Scan tiles for this map_location and group by scene_token.
    - Use AV2 infos to attach per-frame lidar2global to each scene_token present in tiles.
    """
    tiles_root = osp.join(cfg.save_dir, cfg.map_location)

    # First pass: scan tiles and collect which sequences exist & which tiles they occupy.
    scene_to_tiles: Dict[str, Set[Tuple[float, float]]] = {}
    scene_to_tile_paths: Dict[str, List[str]] = {}

    if not osp.isdir(tiles_root):
        raise FileNotFoundError(f"Tiles root '{tiles_root}' does not exist")

    print(f"Scanning tiles in {tiles_root}")
    for tile_dir, x0, y0 in _iter_tile_dirs(tiles_root):
        # Each tile_dir contains files like "<scene_token>_<timestamp>.npy"
        for fname in os.listdir(tile_dir):
            if not fname.endswith(".npy"):
                continue
            fpath = osp.join(tile_dir, fname)
            base = fname[:-4]
            if "_" not in base:
                # Unexpected
                raise ValueError(f"Unexpected format in {fpath}, got {base}")
            scene_token, _ = base.split("_", 1)

            scene_to_tiles.setdefault(scene_token, set()).add((x0, y0))
            scene_to_tile_paths.setdefault(scene_token, []).append(fpath)

    scene_tokens = sorted(scene_to_tiles.keys())
    print(f"Found {len(scene_tokens)} sequences with tiles in map_location={cfg.map_location}")

    # Second pass: load AV2 infos and attach lidar2global per frame
    infos = load_av2_infos(cfg.ann_files)

    # Build map: scene_token -> list of (timestamp_ns, token_str, T_global_lidar)
    tmp_seq_frames: Dict[str, List[Tuple[int, str, np.ndarray]]] = {}

    print(f"Scanning AV2 infos from {cfg.ann_files}")
    for info in tqdm(infos, desc="Infos"):
        scene_id = info["scene_id"]
        if scene_id not in scene_to_tiles:
            # We only care about sequences present in the tiles
            continue

        # lidar_timestamp_ns
        ts_ns = info["lidar_timestamp_ns"]
        token_str = str(info["lidar_timestamp_ns"])

        # city_SE3_ego_lidar_t with .rotation (3x3) and .translation (3,)
        city_SE3_ego = info["city_SE3_ego_lidar_t"]
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = np.asarray(city_SE3_ego.rotation, dtype=np.float64)
        T[:3, 3] = np.asarray(city_SE3_ego.translation, dtype=np.float64)

        tmp_seq_frames.setdefault(scene_id, []).append((ts_ns, token_str, T))

    # Build SequenceData objects
    sequences: List[SequenceData] = []
    for scene_token in scene_tokens:
        if scene_token not in tmp_seq_frames:
            print(
                f"[WARN] scene_token={scene_token} has tiles but no frames in {cfg.ann_files}. "
                f"Skipping this sequence."
            )
            continue

        frames = sorted(tmp_seq_frames[scene_token], key=lambda x: x[0])
        ts_ns_list = [x[0] for x in frames]
        token_list = [x[1] for x in frames]
        T_list = [x[2] for x in frames]

        seq = SequenceData(
            scene_token=scene_token,
            tiles=scene_to_tiles[scene_token],
            tile_paths=scene_to_tile_paths[scene_token],
            frame_tokens=token_list,
            frame_timestamps_ns=ts_ns_list,
            frame_T_global_lidar=T_list,
        )
        sequences.append(seq)

    print(f"Built {len(sequences)} sequences with frames + tiles")
    return sequences


# --------------------------- STEP 2: CANDIDATE PAIRS -----------------


def find_candidate_pairs(
    sequences: List[SequenceData], min_overlap_tiles: int
) -> List[Tuple[int, int]]:
    """
    Return list of index pairs (i, j) of sequences that share at least `min_overlap_tiles` tiles.
    Also print stats about sequences with no candidates.
    """
    n = len(sequences)
    # Pre-extract tile sets
    tile_sets = [seq.tiles for seq in sequences]

    candidates: List[Tuple[int, int]] = []
    has_candidate: Set[int] = set()

    print(f"Finding candidate pairs with at least {min_overlap_tiles} overlapping tiles...")
    for i in range(n):
        for j in range(i + 1, n):
            overlap = len(tile_sets[i].intersection(tile_sets[j]))
            if overlap >= min_overlap_tiles:
                candidates.append((i, j))
                has_candidate.add(i)
                has_candidate.add(j)

    seq_without_candidate = [idx for idx in range(n) if idx not in has_candidate]
    print(f"Total candidate pairs: {len(candidates)}")
    print(f"Sequences without any candidate: {len(seq_without_candidate)} / {n}")
    if len(seq_without_candidate) > 0:
        print("Scene tokens with no candidates:")
        for idx in seq_without_candidate:
            print(f"  - {sequences[idx].scene_token}")

    return candidates


# --------------------------- STEP 3: KISS-MATCHER ALIGNMENT ----------


def load_sequence_static_cloud(seq: SequenceData) -> np.ndarray:
    """
    Load all tiles for a sequence and stack static points (pcl[:, 7] < 0).
    Returns N x 3 array in the OLD/global frame.
    """
    pts_all = []
    for path in seq.tile_paths:
        pcl = np.load(path)  # shape (P, 3 + C)
        if pcl.shape[1] < 8:
            raise ValueError(
                f"Expected pcl with >=8 channels (xyz...class) but got shape {pcl.shape} in {path}"
            )
        # Static points have class < 0 (as per GlobalMapGenerator)
        mask_static = pcl[:, 7] < 0
        #if not mask_static.all():
        #    print("There is at least one False")
        pts = pcl[mask_static, :3]
        pts_all.append(pts)

    return np.vstack(pts_all).astype(np.float64)

def load_sequence_static_cloud_for_tiles(
    seq: SequenceData,
    tiles_subset: Set[Tuple[float, float]],
) -> np.ndarray:
    """
    Load only tiles whose (x0, y0) are in tiles_subset.
    """
    pts_all = []
    # Build map from tile origin to paths once to avoid repeated scanning
    tile_origin_to_paths: Dict[Tuple[float, float], List[str]] = {}
    for path in seq.tile_paths:
        # path: .../map_location/x0_y0/scene_token_timestamp.npy
        tile_dir = osp.dirname(path)
        tile_name = osp.basename(tile_dir)
        if "_" not in tile_name:
            raise Exception(f"Invalid tile_name: {tile_name}")
        x_str, y_str = tile_name.split("_", 1)
        x0 = float(x_str)
        y0 = float(y_str)
        tile_origin_to_paths.setdefault((x0, y0), []).append(path)

    for (x0, y0) in tiles_subset:
        for path in tile_origin_to_paths.get((x0, y0), []):
            pcl = np.load(path)
            if pcl.shape[1] < 8:
                raise Exception("Expected pcl with >=8 channels (xyz...class)")
            mask_static = pcl[:, 7] < 0
            pts = pcl[mask_static, :3]
            if pts.size == 0:
                continue
            pts_all.append(pts)

    if not pts_all:
        return np.zeros((0, 3), dtype=np.float64)
    return np.vstack(pts_all).astype(np.float64)


def cross_sequence_alignment(
    cfg: AV2PoseGraphConfig,
    sequences: List[SequenceData],
    candidate_pairs: List[Tuple[int, int]],
) -> Dict[Tuple[int, int], np.ndarray]:
    """
    Estimate rigid transforms between candidate sequence pairs with KISS-Matcher.

    Returns:
        edges_T: dict with keys (i, j) and values T_ij (4x4), where:
          - i, j are indices into the `sequences` list (i < j).
          - T_ij is the transform from sequence i's cloud to sequence j's cloud (A->B),
            all in the old/original global frame.

    If cfg.debug_save_matches is True, also save merged point clouds for each accepted pair.
    """
    edges_T: Dict[Tuple[int, int], np.ndarray] = {}

    # Pre-load static clouds once per sequence
    if cfg.use_whole_sequence_for_kiss_matcher:
        seq_clouds: Dict[int, np.ndarray] = {}
        print("Loading static point clouds for all sequences (from tiles)...")
        for idx, seq in enumerate(tqdm(sequences, desc="Sequences")):
            pts = load_sequence_static_cloud(seq)
            pts = remove_nan_from_point_cloud(pts)
            seq_clouds[idx] = pts
            print(f"Sequence {idx} ({seq.scene_token}): {pts.shape[0]} static points")

    print("Running KISS-Matcher for candidate pairs...")
    out_matches_dir = osp.join(cfg.out_dir, cfg.map_location, "matches_debug")
    if cfg.debug_save_matches:
        ensure_dir(out_matches_dir)

    for (i, j) in tqdm(candidate_pairs, desc="KISS-Matcher pairs"):
        seq_i = sequences[i]
        seq_j = sequences[j]

        if cfg.use_whole_sequence_for_kiss_matcher:
            A = seq_clouds[i]
            B = seq_clouds[j]
        else:
            # Use only common tiles between sequences
            common_tiles = sequences[i].tiles.intersection(sequences[j].tiles)
            if len(common_tiles) == 0:
                print(f"No common tiles for pair (i={i}, j={j}), skipping.")
                continue

            A = load_sequence_static_cloud_for_tiles(sequences[i], common_tiles)
            B = load_sequence_static_cloud_for_tiles(sequences[j], common_tiles)

        if A.shape[0] < 3000 or B.shape[0] < 3000:
            print(
                f"Skip pair (i={i}, j={j}) due to sparse clouds: "
                f"A: {A.shape[0]}, B: {B.shape[0]}"
            )
            continue

        # Configure KISS-Matcher
        params = kiss_matcher.KISSMatcherConfig(cfg.kiss_matcher_voxel_size)
        matcher = kiss_matcher.KISSMatcher(params)

        # Estimate transform A->B in old global frame
        result = matcher.estimate(A, B)
        num_final_inliers = matcher.get_num_final_inliers()
        matcher.print()
        print(f"num_final_inliers: {num_final_inliers}")

        # For debugging
        #fail_names = []
        #debug_name = f"{i}_{seq_i.scene_token}__{j}_{seq_j.scene_token}_pcl_marked.npy"
        #if debug_name in fail_names:
        #    print("In fail")

        if num_final_inliers < cfg.kiss_min_inliers:
            print(
                f"Registration might have failed for pair (i={i}, j={j}); "
                f"num_final_inliers={num_final_inliers}, threshold={cfg.kiss_min_inliers}. Skipping."
            )
            continue

        T_BA = np.eye(4, dtype=np.float64)
        T_BA[:3, :3] = np.asarray(result.rotation, dtype=np.float64)
        T_BA[:3, 3] = np.asarray(result.translation, dtype=np.float64)
        T_BA_kiss_matcher = T_BA

        if not np.isfinite(T_BA).all():
            print(f"Non-finite transform for pair (i={i}, j={j}), skipping.")
            continue

        # --- 2) KISS-ICP refinement on the FULL static clouds ---
        # Transform A into (approximate) B frame using T_BA
        ones = np.ones((A.shape[0], 1), dtype=np.float64)
        A_h = np.hstack([A.astype(np.float64), ones]).T  # (4, NA)
        A_in_B = (T_BA @ A_h).T[:, :3]

        # Center both clouds around the mean of B
        B_mean = np.mean(B, axis=0)
        A_centered = A_in_B - B_mean
        B_centered = B - B_mean

        # Run KISS-ICP refinement (B_centered as reference, A_centered as current)
        try:
            icp_refine = MyKISSICP(voxel_size=1.0, min_range=0.0, max_range=100.0)
            icp_refine.register_first_frame(B_centered, 0.0)
            # Returns T_BA_refine_local: transform from B_centered -> A_centered
            T_BA_refine_local = icp_refine.register(A_centered, 1.0)

            # Convert local-centered transform back to original B frame
            T_to_center = trans_from_vec(-B_mean)
            T_from_center = trans_from_vec(B_mean)
            T_BA_refine_global = T_from_center @ T_BA_refine_local @ T_to_center

            # Compose refinement with coarse estimate:
            #   T_BA (final) = T_BA_refine_global @ T_BA
            T_BA = T_BA_refine_global @ T_BA
        except Exception as e:
            print(f"KISS-ICP refinement failed for pair (i={i}, j={j}): {e}")
            # Keep coarse T_BA from KISS-Matcher

        if not np.isfinite(T_BA).all():
            print(f"Non-finite transform after refinement for pair (i={i}, j={j}), skipping.")
            continue

        # Debug: save merged cloud in B frame
        if cfg.debug_save_matches:
            ones = np.ones((A.shape[0], 1), dtype=np.float64)
            A_h = np.hstack([A.astype(np.float64), ones]).T  # (4, NA)
            A_in_B = (T_BA @ A_h).T[:, :3]

            B_marked = np.concatenate(
                [B.astype(np.float64), 2 * np.ones((B.shape[0], 1), dtype=np.float64)],
                axis=1,
            )
            A_marked = np.concatenate(
                [A_in_B.astype(np.float64), 1 * np.ones((A_in_B.shape[0], 1), dtype=np.float64)],
                axis=1,
            )
            pcl_marked = np.concatenate([A_marked, B_marked], axis=0)

            #fail_names = []
            debug_name = f"{i}_{seq_i.scene_token}__{j}_{seq_j.scene_token}_pcl_marked.npy"
            #if debug_name in fail_names:
            #    print("In fail")
            np.save(osp.join(out_matches_dir, debug_name), pcl_marked)

        # After refinement, T_BA is A->B
        R = T_BA[:3, :3]
        t = T_BA[:3, 3]

        # Optional: check that rotation is "small"; if not, treat as failure
        angle = np.arccos(np.clip((np.trace(R) - 1) * 0.5, -1.0, 1.0))
        angle_deg = np.rad2deg(angle)
        if angle_deg > 3.0:  # or even 1°
            print(f"Large rotation ({angle_deg:.2f} deg) for pair (i={i}, j={j}); rejecting edge.")
            continue

        # Enforce identity rotation (translation-only edge)
        # T_BA[:3, :3] = np.eye(3, dtype=np.float64)

        # Store edge: transform from i->j (A->B) in old global frame
        edges_T[(i, j)] = T_BA

    print(f"Accepted {len(edges_T)} KISS-Matcher edges.")
    return edges_T


# --------------------------- STEP 4: POSE GRAPH OPTIMIZATION ---------


def compute_connected_components(num_nodes: int, edges: List[Tuple[int, int]]) -> List[List[int]]:
    """Simple DFS-based connected components on an undirected graph."""
    adj: Dict[int, Set[int]] = {i: set() for i in range(num_nodes)}
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)

    visited: Set[int] = set()
    components: List[List[int]] = []

    for start in range(num_nodes):
        if start in visited:
            continue
        stack = [start]
        comp: List[int] = []
        while stack:
            u = stack.pop()
            if u in visited:
                continue
            visited.add(u)
            comp.append(u)
            for v in adj[u]:
                if v not in visited:
                    stack.append(v)
        components.append(comp)

    return components


def optimize_sequence_roots(
    cfg: AV2PoseGraphConfig,
    sequences: List[SequenceData],
    edges_T: Dict[Tuple[int, int], np.ndarray],
) -> List[np.ndarray]:
    """
    Pose-graph optimization over sequence roots.

    Nodes:
        X_s in refined global frame (*), one per sequence index s.

    Measurements:
        - Cross-sequence BetweenFactorPose3 for each accepted KISS-Matcher edge (i, j):
              T_ij : A->B in old global frame, where A=i, B=j.
          Let G_i, G_j be the old global root poses (first frame of seq i/j).
          We define:
              Z_ij = G_i^{-1} * T_ij^{-1} * G_j
          and add BetweenFactorPose3(i, j, Z_ij, noise).
          As in the nuScenes script, this is consistent with GTSAM's convention
          (measurement is approximately X_i^{-1} * X_j).

        - One PriorFactorPose3 per connected component for gauge fixing:
              For each connected component C, choose a representative node r in C
              and fix it at its initial pose (old global).

    Returns:
        roots_star: list of 4x4 matrices T_star_root_s (one per sequence s).
    """
    num_seqs = len(sequences)
    if num_seqs == 0:
        return []

    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()

    # Root poses in old global frame: first frame of each sequence
    root_T_global_old: List[np.ndarray] = []
    for s_idx, seq in enumerate(sequences):
        if len(seq.frame_T_global_lidar) == 0:
            raise ValueError(f"Sequence {seq.scene_token} has no frames.")
        T_root = np.asarray(seq.frame_T_global_lidar[0], dtype=np.float64)
        root_T_global_old.append(T_root)
        initial.insert(s_idx, to_pose3(T_root))

    # Build connected components from cross-sequence edges
    edge_pairs = list(edges_T.keys())
    components = compute_connected_components(num_seqs, edge_pairs)

    print(f"Pose graph has {num_seqs} nodes and {len(edge_pairs)} edges.")
    print(f"Found {len(components)} connected components.")

    # One prior per connected component, fix at old global pose
    prior_noise = build_noise_diag(cfg.prior_sigmas_rpy_txyz)
    for comp in components:
        if len(comp) == 0:
            continue
        root_idx = min(comp)  # deterministic choice
        graph.add(
            gtsam.PriorFactorPose3(
                root_idx, initial.atPose3(root_idx), prior_noise
            )
        )

    # Cross-sequence BetweenFactorPose3 edges
    pair_noise = build_noise_diag(cfg.pair_sigmas_rpy_txyz)
    for (i, j), T_ij in edges_T.items():
        G_i = root_T_global_old[i]
        G_j = root_T_global_old[j]

        # Measurement Z_ij such that:
        #    Z_ij ≈ X_i^{-1} * X_j
        # Using old root poses and KISS-Matcher transform as in nuScenes script:
        Z_ij = np.linalg.inv(G_i) @ np.linalg.inv(T_ij) @ G_j

        graph.add(
            gtsam.BetweenFactorPose3(
                i, j, to_pose3(Z_ij), pair_noise
            )
        )

    # Optimize
    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(200)
    params.setVerbosity("ERROR")
    print("Running Levenberg-Marquardt optimizer...")
    result = gtsam.LevenbergMarquardtOptimizer(graph, initial, params).optimize()
    print("Optimization done.")

    # Extract optimized root poses
    roots_star: List[np.ndarray] = []
    for s_idx in range(num_seqs):
        P_star = result.atPose3(s_idx)
        roots_star.append(pose3_to_matrix(P_star))
    return roots_star


# --------------------------- STEP 5: EXPORT --------------------------


def export_final_poses_and_global_map(
    cfg: AV2PoseGraphConfig,
    sequences: List[SequenceData],
    roots_star: List[np.ndarray],
):
    """
    For each sequence s:
        - Old root pose: T_root_old = first frame's lidar2global.
        - New root pose: T_root_star = roots_star[s].
        - For each frame f with old pose G_f_old, define:
             T_star_frame = T_root_star @ T_root_old^{-1} @ G_f_old

    Also:
        - Transform each sequence's static cloud into refined global frame and concatenate
          to a single global map.
    """
    assert len(sequences) == len(roots_star)

    out_map_dir = osp.join(cfg.out_dir, cfg.map_location)
    ensure_dir(out_map_dir)

    final_T_global_lidar: Dict[str, np.ndarray] = {}
    final_T_globalold_to_globalstar: Dict[str, np.ndarray] = {}
    global_pcl_list: List[np.ndarray] = []

    print("Exporting refined per-frame poses and global map...")
    for s_idx, (seq, T_root_star) in enumerate(zip(sequences, roots_star)):
        T_root_old = np.asarray(seq.frame_T_global_lidar[0], dtype=np.float64)
        T_old_to_star = T_root_star @ np.linalg.inv(T_root_old)

        final_T_globalold_to_globalstar[seq.scene_token] = T_old_to_star

        # Per-frame refined poses
        for token, G_f_old in zip(seq.frame_tokens, seq.frame_T_global_lidar):
            G_f_old = np.asarray(G_f_old, dtype=np.float64)
            T_star_frame = T_old_to_star @ G_f_old
            final_T_global_lidar[token] = T_star_frame

        # Transform static cloud into refined global frame
        pts_old = load_sequence_static_cloud(seq)
        if pts_old.shape[0] == 0:
            continue
        ones = np.ones((pts_old.shape[0], 1), dtype=np.float64)
        pts_old_h = np.hstack([pts_old.astype(np.float64), ones]).T  # (4,N)
        pts_star = (T_old_to_star @ pts_old_h).T[:, :3]
        global_pcl_list.append(pts_star)

    # Save refined poses
    np.save(
        osp.join(out_map_dir, "T_globalold_to_globalstar.npy"),
        final_T_globalold_to_globalstar,
    )
    print(f"Saved optimized T_globalold_to_globalstar for {len(final_T_globalold_to_globalstar)} frames.")

    np.save(
        osp.join(out_map_dir, "T_global_lidar_star.npy"),
        final_T_global_lidar,
    )
    print(f"Saved optimized T_global_lidar for {len(final_T_global_lidar)} frames.")

    # Save global point cloud map
    if len(global_pcl_list) > 0:
        global_pcl = np.concatenate(global_pcl_list, axis=0)
        np.save(osp.join(out_map_dir, "globalmap.npy"), global_pcl)
        print(f"Saved global map with {global_pcl.shape[0]} points to globalmap.npy")
    else:
        print("No points in global map (all sequences empty).")


# --------------------------- MAIN -----------------------------------


def main():
    # ----------- EDIT THESE OR WRAP WITH ARGPARSE --------------------
    config = AV2PoseGraphConfig(
        save_dir="data/av2_gmaps/lidar_map/",        # root where tiles live
        map_location="",                     # e.g. "PIT"
        ann_files=["data/av2/av2_train_infos.pkl", "data/av2/av2_val_infos.pkl"],
        out_dir="lidarpose/av2_pose_graph",
    )
    # -----------------------------------------------------------------

    configs = []
    map_locations = ["ATX", "DTW", "MIA", "PAO", "PIT", "WDC"]
    for map_location in map_locations:
        cfg_map = deepcopy(config)
        cfg_map.map_location = map_location
        if map_location in ["ATX", "PAO", "WDC"]:
            cfg_map.kiss_min_inliers = 7
            cfg_map.use_whole_sequence_for_kiss_matcher = False
        elif map_location in ["DTW", "MIA", "PIT"]:
            # Use different setting to be robust to highly overlapping sequences
            cfg_map.kiss_min_inliers = 15
            cfg_map.use_whole_sequence_for_kiss_matcher = True
        else:
            raise ValueError(f"Unknown map location: {map_location}")
        configs.append(cfg_map)

    for cfg in configs:
        ensure_dir(cfg.out_dir)
        print("=== Step 1: Load sequences (tiles + lidar2global) ===")
        sequences = build_sequences_from_tiles_and_infos(cfg)
        if len(sequences) == 0:
            print("No sequences found. Nothing to do.")
            return

        print("\n=== Step 2: Find candidate sequence pairs ===")
        candidate_pairs = find_candidate_pairs(sequences, cfg.min_overlap_tiles)
        if len(candidate_pairs) == 0:
            print("No candidate pairs found. Pose graph will have only priors; "
                  "refined poses will equal original poses.")

        print("\n=== Step 3: Cross-sequence KISS-Matcher alignment ===")
        edges_T = cross_sequence_alignment(cfg, sequences, candidate_pairs)

        print("\n=== Step 4: Pose graph optimization over sequence roots ===")
        roots_star = optimize_sequence_roots(cfg, sequences, edges_T)

        print("\n=== Step 5: Export final global poses and map ===")
        export_final_poses_and_global_map(cfg, sequences, roots_star)

        print(f"\nAll done for map {cfg.map_location}.")


if __name__ == "__main__":
    main()
