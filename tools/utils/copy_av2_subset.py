# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0
#!/usr/bin/env python3
"""
Copy a load_interval-subsampled subset of Argoverse2 sensor data to a target directory.

What it copies:
  - For each scene referenced by infos[::load_interval]:
      - FULL scene folders:
          <split>/<scene_id>/calibration/
          <split>/<scene_id>/map/
      - FULL per-scene files:
          <split>/<scene_id>/annotations.feather
          <split>/<scene_id>/city_SE3_egovehicle.feather
      - Only required sensor files:
          LiDAR:  <split>/<scene_id>/sensors/lidar/<lidar_timestamp_ns>.feather
          Images: whatever is referenced in cam_info["fpath"] (relative to dataset root)

Supports local -> local and local -> remote (user@host:/path) copy.
Uses rsync over ssh if available (fast), otherwise can generate scp commands.
"""

import argparse
import os
import pickle
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Set, Tuple


PER_SCENE_FILES = [
    "annotations.feather",
    "city_SE3_egovehicle.feather",
]


def load_infos(pkl_path: Path) -> List[Dict]:
    """Load AV2 infos from a pickle file with format {'infos': [...]} or directly a list."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict) and "infos" in data:
        return data["infos"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unknown pkl structure in {pkl_path}")


def is_remote_path(dst: str) -> bool:
    """
    Heuristic for remote path: 'user@host:/abs/path'.
    If dst starts with '/', we assume it's local.
    """
    return (":" in dst) and (not dst.startswith("/"))


def split_remote(dst: str) -> Tuple[str, str]:
    """Split 'user@host:/path' into ('user@host', '/path')."""
    userhost, remote_path = dst.split(":", 1)
    return userhost, remote_path


def which(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def run(cmd: List[str], dry_run: bool = False):
    printable = " ".join(cmd)
    if dry_run:
        print(f"[DRY-RUN] {printable}")
        return
    print(f"[RUN] {printable}")
    subprocess.run(cmd, check=True)


def normalize_relpath(p: str) -> str:
    """Normalize to a clean relative path."""
    return str(Path(p))


def ensure_relative_to_root(p: str) -> str:
    """
    Fix: cam_info["fpath"] should be relative.
    If it starts with '/', strip it so rsync/scp work.
    """
    return str(p).lstrip("/")


def collect_required_paths(
    infos: List[Dict],
    split: str,
    load_interval: int,
) -> Tuple[Set[str], Set[Tuple[str, str]]]:
    """
    Returns:
      files_to_copy_rel: relpaths under data-root
      scenes: set of (scene_id, split) pairs
    """
    chosen_infos = infos[::load_interval]

    files_to_copy_rel: Set[str] = set()
    scenes: Set[Tuple[str, str]] = set()

    for info in chosen_infos:
        scene_id = info.get("scene_id", None)
        if scene_id is None:
            continue

        scenes.add((scene_id, split))

        # LiDAR frame path
        ts = info.get("lidar_timestamp_ns", None)
        if ts is not None:
            lidar_rel = f"{split}/{scene_id}/sensors/lidar/{ts}.feather"
            files_to_copy_rel.add(normalize_relpath(lidar_rel))

        # Camera image paths (from cam_info['fpath'])
        cam_infos = info.get("cam_infos", None)
        if isinstance(cam_infos, dict):
            for _, cam_info in cam_infos.items():
                if cam_info is None:
                    continue
                fpath = cam_info.get("fpath", None)
                if fpath is None:
                    continue
                fpath = ensure_relative_to_root(fpath)
                files_to_copy_rel.add(normalize_relpath(fpath))

    # Add per-scene files to copy (independent of load_interval)
    for (scene_id, split) in scenes:
        for fname in PER_SCENE_FILES:
            files_to_copy_rel.add(normalize_relpath(f"{split}/{scene_id}/{fname}"))

    return files_to_copy_rel, scenes


def write_files_from_list(files_rel: Set[str], out_txt: Path):
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(out_txt, "w") as f:
        for p in sorted(files_rel):
            f.write(p + "\n")


def ssh_mkdir_p(dst: str, rel_dir: str, dry_run: bool):
    """Ensure remote directory exists: ssh user@host 'mkdir -p /dst/.../rel_dir'"""
    userhost, remote_root = split_remote(dst)
    remote_full = str(Path(remote_root) / rel_dir)
    run(["ssh", userhost, f"mkdir -p {remote_full}"], dry_run=dry_run)


def copy_with_rsync(
    data_root: Path,
    dst: str,
    files_rel: Set[str],
    scenes: Set[Tuple[str, str]],
    dry_run: bool,
):
    """
    Fast path:
      - copies required sensor files + per-scene feather files using rsync --files-from
      - copies full calibration/ and map/ dirs per scene
    """
    tmp_list = Path(".tmp_av2_copy") / "files_to_copy.txt"
    write_files_from_list(files_rel, tmp_list)

    # 1) Copy selected files (images + lidar frames + per-scene feather)
    rsync_cmd = [
        "rsync",
        "-a",
        "--info=progress2",
        f"--files-from={str(tmp_list)}",
        str(data_root) + "/",      # must end with /
        dst.rstrip("/") + "/",     # dest root
    ]
    run(rsync_cmd, dry_run=dry_run)

    # 2) Copy full calibration/ and map/ folders per scene
    remote = is_remote_path(dst)

    for (scene_id, split) in sorted(scenes):
        cal_src = data_root / split / scene_id / "calibration"
        map_src = data_root / split / scene_id / "map"

        cal_rel_dst = f"{split}/{scene_id}/calibration"
        map_rel_dst = f"{split}/{scene_id}/map"

        if remote:
            ssh_mkdir_p(dst, cal_rel_dst, dry_run=dry_run)
            ssh_mkdir_p(dst, map_rel_dst, dry_run=dry_run)
        else:
            (Path(dst) / cal_rel_dst).mkdir(parents=True, exist_ok=True)
            (Path(dst) / map_rel_dst).mkdir(parents=True, exist_ok=True)

        if cal_src.exists():
            run(
                ["rsync", "-a", str(cal_src) + "/", f"{dst.rstrip('/')}/{cal_rel_dst}/"],
                dry_run=dry_run,
            )
        else:
            print(f"[WARN] Missing: {cal_src}")

        if map_src.exists():
            run(
                ["rsync", "-a", str(map_src) + "/", f"{dst.rstrip('/')}/{map_rel_dst}/"],
                dry_run=dry_run,
            )
        else:
            print(f"[WARN] Missing: {map_src}")


def copy_with_scp_commands(
    data_root: Path,
    dst: str,
    files_rel: Set[str],
    scenes: Set[Tuple[str, str]],
    out_sh: Path,
):
    """
    Generates a bash script with scp commands.
    Slower than rsync, but matches the classic scp workflow.
    """
    out_sh.parent.mkdir(parents=True, exist_ok=True)

    userhost, remote_path = split_remote(dst)

    def mkdir_remote_cmd(rel_dir: str) -> str:
        remote_full = str(Path(remote_path) / rel_dir)
        return f"ssh {userhost} 'mkdir -p {remote_full}'"

    with open(out_sh, "w") as f:
        f.write("#!/usr/bin/env bash\n")
        f.write("set -euo pipefail\n\n")

        # Ensure base exists
        f.write(f"ssh {userhost} 'mkdir -p {remote_path}'\n\n")

        # 1) copy calibration/ and map/ dirs per scene
        for (scene_id, split) in sorted(scenes):
            cal_rel = f"{split}/{scene_id}/calibration"
            map_rel = f"{split}/{scene_id}/map"

            f.write(f"# Scene {split}/{scene_id}\n")
            f.write(mkdir_remote_cmd(cal_rel) + "\n")
            f.write(mkdir_remote_cmd(map_rel) + "\n")
            f.write(f"scp -r '{data_root}/{cal_rel}' '{dst}/{split}/{scene_id}/'\n")
            f.write(f"scp -r '{data_root}/{map_rel}' '{dst}/{split}/{scene_id}/'\n\n")

        # 2) copy selected files (includes per-scene feather files)
        f.write("# Selected files (lidar frames + camera images + per-scene feather)\n")
        for rel in sorted(files_rel):
            rel_parent = str(Path(rel).parent)
            f.write(mkdir_remote_cmd(rel_parent) + "\n")
            f.write(f"scp '{data_root}/{rel}' '{dst}/{rel}'\n")

    os.chmod(out_sh, 0o755)
    print(f"[OK] Wrote SCP script: {out_sh}")


def copy_local_fallback(
    data_root: Path,
    dst_root: Path,
    files_rel: Set[str],
    scenes: Set[Tuple[str, str]],
):
    """Pure python local copy (no ssh)."""
    dst_root.mkdir(parents=True, exist_ok=True)

    # copy full dirs
    for (scene_id, split) in sorted(scenes):
        for folder in ["calibration", "map"]:
            src = data_root / split / scene_id / folder
            if not src.exists():
                print(f"[WARN] Missing: {src}")
                continue

            for item in src.rglob("*"):
                if item.is_dir():
                    continue
                rel = item.relative_to(data_root)
                out_path = dst_root / rel
                out_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, out_path)

    # copy selected files
    for rel in sorted(files_rel):
        src = data_root / rel
        dst = dst_root / rel
        if not src.exists():
            print(f"[WARN] Missing file: {src}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True,
                        help="Argoverse2 root, e.g. /data/av2")
    parser.add_argument("--dst", type=str, required=True,
                        help="Target root. Either local (/path) or remote user@host:/path")
    parser.add_argument("--pkl", type=str, required=True,
                        help="Path to av2_*_infos.pkl (contains dict with key 'infos')")
    parser.add_argument("--split", type=str, required=True,
                        help="Split name used in the filesystem, e.g. train or val")
    parser.add_argument("--load-interval", type=int, default=5,
                        help="Subsampling factor. Will copy infos[::load_interval]")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands, do not copy.")
    parser.add_argument("--force-scp", action="store_true",
                        help="Force generating scp script even if rsync exists.")
    parser.add_argument("--scp-script", type=str, default="copy_subset_scp.sh",
                        help="Where to write scp script if --force-scp is used.")
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    pkl_path = Path(args.pkl).expanduser().resolve()
    dst = args.dst

    infos = load_infos(pkl_path)

    files_rel, scenes = collect_required_paths(
        infos=infos,
        split=args.split,
        load_interval=args.load_interval,
    )

    selected = len(infos[::args.load_interval])
    print(f"[INFO] PKL infos total: {len(infos)}")
    print(f"[INFO] Using load_interval={args.load_interval} -> selected {selected}")
    print(f"[INFO] Scenes referenced: {len(scenes)}")
    print(f"[INFO] Total files to copy (sensor + per-scene feather): {len(files_rel)}")
    print(f"[INFO] Per-scene files included: {PER_SCENE_FILES}")

    remote = is_remote_path(dst)

    # Prefer rsync if possible
    if not args.force_scp and which("rsync"):
        copy_with_rsync(
            data_root=data_root,
            dst=dst,
            files_rel=files_rel,
            scenes=scenes,
            dry_run=args.dry_run,
        )
        print("[OK] Done via rsync.")
        return

    # Remote without rsync -> generate scp script
    if remote:
        out_sh = Path(args.scp_script).expanduser().resolve()
        copy_with_scp_commands(
            data_root=data_root,
            dst=dst,
            files_rel=files_rel,
            scenes=scenes,
            out_sh=out_sh,
        )
        print("[OK] Run the generated script on the SOURCE machine.")
        return

    # Local fallback
    copy_local_fallback(
        data_root=data_root,
        dst_root=Path(dst).expanduser().resolve(),
        files_rel=files_rel,
        scenes=scenes,
    )
    print("[OK] Done via local python copy.")


if __name__ == "__main__":
    main()
