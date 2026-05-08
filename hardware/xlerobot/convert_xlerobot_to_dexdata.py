import os
import json
import subprocess
import pandas as pd
import numpy as np
from loguru import logger
from tqdm import tqdm
import click
from typing import List, Dict, Any, Optional, Tuple, Union

# --- Constants & Types ---
CAMERA_PREFIX = "observation.images."


def trim_video(
    src_path: str,
    dst_path: str,
    start_time: float,
    duration: float) -> bool:
    """
    Trims a video using ffmpeg from start_time with a specific duration.
    Re-encodes to ensure frame accuracy and compatibility.
    """
    # Mac optimization: Use h264_videotoolbox if available for hardware acceleration
    # For now, default to libx264 for maximum compatibility across systems
    cmd = [
        "ffmpeg",
        "-y",               # Overwrite output
        "-ss", str(start_time),
        "-i", src_path,
        "-t", str(duration),
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "23",
        "-c:a", "copy",
        dst_path
    ]

    try:
        subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg error for {src_path}: {e}")
        return False


class DatasetMeta:
    """Handles loading and accessing LeRobot dataset metadata."""

    def __init__(self, dataset_dir: str):
        self.dataset_dir = dataset_dir
        self.meta_dir = os.path.join(dataset_dir, "meta")

        self.info = self._load_info()
        self.tasks = self._load_tasks()

    def _load_info(self) -> Dict[str, Any]:
        info_path = os.path.join(self.meta_dir, "info.json")
        with open(info_path, "r") as f:
            return json.load(f)

    def _load_tasks(self) -> List[str]:
        parquet_path = os.path.join(self.meta_dir, "tasks.parquet")
        df = pd.read_parquet(parquet_path)
        return df.index.astype(str).tolist()

    @property
    def fps(self) -> float:
        return float(self.info["fps"])

    @property
    def camera_folders(self) -> List[str]:
        return sorted([
            key for key, spec in self.info["features"].items()
            if isinstance(spec, dict) and spec.get("dtype") == "video"
        ])


class EpisodeParser:
    """Handles conversion of a single episode's data to DexData format."""
    @staticmethod
    def parse(df: pd.DataFrame,
              task_list: List[str],
              camera_map: Dict[str,
                               str]) -> List[Dict[str,
                                        Any]]:
        data_list = []

        # Prepare camera templates once per episode
        sorted_cams = sorted(camera_map.items())
        cam_templates = [
            {"key": f"images_{idx + 1}", "url": url, "_camera_name": cam_name}
            for idx, (cam_name, url) in enumerate(sorted_cams)
        ]

        if df.empty:
            return []

        prompt = str(task_list[int(df.iloc[0]["task_index"])])

        for _, row in df.iterrows():
            state = row["observation.state"]
            if isinstance(state, np.ndarray):
                state = state.tolist()
            elif not isinstance(state, list):
                state = [state]

            action = row["action"]
            if isinstance(action, np.ndarray):
                action = action.tolist()

            entry = {
                "state": state,
                "prompt": prompt,
                "action": action,
                "extra": {
                    "timestamp": row["timestamp"],
                    "episode_index": int(row["episode_index"]),
                    "frame_index": int(row["frame_index"]),
                    "is_robot": True
                }
            }

            for tpl in cam_templates:
                entry[tpl["key"]] = {
                    "type": "video",
                    "url": tpl["url"],
                    "frame_idx": int(row["frame_index"]),
                    "_camera_name": tpl["_camera_name"]
                }

            data_list.append(entry)
        return data_list


def get_latest_episode_idx(output_dir: str) -> int:
    """Counts existing jsonl files to determine the next episode index."""
    return sum(1 for entry in os.scandir(output_dir)
               if entry.is_file() and entry.name.endswith(".jsonl"))


def save_jsonl(data_list: List[Dict[str, Any]], path: str) -> None:
    """Saves a list of dictionaries to a JSONL file."""
    with open(path, "w") as f:
        for data in data_list:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")


@click.command()
@click.option("-i", "--lerobot_dir", type=str, required=True,
              help="Input LeRobot dataset directory")
@click.option("-o", "--output_dir", type=str, required=True,
              help="Output directory for DexData")
def main(lerobot_dir: str, output_dir: str) -> None:
    task_name = os.path.basename(lerobot_dir.rstrip("/\\"))
    logger.info(f"Starting conversion for task: {task_name}")

    # 1. Load Metadata
    meta = DatasetMeta(lerobot_dir)
    camera_folders = meta.camera_folders
    logger.info(f"Detected cameras: {camera_folders}")

    # 2. Prepare Output Paths
    output_jsonl_dir = os.path.join(output_dir, "jsonl", task_name)
    output_video_dir = os.path.join(output_dir, "video", task_name)
    os.makedirs(output_jsonl_dir, exist_ok=True)
    os.makedirs(output_video_dir, exist_ok=True)

    next_ep_idx = get_latest_episode_idx(output_jsonl_dir)

    # 3. Main Loop over Meta-Episodes
    meta_episodes_dir = os.path.join(meta.meta_dir, "episodes")
    data_path_tpl = meta.info["data_path"]
    video_path_tpl = meta.info["video_path"]

    data_df_cache: Dict[str, Dict[int, pd.DataFrame]] = {}

    # Iterate through metadata chunks
    for chunk_name in sorted(os.listdir(meta_episodes_dir)):
        chunk_path = os.path.join(meta_episodes_dir, chunk_name)
        if not os.path.isdir(chunk_path):
            continue

        meta_files = sorted([f for f in os.listdir(
            chunk_path) if f.endswith(".parquet")])

        for m_file in tqdm(meta_files, desc=f"Chunk {chunk_name}"):
            meta_df = pd.read_parquet(os.path.join(chunk_path, m_file))

            for _, row in meta_df.iterrows():
                ep_idx = int(row['episode_index'])

                # Load corresponding Data Parquet if not cached
                d_rel_path = data_path_tpl.format(
                    chunk_index=int(row['data/chunk_index']),
                    file_index=int(row['data/file_index'])
                )
                d_full_path = os.path.join(lerobot_dir, d_rel_path)

                if d_full_path not in data_df_cache:
                    data_df_cache.clear()  # Primitive memory management
                    df = pd.read_parquet(d_full_path)
                    data_df_cache[d_full_path] = dict(
                        tuple(df.groupby('episode_index')))

                ep_df = data_df_cache[d_full_path].get(ep_idx)
                if ep_df is None:
                    logger.warning(f"Episode {ep_idx} missing in {d_rel_path}")
                    continue

                # Process Episode
                eid = next_ep_idx
                next_ep_idx += 1

                # 3a. Trim Videos
                episode_cam_map = {}
                duration = int(row['length']) / meta.fps

                for cam_folder in camera_folders:
                    v_key = f"videos/{cam_folder}"
                    v_rel_path = video_path_tpl.format(
                        video_key=cam_folder,
                        chunk_index=int(row[f"{v_key}/chunk_index"]),
                        file_index=int(row[f"{v_key}/file_index"])
                    )
                    src_v = os.path.join(lerobot_dir, v_rel_path)

                    clean_name = cam_folder.replace(CAMERA_PREFIX, "")
                    dst_filename = f"episode_{eid:05d}_{clean_name}.mp4"
                    dst_path = os.path.join(output_video_dir, dst_filename)

                    if trim_video(
                            src_v,
                            dst_path,
                            row[f"{v_key}/from_timestamp"],
                            duration):
                        episode_cam_map[clean_name] = f"{task_name}/{dst_filename}"

                # 3b. Parse and Save JSONL
                episode_data = EpisodeParser.parse(ep_df, meta.tasks, episode_cam_map)
                if episode_data:
                    save_jsonl(
                        episode_data,
                        os.path.join(
                            output_jsonl_dir,
                            f"episode_{eid:05d}.jsonl"))


if __name__ == "__main__":
    main()
