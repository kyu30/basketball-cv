import configparser
from pathlib import Path

import pandas as pd

DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "sportsmot"
DATASET_ROOT = DATA_ROOT / "dataset"

# SportsMOT withholds ground truth for the `test` split (competition holdout),
# so EDA and training data can only be built from train/val.
BASKETBALL_SPLITS = ("train", "val")

GT_COLUMNS = [
    "frame",
    "track_id",
    "x",
    "y",
    "w",
    "h",
    "conf",
    "class",
    "visibility",
]


def _basketball_clip_names() -> set[str]:
    with open(DATA_ROOT / "splits_txt" / "basketball.txt") as f:
        return {line.strip() for line in f if line.strip()}


def get_basketball_clips(split: str) -> list[Path]:
    names = _basketball_clip_names()
    split_dir = DATASET_ROOT / split / split  # dataset ships with a doubled folder
    return sorted(d for d in split_dir.iterdir() if d.is_dir() and d.name in names)


def load_seqinfo(clip_path: Path) -> dict:
    config = configparser.ConfigParser()
    config.read(clip_path / "seqinfo.ini")
    seq = dict(config["Sequence"])
    return {
        "name": seq["name"],
        "framerate": int(seq["framerate"]),
        "seqlength": int(seq["seqlength"]),
        "imwidth": int(seq["imwidth"]),
        "imheight": int(seq["imheight"]),
    }


def load_gt(clip_path: Path) -> pd.DataFrame:
    return pd.read_csv(clip_path / "gt" / "gt.txt", header=None, names=GT_COLUMNS)


def build_dataset(
    splits: tuple[str, ...] = BASKETBALL_SPLITS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    clip_rows = []
    annotation_frames = []

    for split in splits:
        for clip_path in get_basketball_clips(split):
            info = load_seqinfo(clip_path)
            clip_rows.append({"split": split, **info})

            gt = load_gt(clip_path)
            gt.insert(0, "clip", info["name"])
            gt.insert(0, "split", split)
            annotation_frames.append(gt)

    clips_df = pd.DataFrame(clip_rows).rename(columns={"name": "clip"})
    annotations_df = pd.concat(annotation_frames, ignore_index=True)
    return clips_df, annotations_df
