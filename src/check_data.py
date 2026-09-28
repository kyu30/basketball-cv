import configparser
from pathlib import Path

clip_path = Path("./data/sportsmot/dataset/train/train/v_1LwtoLPw2TU_c016")

# read seqinfo
config = configparser.ConfigParser()
config.read(clip_path / "seqinfo.ini")
print(dict(config["Sequence"]))

# check frame count matches
frames = [f for f in (clip_path / "img1").glob("*.jpg") if not f.name.startswith("._")]
print(f"{len(frames)} frames found")

# peek at ground truth format
with open(clip_path / "gt" / "gt.txt") as f:
    for _ in range(5):
        print(f.readline().strip())