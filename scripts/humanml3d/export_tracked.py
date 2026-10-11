"""Turn UniPhys's tracked AMASS into the episodes of HumanML3D's test items, to score what PULSE's tracking leaves of them.

    python scripts/humanml3d/export_tracked.py --test ../../../../data/humanml3d_test.pt \
        --humanml3d ../../../../data/HumanML3D --amass <AMASS>/sequences --out output/humanml3d/pulse.npz

A test item is a motion of AMASS, its mirrored copy, or an annotated part of either. Its episode is
cut from the motion as PULSE tracked it (data/amass_state-action-pairs), the frames HumanML3D takes,
brought to 30 fps; a mirrored item is the mirror image of the tracked motion, which the symmetric
humanoid could have performed as well. Items of HumanAct12 and motions UniPhys left out have no
episode. An episode is 300 steps, the last frame held; the benchmark's scorer applies its fall rule.
"""
import argparse
import csv
import zipfile
from pathlib import Path

import joblib
import numpy as np
import torch
from tqdm import tqdm

from package_training import FPS, HUMANML3D_FPS, SKIPPED_SECONDS

STEPS = 300
MIRRORED = [0, 5, 6, 7, 8, 1, 2, 3, 4, 9, 10, 11, 12, 13, 19, 20, 21, 22, 23, 14, 15, 16, 17, 18]  # the humanoid's bodies, left and right swapped


def part_start(lines, caption, length):
    """Among a motion's caption lines, the frame (20 fps) at which the part of `length` frames begins that `caption` describes."""
    best = None
    for line in lines:
        text, _, start, stop = line.strip().split("#")[:4]
        start, stop = (0. if np.isnan(float(t)) else float(t) for t in (start, stop))
        first, extra = int(start * HUMANML3D_FPS), int(stop * HUMANML3D_FPS) - int(start * HUMANML3D_FPS) - length
        if text == caption and (start, stop) != (0, 0) and extra >= 0 and (best is None or extra < best[1]):
            best = (first, extra)  # the part ends with the motion where its stop lies beyond it
    if best is None:
        raise ValueError(f"No part of {length} frames is annotated with: {caption}")
    return best[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=Path, required=True, help="the test items of the benchmark (humanml3d_test.pt)")
    parser.add_argument("--humanml3d", type=Path, required=True, help="index.csv and texts.zip")
    parser.add_argument("--amass", type=Path, required=True, help="AMASS, for the frame rate of every motion")
    parser.add_argument("--tracked", type=Path, default=Path("data/amass_state-action-pairs"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    test = torch.load(args.test)
    texts = zipfile.ZipFile(args.humanml3d / "texts.zip")
    with open(args.humanml3d / "index.csv") as file:
        rows = {row["new_name"][:-4]: row for row in csv.DictReader(file)}
    episodes, keys, missing = [], [], dict(humanact12=0, untracked=0, short=0)
    for key, motion, captions in zip(tqdm(test["key"]), test["motion"], test["captions"]):
        name = key.split("_")[-1]  # a part is keyed <letter>_<motion>
        row = rows[name[1:] if name.startswith("M") else name]  # Python 3.8
        if "humanact12" in row["source_path"]:
            missing["humanact12"] += 1
            continue
        source = row["source_path"][len("./pose_data/"):-len(".npy")]
        if not args.tracked.joinpath(source + ".pkl").exists():
            missing["untracked"] += 1
            continue
        tracked = np.asarray(joblib.load(args.tracked / (source + ".pkl"))["body_pos"], dtype=np.float64)
        framerate = float(np.load(args.amass / (source + ".npz"))["mocap_framerate"])
        rate = int(framerate / HUMANML3D_FPS) / int(framerate / FPS)  # tracked frames per HumanML3D frame
        first = SKIPPED_SECONDS.get(source.split("/")[0], 0) * HUMANML3D_FPS + int(row["start_frame"])
        if "_" in key:
            first += part_start(texts.read(f"texts/{name}.txt").decode().splitlines(), captions[0]["text"], len(motion))
        # the tracked frames lie framerate / int(framerate / 30) apart per second: read them at 30 fps
        seconds = (len(motion) - 1) / HUMANML3D_FPS
        at = first * rate + np.arange(int(seconds * FPS + 1e-9) + 1) / FPS * framerate / int(framerate / FPS)
        at = at[at <= len(tracked) - 1]
        if len(at) < 30:
            missing["short"] += 1
            continue
        low = np.floor(at).astype(int)
        high, weight = np.minimum(low + 1, len(tracked) - 1), (at - low)[:, None, None]
        body = (tracked[low] * (1 - weight) + tracked[high] * weight)[:STEPS]
        if name.startswith("M"):  # the mirror image in the plane x = 0
            body = body[:, MIRRORED] * [-1., 1, 1]
        body[..., :2] -= body[0, 0, :2]
        episodes.append(np.concatenate((body, np.repeat(body[-1:], STEPS - len(body), 0))).astype(np.float32))
        keys.append(key)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, body_pos=np.stack(episodes), fps=FPS, key=np.array(keys), caption=np.zeros(len(keys), dtype=np.int64))
    print(f"{len(keys)} of {len(test['key'])} test items in {args.out}; without an episode: {missing}")


if __name__ == "__main__":
    main()
