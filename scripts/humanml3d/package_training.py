"""Package HumanML3D's training split as tracked motions with captions, in the format of the BABEL training set.

    python scripts/humanml3d/package_training.py --humanml3d ../../../../data/HumanML3D \
        --amass <AMASS>/sequences --retargeted ../../../../data/humanml3d_train_humanact12.pt

A sequence is a motion that PULSE tracked to its end. The AMASS motions come from UniPhys's tracked
AMASS, cut to the frames HumanML3D takes. Their mirrored copies and the motions from HumanAct12 come
from tracking the references of `reference_motions.py`:

    python phc/run_hydra.py learning=im_z_fit env=env_im_vae robot=smpl_humanoid exp_name=pulse_vae_iclr \
        env.motion_file=data/humanml3d_reference/humanml3d_train.pkl env.num_envs=2048 env.cycle_motion=False \
        "env.models=['output/HumanoidIm/phc_3/Humanoid_00258000.pth','output/HumanoidIm/phc_comp_3/Humanoid_00023501.pth']" \
        robot.real_weight_porpotion_boxes=False test=True im_eval=True epoch=-1 headless=True no_virtual_display=True

Every caption of a motion becomes a label over the frames it describes, so the dataset draws one of
them per clip. Writes data/humanml3d_state-action-text-pairs/humanml3d_train.pkl and copies the cache
of text embeddings next to it; the dataset embeds the captions the cache lacks.

HumanML3D counts frames at 20 fps after dropping the first seconds of some datasets, and keeps every
int(framerate / 20)-th AMASS frame; the tracked motions keep every int(framerate / 30)-th.
"""
import argparse
import csv
import shutil
import zipfile
from multiprocessing.pool import ThreadPool
from pathlib import Path

import joblib
import numpy as np
import torch
from tqdm import tqdm

FPS = 30  # of the tracked motions
HUMANML3D_FPS = 20
SKIPPED_SECONDS = {"Eyes_Japan_Dataset": 3, "MPI_HDM05": 3, "TotalCapture": 1, "MPI_Limits": 1, "Transitions_mocap": .5}
KEYS = dict(root_state_all="root_state", dof_state_all="dof_state", body_pos_all="body_pos", action_all="action",
            z_all="pulse_z")
TRACKED_HERE = dict(root_state_all="root_state_all", dof_state_all="dof_state_all", body_pos_all="pred_pos_all",
                    action_all="action_all", z_all="z_all")
MIN_FRAMES = 34  # a clip is 32 frames and the one after


def labels(captions, seconds, rate=None):
    """Captions as (text, start, stop in HumanML3D seconds; 0, 0 for all of it) -> labels over the tracked frames."""
    described = []
    for caption, start, stop in captions:
        start, stop = (0. if np.isnan(t) else t * HUMANML3D_FPS * rate / FPS for t in (start, stop)) if rate else (0., 0.)
        if start == stop == 0:
            start, stop = 0., seconds
        described.append(dict(proc_label=caption, start_t=start, end_t=min(stop, seconds)))
    # The dataset rescales the labels by their last end over the length, and skips a label without a start.
    return described + [dict(end_t=seconds)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--humanml3d", type=Path, required=True, help="train.txt, index.csv and texts.zip")
    parser.add_argument("--amass", type=Path, required=True, help="AMASS, for the frame rate of every motion")
    parser.add_argument("--retargeted", type=Path, required=True, help="the HumanAct12 motions, for their captions")
    parser.add_argument("--tracked", type=Path, default=Path("data/amass_state-action-pairs"))
    parser.add_argument("--tracked-here", type=Path,
                        default=Path("data/amass_tracking_results/eval_info_humanml3d_train_sample.pkl"))
    parser.add_argument("--out", type=Path, default=Path("data/humanml3d_state-action-text-pairs/humanml3d_train.pkl"))
    args = parser.parse_args()
    train = set(args.humanml3d.joinpath("train.txt").read_text().split())
    texts = zipfile.ZipFile(args.humanml3d / "texts.zip")

    def captions(name):
        lines = (line.strip().split("#") for line in texts.read(f"texts/{name}.txt").decode().splitlines())
        return [(caption, float(start), float(stop)) for caption, _, start, stop in lines]

    with open(args.humanml3d / "index.csv") as file:
        rows = {row["new_name"][:-4]: row for row in csv.DictReader(file) if "humanact12" not in row["source_path"]}
    sources = {name: row["source_path"][len("./pose_data/"):-len(".npy")] for name, row in rows.items()}  # <dataset>/<subject>/<motion>_poses
    wanted = sorted({sources[name] for name in rows if name in train or "M" + name in train})
    with ThreadPool(32) as pool:
        framerates = pool.map(lambda source: float(np.load(args.amass / (source + ".npz"))["mocap_framerate"]), wanted)
    # tracked frames per HumanML3D frame
    rate = {source: int(f / HUMANML3D_FPS) / int(f / FPS) for source, f in zip(wanted, framerates)}

    packaged = {name: [] for name in (*KEYS, "is_succ_all", "frame_labels_all", "motion_file")}
    count = dict(amass=0, mirrored=0, humanact12=0)

    def add(name, arrays, described, kind):
        for key, array in arrays.items():
            packaged[key].append(array)
        packaged["is_succ_all"].append(True)
        packaged["frame_labels_all"].append(described)
        packaged["motion_file"].append(name)
        count[kind] += 1

    for name in tqdm([name for name in rows if name in train], desc="AMASS"):
        if not args.tracked.joinpath(sources[name] + ".pkl").exists():
            continue
        tracked = joblib.load(args.tracked / (sources[name] + ".pkl"))
        first = SKIPPED_SECONDS.get(sources[name].split("/")[0], 0) * HUMANML3D_FPS
        begin, end = (round((first + int(rows[name][edge])) * rate[sources[name]]) for edge in ("start_frame", "end_frame"))
        end = min(end, len(tracked["action"]))
        if tracked["is_succ"] and end - begin >= MIN_FRAMES:
            add(name, {key: tracked[source][begin:end] for key, source in KEYS.items()},
                labels(captions(name), (end - begin) / FPS, rate[sources[name]]), "amass")

    here = joblib.load(args.tracked_here)
    retargeted = torch.load(args.retargeted)
    for index in tqdm(here["succ_idxes"], desc="mirrored AMASS and HumanAct12"):
        name = here["motion_keys"][index]
        arrays = {key: here[source][index] for key, source in TRACKED_HERE.items()}
        seconds = len(arrays["action_all"]) / FPS
        if len(arrays["action_all"]) < MIN_FRAMES:
            continue
        if name in retargeted:
            add(name, arrays, labels([(caption, 0., 0.) for caption in retargeted[name]["captions"]], seconds), "humanact12")
        else:
            add(name, arrays, labels(captions(name), seconds, rate[sources[name[1:]]]), "mirrored")

    packaged["succ_idxes"] = list(range(len(packaged["motion_file"])))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(packaged, args.out)
    shutil.copy("data/babel_state-action-text-pairs/text_embedding_dict_clip.pkl", args.out.parent)
    frames = sum(len(a) for a in packaged["action_all"])
    print(f"{len(packaged['motion_file'])} motions, {frames / FPS / 3600:.1f} hours: {count}")


if __name__ == "__main__":
    main()
