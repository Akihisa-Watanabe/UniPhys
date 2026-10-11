"""Make reference motions for PULSE of what HumanML3D's training split has beyond the tracked AMASS.

    python scripts/humanml3d/reference_motions.py --humanml3d ../../../../data/HumanML3D \
        --amass <AMASS>/sequences --retargeted ../../../../data/humanml3d_train_humanact12.pt

Two kinds of motion have no tracked counterpart: the mirrored copies of the AMASS motions, made here
by mirroring the SMPL parameters of the same frames `package_training.py` takes, and the motions from
HumanAct12, read as the SMPL poses they were retargeted to. Both are prepared as
scripts/data_process/process_amass_db.py and convert_amass_isaac.py prepare AMASS: the hands at rest,
the lowest vertex of the first frame on the floor, the neutral body. Writes
data/humanml3d_reference/humanml3d_train.pkl, keyed by HumanML3D's names.
"""
import argparse
import csv
from pathlib import Path

import joblib
import numpy as np
import torch
from scipy.spatial.transform import Rotation as sRot
from smpl_sim.smpllib.smpl_joint_names import SMPL_BONE_ORDER_NAMES
from smpl_sim.smpllib.smpl_parser import SMPL_Parser
from tqdm import tqdm

from package_training import FPS, HUMANML3D_FPS, SKIPPED_SECONDS
from poselib.skeleton.skeleton3d import SkeletonState, SkeletonTree

MUJOCO_NAMES = ['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe', 'Torso', 'Spine',
                'Chest', 'Neck', 'Head', 'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand', 'R_Thorax',
                'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand']
SMPL_TO_MUJOCO = [SMPL_BONE_ORDER_NAMES.index(name) for name in MUJOCO_NAMES]
MIRRORED = [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20, 23, 22]  # SMPL joints, left and right swapped


def on_the_floor(parser, pose, trans, betas):
    """Lower or raise trans[T,3] until the lowest vertex of the first frame touches the floor."""
    vertices, _ = parser.get_joints_verts(torch.from_numpy(pose[:1]).float(), betas, th_trans=torch.from_numpy(trans[:1]).float())
    trans = trans.copy()
    trans[:, 2] -= vertices[..., 2].min().item()
    return trans


def reference(tree, pose, trans):
    """SMPL pose[T,72] and translation[T,3] -> a motion as the simulated humanoid's motion library reads it."""
    frames = len(pose)
    pose = np.concatenate((pose[:, :66], np.zeros((frames, 6))), 1)
    local = sRot.from_rotvec(pose.reshape(frames, 24, 3)[:, SMPL_TO_MUJOCO].reshape(-1, 3)).as_quat().reshape(frames, 24, 4)
    root = torch.from_numpy(trans) + tree.local_translation[0]
    state = SkeletonState.from_rotation_and_root_translation(tree, torch.from_numpy(local), root, is_local=True)
    # the humanoid stands upright at zero rotation
    world = (sRot.from_quat(state.global_rotation.reshape(-1, 4).numpy()) * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv())
    world = world.as_quat().reshape(frames, 24, 4)
    state = SkeletonState.from_rotation_and_root_translation(tree, torch.from_numpy(world), root, is_local=False)
    return dict(pose_quat_global=world, pose_quat=state.local_rotation.numpy(), trans_orig=trans, root_trans_offset=root,
                beta=np.zeros(16), gender="neutral", pose_aa=pose, fps=float(FPS))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--humanml3d", type=Path, required=True, help="train.txt and index.csv")
    parser.add_argument("--amass", type=Path, required=True)
    parser.add_argument("--retargeted", type=Path, required=True, help="the HumanAct12 motions as SMPL poses")
    parser.add_argument("--tracked", type=Path, default=Path("data/amass_state-action-pairs"))
    parser.add_argument("--out", type=Path, default=Path("data/humanml3d_reference/humanml3d_train.pkl"))
    args = parser.parse_args()
    tree = SkeletonTree.from_mjcf("phc/data/assets/mjcf/smpl_humanoid.xml")
    bodies = {gender: SMPL_Parser(model_path="data/smpl", gender=gender, use_pca=False, create_transl=False)
              for gender in ("neutral", "male", "female")}
    train = set(args.humanml3d.joinpath("train.txt").read_text().split())
    motions = {}

    with open(args.humanml3d / "index.csv") as file:
        rows = [row for row in csv.DictReader(file)
                if "M" + row["new_name"][:-4] in train and "humanact12" not in row["source_path"]]
    for row in tqdm(rows, desc="mirrored AMASS"):
        source = row["source_path"][len("./pose_data/"):-len(".npy")]
        if not args.tracked.joinpath(source + ".pkl").exists():  # UniPhys left the motion out as infeasible
            continue
        recorded = np.load(args.amass / (source + ".npz"))
        framerate = float(recorded["mocap_framerate"])
        skip = int(framerate / FPS)
        pose = np.concatenate((recorded["poses"][::skip, :66], np.zeros((len(recorded["poses"][::skip]), 6))), 1)
        gender = recorded["gender"].item() if recorded["gender"].shape == () else str(recorded["gender"])
        gender = gender.decode() if isinstance(gender, bytes) else gender
        trans = on_the_floor(bodies[gender], pose, recorded["trans"][::skip], torch.from_numpy(recorded["betas"][None]).float())
        rate = int(framerate / HUMANML3D_FPS) / skip
        first = SKIPPED_SECONDS.get(source.split("/")[0], 0) * HUMANML3D_FPS
        tracked = len(joblib.load(args.tracked / (source + ".pkl"))["action"])  # UniPhys cut some motions short
        begin, end = round((first + int(row["start_frame"])) * rate), min(round((first + int(row["end_frame"])) * rate), tracked)
        if end - begin < 34:
            continue
        # the mirror image in the plane x = 0
        pose = pose[begin:end].reshape(-1, 24, 3)[:, MIRRORED] * np.array([1., -1, -1])
        motions["M" + row["new_name"][:-4]] = reference(tree, pose.reshape(-1, 72), trans[begin:end] * np.array([-1., 1, 1]))

    pelvis = tree.local_translation[0].numpy()
    for name, item in tqdm(torch.load(args.retargeted).items(), desc="HumanAct12"):
        pose = item["pose"].double().numpy().reshape(-1, 72)
        trans = on_the_floor(bodies["neutral"], pose, item["root"].double().numpy() - pelvis, torch.zeros(1, 10))
        motions[name] = reference(tree, pose, trans)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(motions, args.out, compress=True)
    frames = sum(len(motion["pose_aa"]) for motion in motions.values())
    print(f"{len(motions)} reference motions, {frames / FPS / 3600:.1f} hours, in {args.out}")


if __name__ == "__main__":
    main()
