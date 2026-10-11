"""Turn the episodes `evaluate_t2m_babel` saved into the file the benchmark's scorer reads.

    python scripts/humanml3d/export_episodes.py output/UniPhys/evalate_t2m/<run>/saved_episodes.pkl \
        --prompts output/humanml3d/prompts.json --out output/humanml3d/public.npz

As CLoSD does when it records, the first 16 control steps of an episode are left out and the next 300
are kept. The scorer applies the fall rule itself, so every episode is exported.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np

SKIPPED, KEPT = 16, 300  # control steps at 30 Hz
FPS = 30


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("episodes", type=Path)
    parser.add_argument("--prompts", type=Path, required=True, help="prompts.json of scripts/humanml3d/prompts.py")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    saved = joblib.load(args.episodes)
    prompts = json.loads(args.prompts.read_text())
    body_pos = np.asarray(saved["body_pos"], dtype=np.float32)  # (steps, episodes, 24, 3)
    if len(body_pos) < SKIPPED + KEPT:
        raise ValueError(f"Episodes of {len(body_pos)} steps; {SKIPPED + KEPT} are needed")
    lines = args.prompts.with_suffix(".txt").read_text().splitlines()
    episodes = body_pos.shape[1]
    # `evaluate_t2m_babel` goes through the prompts five times, in order.
    line = np.arange(episodes) % len(prompts)
    if [lines[i] for i in line] != list(saved["text"]):
        raise ValueError("The saved episodes were not prompted with these prompts")
    key, caption = zip(*(prompts[i] for i in line))
    np.savez(args.out, body_pos=body_pos[SKIPPED:SKIPPED + KEPT].transpose(1, 0, 2, 3), fps=FPS,
             key=np.array(key), caption=np.array(caption))
    print(f"{episodes} episodes of {KEPT} steps in {args.out}")


if __name__ == "__main__":
    main()
