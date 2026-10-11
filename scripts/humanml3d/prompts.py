"""Write the captions of HumanML3D's test items as prompts for `evaluate_t2m_babel`.

    python scripts/humanml3d/prompts.py --test ../../../../data/humanml3d_test.pt --out output/humanml3d/prompts.txt

One caption per test item, drawn at random. Next to the prompts, prompts.json holds the key of the
item and the index of the caption of every line; `evaluate_t2m_babel` simulates each line five times.
"""
import argparse
import json
import random
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", required=True, help="the test items of the benchmark (humanml3d_test.pt)")
    parser.add_argument("--out", type=Path, default=Path("output/humanml3d/prompts.txt"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--items", type=int, help="only the first items, to try the pipeline")
    args = parser.parse_args()
    test = torch.load(args.test)
    rng = random.Random(args.seed)
    chosen = [(key, rng.randrange(len(captions))) for key, captions in zip(test["key"], test["captions"])][:args.items]
    captions = dict(zip(test["key"], test["captions"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(captions[key][caption]["text"].strip() + "\n" for key, caption in chosen))
    args.out.with_suffix(".json").write_text(json.dumps(chosen))
    print(f"{len(chosen)} prompts in {args.out}")


if __name__ == "__main__":
    main()
