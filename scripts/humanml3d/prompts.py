"""Write the captions of HumanML3D's test items as prompts for `evaluate_t2m_babel`.

    python scripts/humanml3d/prompts.py --test ../../../../data/humanml3d_test.pt --out output/humanml3d/prompts.txt
    python scripts/humanml3d/prompts.py --test ../../../../data/humanml3d_test.pt --labels \
        --out output/humanml3d/prompts_labels.txt

One caption per test item, drawn at random. Next to the prompts, prompts.json holds the key of the
item and the index of the caption of every line; `evaluate_t2m_babel` simulates each line five times.

UniPhys was trained with BABEL's labels, short phrases that begin with a verb in its base form
("walk forward", "stand still"). The public checkpoint follows such a phrase and largely ignores
the same words as HumanML3D writes them ("a person walks forward."). With --labels a caption is
written in that form from HumanML3D's own lemmas, without the subject it begins with and the
auxiliary after it: "the man is walking backwards." becomes "walk backwards".
"""
import argparse
import json
import random
from pathlib import Path

import torch


DETERMINERS = {"a", "an", "the", "this", "that", "one", "some"}
SUBJECTS = {"person", "man", "woman", "figure", "someone", "somebody", "he", "she", "they", "it", "sim", "toon", "character",
            "robot", "human", "subject", "user", "individual", "guy", "people", "stick", "stickman"}
BE = {"be", "is", "are", "was", "were", "am", "been", "being"}


def label(tokens):
    """HumanML3D's `lemma/POS` tokens of a caption -> its lemmas without the subject it begins with."""
    lemmas = [token.rsplit("/", 1)[0] for token in tokens]
    start = int(lemmas[0] in DETERMINERS)
    subject = start
    while subject < len(lemmas) and lemmas[subject] in SUBJECTS:
        subject += 1
    if subject > start:  # a subject, and the auxiliary that follows it
        start = subject
        while start < len(lemmas) and lemmas[start] in BE:
            start += 1
    return " ".join(lemmas[start:] or lemmas)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", required=True, help="the test items of the benchmark (humanml3d_test.pt)")
    parser.add_argument("--out", type=Path, default=Path("output/humanml3d/prompts.txt"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--items", type=int, help="only the first items, to try the pipeline")
    parser.add_argument("--labels", action="store_true", help="write the captions in the form of UniPhys's training labels")
    args = parser.parse_args()
    test = torch.load(args.test)
    rng = random.Random(args.seed)
    chosen = [(key, rng.randrange(len(captions))) for key, captions in zip(test["key"], test["captions"])][:args.items]
    captions = dict(zip(test["key"], test["captions"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    prompt = (lambda caption: label(caption["tokens"])) if args.labels else (lambda caption: caption["text"].strip())
    args.out.write_text("".join(prompt(captions[key][caption]) + "\n" for key, caption in chosen))
    args.out.with_suffix(".json").write_text(json.dumps(chosen))
    print(f"{len(chosen)} prompts in {args.out}")


if __name__ == "__main__":
    main()
