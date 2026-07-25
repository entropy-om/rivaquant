"""Publish RivaQuant to Hugging Face: weights, eval report, and a model card
carrying only the numbers actually measured — no unvalidated claims.
"""
import argparse
import json
import os


def build_card(eval_report: dict) -> str:
    samples = "\n".join(f"- `{s}`" for s in eval_report["self_naming_samples"])
    return f"""---
license: mit
tags:
- bitnet
- ternary
- from-scratch
- tinystories
- pretraining
---

# RivaQuant

A small (162M-param), from-scratch decoder-only transformer with **BitNet
b1.58 ternary weights** ({{-1, 0, 1}}) in every attention/MLP projection,
trained on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories).
Architecture and training code: [entropy-om/rivaquant](https://github.com/entropy-om/rivaquant).

RivaQuant is the project/repo name, not a self-chosen identity — see
"Does it know its own name?" below for why.

## What this actually is

- **Architecture**: BitNet b1.58 ternary linear layers (weight quantization
  via absmean + sign, per-token int8 activation quantization, straight-through
  estimator for gradients), RoPE, standard decoder-only transformer block
  structure otherwise. Adapted from real reference implementations
  (kyegomez/BitNet's math, Microsoft's BitNet b1.58 paper), not reinvented
  from memory.
- **Params**: {eval_report["params"]:,}
- **Trained**: {eval_report["trained_steps"]:,} steps (best checkpoint by
  validation loss, out of 20,000 total steps run) on TinyStories, batch
  size 8 with 4x gradient accumulation (effective batch 32), block size 256.
- **Validation perplexity**: {eval_report["val_perplexity"]:.3f}

## What this is not

Not a general-purpose assistant, not instruction-tuned, not evaluated on
anything beyond TinyStories-style short story completion. Read the actual
architecture and eval code before drawing conclusions from perplexity alone.

## Does it know its own name?

No. Prompted with "My name is" / "I am called" / "You can call me" (15
samples, unfiltered), it produces a different plausible children's-story
character name every time — a direct artifact of training purely on
TinyStories, which is full of characters introducing themselves that way.
There is no consistent self-identity to report, so none is claimed:

{samples}

## Training bugs hit and fixed along the way

1. `torch.nn.RMSNorm` requires torch>=2.4; the training environment shipped
   2.1.0. Custom RMSNorm module, same math, no version dependency.
2. CUDA OOM at batch=32/block=512 on a 24GB card — BitNet's straight-through
   estimator keeps extra activation copies per layer, more memory-hungry
   than plain `nn.Linear` at the same nominal size. Fixed with a smaller
   micro-batch + gradient accumulation to keep the same effective batch.

Both caught mid-training by an automated cost-safety watcher that stops the
GPU on any crash signature, understood, fixed, verified with a smoke test,
and relaunched — not discovered after the fact.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default="PeetPedro/rivaquant")
    ap.add_argument("--artifacts-dir",
                     default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "artifacts"))
    args = ap.parse_args()

    with open(os.path.join(args.artifacts_dir, "eval_report.json")) as f:
        eval_report = json.load(f)

    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo_id=args.repo_id, private=False, exist_ok=True)

    api.upload_file(
        path_or_fileobj=os.path.join(args.artifacts_dir, "best.pt"),
        path_in_repo="rivaquant.pt", repo_id=args.repo_id,
    )
    api.upload_file(
        path_or_fileobj=os.path.join(args.artifacts_dir, "eval_report.json"),
        path_in_repo="eval_report.json", repo_id=args.repo_id,
    )
    card = build_card(eval_report)
    api.upload_file(
        path_or_fileobj=card.encode("utf-8"),
        path_in_repo="README.md", repo_id=args.repo_id,
        commit_message="Add model card",
    )
    print(f"published: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
