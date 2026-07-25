"""Evaluate the trained checkpoint honestly, let it name itself, then publish
to Hugging Face as a public repo with a model card carrying the real
numbers — no claim goes in the card that wasn't measured here.
"""
import argparse
import json
import math
import os

import tiktoken
import torch

from model import RivaQuant, RivaQuantConfig
from train import DATA_DIR, get_batch


@torch.no_grad()
def final_val_perplexity(model: RivaQuant, cfg: RivaQuantConfig, iters: int = 100) -> float:
    model.eval()
    losses = []
    for _ in range(iters):
        x, y = get_batch("val", cfg)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    mean_loss = sum(losses) / len(losses)
    return math.exp(mean_loss)


@torch.no_grad()
def self_naming_samples(model: RivaQuant, enc, device, n: int = 5) -> list[str]:
    """Prompt the model to name itself — per instruction, it names itself,
    we don't impose a name on it. Multiple samples since a 125M ternary model
    on TinyStories will be noisy; report what actually comes out, not a
    cherry-picked one."""
    prompts = ["My name is", "I am called", "You can call me"]
    outputs = []
    for p in prompts:
        for _ in range(n):
            idx = torch.tensor([enc.encode(p)], device=device)
            out = model.generate(idx, max_new_tokens=20, temperature=0.8, top_k=40)
            outputs.append(enc.decode(out[0].tolist()))
    return outputs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="/workspace/rivaquant-out/best.pt")
    ap.add_argument("--hf-repo", default="PeetPedro/rivaquant")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg: RivaQuantConfig = ckpt["cfg"]
    model = RivaQuant(cfg).to(device)
    model.load_state_dict(ckpt["model"])

    val_ppl = final_val_perplexity(model, cfg)
    print(f"final val perplexity: {val_ppl:.3f}  (trained {ckpt['step']} steps)")

    enc = tiktoken.get_encoding("gpt2")
    naming = self_naming_samples(model, enc, device)
    print("self-naming samples:")
    for s in naming:
        print(f"  {s!r}")

    report = {
        "val_perplexity": val_ppl,
        "trained_steps": ckpt["step"],
        "params": model.num_params(),
        "self_naming_samples": naming,
    }
    out_path = os.path.join(os.path.dirname(args.checkpoint), "eval_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {out_path}")
    print("Run push_to_hub.py separately once you've read the naming samples "
          "and picked what the model card should call it.")


if __name__ == "__main__":
    main()
