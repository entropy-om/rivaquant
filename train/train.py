"""RivaQuant pretraining loop. Standard nanoGPT-style training mechanics
(AdamW, cosine LR with warmup, gradient clipping) — the only thing novel
here is the model architecture (model/transformer.py), not the training
loop, deliberately, so a bad run can only be the ternary-weights bet, not a
training-loop bug.
"""
import math
import os
import time

import numpy as np
import torch

from model import RivaQuant, RivaQuantConfig

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUT_DIR = os.environ.get("RIVAQUANT_OUT", "/workspace/rivaquant-out")
STATUS_PATH = os.path.join(OUT_DIR, "status.json")
LOG_PATH = os.path.join(OUT_DIR, "train.log")

BLOCK_SIZE = int(os.environ.get("RIVAQUANT_BLOCK_SIZE", "512"))
BATCH_SIZE = int(os.environ.get("RIVAQUANT_BATCH_SIZE", "32"))
MAX_STEPS = int(os.environ.get("RIVAQUANT_MAX_STEPS", "20000"))
EVAL_INTERVAL = int(os.environ.get("RIVAQUANT_EVAL_INTERVAL", "250"))
LR = float(os.environ.get("RIVAQUANT_LR", "3e-4"))
WARMUP_STEPS = int(os.environ.get("RIVAQUANT_WARMUP_STEPS", "500"))
DEVICE = os.environ.get("RIVAQUANT_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")


def write_status(**fields) -> None:
    import json
    os.makedirs(OUT_DIR, exist_ok=True)
    fields["updated_at"] = time.time()
    with open(STATUS_PATH, "w") as f:
        json.dump(fields, f, indent=2)


def log(msg: str) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def get_batch(split: str, cfg: RivaQuantConfig):
    path = os.path.join(DATA_DIR, f"{split}.bin")
    data = np.memmap(path, dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - cfg.block_size - 1, (BATCH_SIZE,))
    x = torch.stack([torch.from_numpy(data[i:i + cfg.block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + cfg.block_size].astype(np.int64)) for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


def lr_at(step: int) -> float:
    if step < WARMUP_STEPS:
        return LR * step / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(1, MAX_STEPS - WARMUP_STEPS)
    return 0.5 * LR * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def estimate_val_loss(model: RivaQuant, cfg: RivaQuantConfig, iters: int = 20) -> float:
    model.eval()
    losses = []
    for _ in range(iters):
        x, y = get_batch("val", cfg)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def main() -> None:
    cfg = RivaQuantConfig(block_size=BLOCK_SIZE)
    model = RivaQuant(cfg).to(DEVICE)
    log(f"model params: {model.num_params():,}  device: {DEVICE}")
    write_status(stage="training", step=0, params=model.num_params())

    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.1)
    best_val = float("inf")

    for step in range(MAX_STEPS):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)

        x, y = get_batch("train", cfg)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 50 == 0:
            log(f"step {step}/{MAX_STEPS}  train_loss {loss.item():.4f}  lr {lr_at(step):.2e}")
            write_status(stage="training", step=step, max_steps=MAX_STEPS,
                          train_loss=loss.item(), best_val_loss=best_val)

        if step > 0 and step % EVAL_INTERVAL == 0:
            val_loss = estimate_val_loss(model, cfg)
            log(f"step {step}  val_loss {val_loss:.4f}")
            if val_loss < best_val:
                best_val = val_loss
                torch.save({"model": model.state_dict(), "cfg": cfg, "step": step},
                           os.path.join(OUT_DIR, "best.pt"))
            write_status(stage="training", step=step, max_steps=MAX_STEPS,
                          train_loss=loss.item(), val_loss=val_loss, best_val_loss=best_val)

    torch.save({"model": model.state_dict(), "cfg": cfg, "step": MAX_STEPS},
               os.path.join(OUT_DIR, "final.pt"))
    write_status(stage="done", step=MAX_STEPS, best_val_loss=best_val)
    log(f"done. best_val_loss={best_val:.4f}")


if __name__ == "__main__":
    main()
