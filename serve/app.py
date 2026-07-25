"""Dedicated free-tier inference endpoint for RivaQuant. CPU-only — a 162M
ternary model doesn't need a GPU to serve, and this is meant to be a free,
always-on demo, not a production API.
"""
import tiktoken
import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from huggingface_hub import hf_hub_download
from pydantic import BaseModel

from model import RivaQuant, RivaQuantConfig

app = FastAPI(title="RivaQuant")
enc = tiktoken.get_encoding("gpt2")

_ckpt_path = hf_hub_download("PeetPedro/rivaquant", "rivaquant.pt")
_ckpt = torch.load(_ckpt_path, map_location="cpu", weights_only=False)
_cfg: RivaQuantConfig = _ckpt["cfg"]
model = RivaQuant(_cfg)
model.load_state_dict(_ckpt["model"])
model.eval()


class GenerateRequest(BaseModel):
    prompt: str = "Once upon a time"
    max_new_tokens: int = 60
    temperature: float = 0.8
    top_k: int = 40


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return """<!doctype html>
<html><head><title>RivaQuant</title>
<style>body{font-family:monospace;max-width:640px;margin:3em auto;padding:0 1em}
textarea,input{width:100%;box-sizing:border-box;font-family:monospace}
pre{white-space:pre-wrap;background:#111;color:#eee;padding:1em;border-radius:4px}</style>
</head><body>
<h1>RivaQuant</h1>
<p>162M-param, from-scratch, BitNet b1.58 ternary transformer, trained on TinyStories.
No consistent self-identity — <a href="https://huggingface.co/PeetPedro/rivaquant">read why</a>.
<a href="https://github.com/entropy-om/rivaquant">source</a>.</p>
<textarea id="p" rows="3">Once upon a time</textarea>
<button onclick="go()">generate</button>
<pre id="out"></pre>
<script>
async function go(){
  const r = await fetch('/generate', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({prompt: document.getElementById('p').value})});
  const j = await r.json();
  document.getElementById('out').textContent = j.text;
}
</script>
</body></html>"""


@app.post("/generate")
@torch.no_grad()
def generate(req: GenerateRequest) -> dict:
    idx = torch.tensor([enc.encode(req.prompt)])
    out = model.generate(idx, max_new_tokens=req.max_new_tokens,
                          temperature=req.temperature, top_k=req.top_k)
    return {"text": enc.decode(out[0].tolist())}


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "params": model.num_params()}
