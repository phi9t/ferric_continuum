#!/usr/bin/env python3
"""HuggingFace reference forward for numeric compatibility with tnsr's Rust Qwen3.

Loads a local ``Qwen/Qwen3-0.6B`` checkpoint, runs a single forward on a prompt
(or an explicit token-id sequence), and writes the **last-position**
full-vocabulary logits row as JSON.  The schema is identical to the dump emitted
by ``qwen3_infer --dump-logits`` so a single comparator (``compare_logits.py``)
handles both sides::

    {"token_ids": [...], "prompt": "...", "vocab_size": V, "logits": [f32; V]}

To keep the comparison honest w.r.t. tnsr's f32 CPU math, the model is forced to
**float32** and ``attn_implementation="eager"`` (no fused/flash kernels, which
would introduce their own numeric drift).

This is *not* wired into Bazel: torch/transformers are not in
``requirements_lock.txt``.  Run it from a host venv, e.g.::

    python3 -m venv .venv-hf && .venv-hf/bin/pip install torch transformers
    .venv-hf/bin/python hf_reference.py --model-dir <dir> \\
        --prompt "The capital of France is" --out /tmp/hf_logits_0.json
"""

import argparse
import json
import sys


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", required=True, help="local Qwen3 checkpoint dir")
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument(
        "--token-ids",
        default=None,
        help="comma-separated explicit token ids (bypasses the HF tokenizer)",
    )
    p.add_argument("--out", required=True, help="output JSON path")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    import torch  # deferred so --help works without torch installed
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(0)

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    model.eval()

    if args.token_ids is not None:
        ids = [int(x) for x in args.token_ids.split(",") if x.strip() != ""]
    else:
        ids = tok.encode(args.prompt, add_special_tokens=False)

    if not ids:
        print("hf_reference: prompt encoded to zero tokens", file=sys.stderr)
        return 1

    print(f"hf_reference: token_ids = {ids}", file=sys.stderr)

    input_ids = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        out = model(input_ids=input_ids)
    # logits: [1, T, V] -> last position row [V]
    last_row = out.logits[0, -1, :].to(torch.float32).cpu().numpy()
    vocab_size = int(last_row.shape[0])

    payload = {
        "token_ids": ids,
        "prompt": args.prompt,
        "vocab_size": vocab_size,
        "logits": [float(x) for x in last_row.tolist()],
    }
    with open(args.out, "w") as f:
        json.dump(payload, f)
    print(f"hf_reference: wrote {args.out} (vocab={vocab_size})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
