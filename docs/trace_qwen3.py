"""Trace the transitive closure of ATen ops used by a Qwen3 transformer.

Runs a tiny Qwen3ForCausalLM forward (and backward) under a TorchDispatchMode
that records every low-level aten op the whole stack decomposes into. Also
records per-module op sets by wrapping the dispatch counting with nn forward
hooks so we can attribute ops to transformer components.
"""

import collections
import sys

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM


class OpRecorder(TorchDispatchMode):
    """Records every aten op dispatched while active, with call counts."""

    def __init__(self):
        self.counts = collections.Counter()
        # module attribution: name of the currently-executing leaf module
        self.current = None
        self.by_module = collections.defaultdict(collections.Counter)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = func._schema.name  # e.g. "aten::add"
        overload = str(func)      # e.g. "aten.add.Tensor"
        self.counts[name] += 1
        if self.current is not None:
            self.by_module[self.current][overload] += 1
        return func(*args, **(kwargs or {}))


def build_tiny_qwen3(attn_impl):
    cfg = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,   # GQA: fewer KV heads than query heads
        head_dim=8,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        attn_implementation=attn_impl,
    )
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(cfg)
    model.eval()
    return model


def trace(attn_impl, do_backward):
    model = build_tiny_qwen3(attn_impl)
    input_ids = torch.randint(0, 64, (1, 8))

    rec = OpRecorder()

    # Attribute ops to leaf modules via forward pre/post hooks that set/clear
    # rec.current. Only leaf (childless) modules get their own bucket.
    handles = []
    for name, mod in model.named_modules():
        if len(list(mod.children())) == 0 and name:
            def pre(m, inp, _name=name):
                rec._saved = rec.current
                rec.current = type(m).__name__
            def post(m, inp, out, _name=name):
                rec.current = getattr(rec, "_saved", None)
            handles.append(mod.register_forward_pre_hook(pre))
            handles.append(mod.register_forward_hook(post))

    with rec:
        if do_backward:
            out = model(input_ids).logits
            loss = out.float().sum()
            loss.backward()
        else:
            with torch.no_grad():
                model(input_ids)

    for h in handles:
        h.remove()
    return rec


def main():
    attn_impl = sys.argv[1] if len(sys.argv) > 1 else "eager"
    do_backward = "--backward" in sys.argv

    rec = trace(attn_impl, do_backward)

    print(f"# attn_implementation={attn_impl} backward={do_backward}")
    print(f"# distinct aten ops: {len(rec.counts)}  total dispatches: {sum(rec.counts.values())}")
    print("\n## Transitive closure (aten op : call count)")
    for name, c in sorted(rec.counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"{c:5d}  {name}")

    print("\n## Ops by leaf module type")
    for modtype in sorted(rec.by_module):
        ops = rec.by_module[modtype]
        opnames = sorted(ops)
        print(f"\n### {modtype}")
        for o in opnames:
            print(f"    {ops[o]:4d}  {o}")


if __name__ == "__main__":
    main()
