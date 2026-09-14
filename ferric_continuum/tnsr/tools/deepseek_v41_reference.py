#!/usr/bin/env python3
"""Reference last-position logits for the DeepSeek V4.1 **text-only** path.

This is the Level-6 parity reference.  It emits the same JSON schema the Rust
CLI (``deepseek_v41_infer --dump-logits``) writes so
``deepseek_v41_compare_logits.py`` can diff the two.  The ``--dspark`` mode
additionally emits the tiny ``forward_spec`` reference (``output_ids`` /
``logits`` / ``confidence``) recomputed from the shared ``generate_dspark_model``
helper, the same numeric source the Rust ``forward_spec`` parity test consumes.

Two modes:

``--emit-tiny-checkpoint <dir>``
    Write a tiny, deterministic converted-style checkpoint (``config.json`` in
    the ``from_inference_json`` schema + a single ``model.safetensors`` with the
    *upstream* ``[out, in]`` tensor names the Rust loader consumes).  The tensor
    values are the exact ones the reference logits are computed from, so the
    Rust loader and this reference operate on the same numbers.

``--out <path>`` (with ``--checkpoint <dir>``)
    Load the tiny checkpoint's ``config.json`` + tensors back, recompute the
    expected last-position logits with the pure-Python ``*_ref`` helpers in
    ``deepseek_v41_fixture_gen.py`` (the very helpers the Rust tiny-model
    forward is validated against in ``deepseek_v41_model_test.rs``), and write
    the parity JSON.

Why not run upstream ``Transformer.forward`` directly?  The tnsr Wave-1
attention seam uses a sigmoid gate and skips RoPE / online-softmax
``sparse_attn`` (see ``src/deepseek_v41/attention.rs``), so a faithful upstream
forward would *not* numerically match tnsr.  Real-checkpoint parity against the
released ``model.py`` is a later wave and is recorded as SKIP by the verifier
when no weights are present.  To keep this reference honest we still *path
import* upstream ``model.py`` and record the outcome in ``reference_backend`` /
``kernel_patch`` metadata, but the numbers come from the shared ``*_ref``
helpers.

The reference is intentionally the *same math* as the Rust forward, expressed
independently in Python, so a mismatch flags a real Rust regression (wrong
layout, dropped term, bad dequant) rather than a modeling disagreement.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


def _repo_root_from_here() -> Path:
    # tools/ -> tnsr/ -> ferric_continuum/ -> repo root
    return Path(__file__).resolve().parents[3]


def _import_fixture_gen(repo_root: Path):
    """Import the fixture generator module so we can reuse its ``*_ref`` math."""
    fg_path = repo_root / "ferric_continuum/tnsr/tools/deepseek_v41_fixture_gen.py"
    if not fg_path.exists():
        raise SystemExit(f"missing fixture generator: {fg_path}")
    spec = importlib.util.spec_from_file_location("deepseek_v41_fixture_gen", fg_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"could not import {fg_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record_upstream_backend(repo_root: Path, fg) -> tuple[str, str]:
    """Path-import upstream ``model.py`` purely to record provenance metadata.

    Returns ``(reference_backend, kernel_patch)``.  We never *execute* a forward
    through it in tiny mode (see the module docstring), so a failed import is a
    benign, recorded skip — not an error.
    """
    model_py = repo_root / "ferric_continuum/tnsr/third_party/deepseek_v41/upstream/inference/model.py"
    if not model_py.exists():
        return ("ref-helpers:no-upstream-model.py", "none")
    try:
        _module, status = fg.import_upstream_model_by_path(model_py)
    except Exception as err:  # noqa: BLE001
        return (f"ref-helpers:upstream-import-error:{type(err).__name__}", "none")
    # The tiny reference does not call upstream GPU kernels; it records that the
    # stub installer replaced them (act_quant/fp8_gemm/... raise; hc_split_sinkhorn
    # is a pure-torch monkeypatch) so a future real forward has the provenance.
    kernel_patch = (
        "pure-torch:hc_split_sinkhorn;raising-stubs:act_quant,fp4_act_quant,"
        "fp4_gemm,fp8_gemm,sparse_attn"
    )
    return (f"ref-helpers+upstream-model:{status}", kernel_patch)


# --------------------------------------------------------------------------
# Tiny checkpoint construction
# --------------------------------------------------------------------------
#
# Dimensions match the loader's TINY_CONFIG_JSON (dim=4, 1 sliding-window layer,
# hc_mult=2, n_heads=2, head_dim=2, q_lora=3, o_lora=2, o_groups=2, inter=3,
# experts=2, topk=1) so the Rust loader accepts what we write.  We keep the math
# small but non-degenerate: deterministic ramps, no zeros-everywhere.

TINY = dict(
    vocab_size=5,
    dim=4,
    n_layers=1,
    n_heads=2,
    head_dim=2,
    rope_head_dim=1,
    q_lora=3,
    o_lora=2,
    o_groups=2,
    inter=3,
    experts=2,
    topk=1,
    hc_mult=2,
    window=4,
    image_token_id=4,
    gate_temp=1.0,
    route_scale=1.0,
    swiglu_limit=0.0,
    norm_topk_prob=True,
    hc_sinkhorn_iters=2,
    hc_eps=1e-6,
    rms_eps=1e-6,
    ids=[1, 3],
)


def _tiny_config_json() -> dict:
    t = TINY
    return {
        "vocab_size": t["vocab_size"],
        "dim": t["dim"],
        "moe_inter_dim": t["inter"],
        "n_layers": t["n_layers"],
        "n_heads": t["n_heads"],
        "head_dim": t["head_dim"],
        "rope_head_dim": t["rope_head_dim"],
        "q_lora_rank": t["q_lora"],
        "o_lora_rank": t["o_lora"],
        "o_groups": t["o_groups"],
        "norm_eps": t["rms_eps"],
        "rope_theta": 10000.0,
        "rope_factor": 1.0,
        "original_seq_len": 0,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "window_size": t["window"],
        "compress_ratios": [0],
        "compress_rope_theta": 40000.0,
        "kv_source_layers": [],
        "index_source_layers": [],
        "index_n_heads": 2,
        "index_head_dim": 2,
        "index_topk": 1,
        "candidate_source_layer": 0,
        "candidate_topk_blocks": 1,
        "candidate_block_size": 1,
        "hc_mult": t["hc_mult"],
        "hc_sinkhorn_iters": t["hc_sinkhorn_iters"],
        "hc_eps": t["hc_eps"],
        "n_routed_experts": t["experts"],
        "n_shared_experts": 1,
        "n_activated_experts": t["topk"],
        "score_func": "sqrtsoftplus",
        "route_scale": t["route_scale"],
        "swiglu_limit": t["swiglu_limit"],
        "engram_layer_ids": [],
        "engram_num_embeddings": [],
        "engram_max_ngram_size": 3,
        "engram_vocab_size": 5,
        "engram_n_heads": 2,
        "engram_head_dim": 2,
        "engram_pad_id": 2,
        "engram_compressed_vocab_size": 5,
        "image_token_id": t["image_token_id"],
        "dtype": "f32",
        "expert_dtype": "f32",
        "n_mtp_layers": 0,
        "dspark_block_size": 0,
        "dspark_noise_token_id": 0,
        "dspark_target_layer_ids": [],
        "dspark_markov_rank": 0,
        "dspark_n_routed_experts": 0,
        "dspark_n_activated_experts": 0,
        "vision_n_layers": 0,
        "vision_dim": 0,
        "vision_n_heads": 0,
        "vision_inter_dim": 0,
        "vision_patch_size": 0,
        "vision_rope_theta": 0.0,
        "vision_downsample_ratio": 0,
        "vision_max_n_token": 0,
        "vision_min_pixels": 0,
        "vision_max_wh_ratio": None,
    }


def _ramp(n: int, mod: int, sub: float, div: float) -> list[float]:
    return [((i % mod) - sub) / div for i in range(n)]


def _tiny_weights() -> dict:
    """Deterministic tiny weights expressed in *tnsr* ``[in, out]`` layout.

    ``matmul_rows`` in the fixture generator consumes ``[in, out]`` (indexed
    ``w[i*dout + j]``), which is exactly what the Rust loader produces after it
    transposes the stored upstream ``[out, in]`` linears.  So we generate the
    tnsr layout here, hand it to the ``*_ref`` helpers for the reference logits,
    and *transpose back* to ``[out, in]`` only when serializing the checkpoint.
    """
    t = TINY
    dim, hc = t["dim"], t["hc_mult"]
    nh, hd = t["n_heads"], t["head_dim"]
    q_lora, o_lora, o_groups = t["q_lora"], t["o_lora"], t["o_groups"]
    inter, experts = t["inter"], t["experts"]
    mix_hc = (2 + hc) * hc
    group_in = (nh // o_groups) * hd

    w = {}
    # globals
    w["embed"] = _ramp(t["vocab_size"] * dim, 11, 5, 7.0)  # [V, D] (copied as-is)
    w["lm_head"] = _ramp(dim * t["vocab_size"], 13, 6, 8.0)  # tnsr [D, V]
    w["final_norm"] = [1.0, 0.875, 1.125, 0.75]

    # attention (tnsr [in, out] where relevant)
    w["attn_norm"] = [1.0, 0.875, 1.25, 0.75]
    w["ffn_norm"] = [0.8, 1.1, 0.9, 1.2]
    w["wq_a"] = _ramp(dim * q_lora, 5, 2, 7.0)  # [D, q_lora]
    w["q_norm"] = [1.0, 0.75, 1.25][:q_lora]
    w["wq_b"] = _ramp(q_lora * nh * hd, 11, 5, 6.0)  # [q_lora, nh*hd]
    w["wkv"] = _ramp(dim * hd, 13, 6, 8.0)  # [D, hd]
    w["kv_norm"] = [1.0, 1.5][:hd]
    # wo_a in tnsr flat [g, group_in, o_lora]
    w["wo_a"] = _ramp(o_groups * group_in * o_lora, 9, 4, 6.0)
    w["wo_b"] = _ramp(o_groups * o_lora * dim, 7, 3, 5.0)  # [g*o_lora, D]
    w["attn_sink"] = [0.25, -0.1][:nh]

    # moe: gate row-major [experts, dim]
    w["gate_weight"] = _ramp(experts * dim, 13, 6, 5.0)
    w["correction_bias"] = [0.0, 0.25, -0.1, 0.15][:experts]
    w["experts"] = []
    for e in range(experts):
        w["experts"].append(
            {
                "w1": [((i + e) % 7 - 3) / 6.0 for i in range(dim * inter)],
                "w2": [((i + 2 * e) % 11 - 5) / 8.0 for i in range(inter * dim)],
                "w3": [((i + 3 * e) % 5 - 2) / 5.0 for i in range(dim * inter)],
            }
        )
    w["shared_expert"] = {
        "w1": [((i % 7) - 3) / 8.0 for i in range(dim * inter)],
        "w2": [((i % 9) - 4) / 7.0 for i in range(inter * dim)],
        "w3": [((i % 5) - 2) / 6.0 for i in range(dim * inter)],
    }

    # hyper-connection
    w["hc_attn_fn"] = _ramp(mix_hc * hc * dim, 17, 8, 30.0)
    w["hc_attn_base"] = _ramp(mix_hc, 7, 3, 20.0)
    w["hc_attn_scale"] = [0.15, 0.2, 0.1]
    w["hc_ffn_fn"] = _ramp(mix_hc * hc * dim, 19, 9, 35.0)
    w["hc_ffn_base"] = _ramp(mix_hc, 5, 2, 18.0)
    w["hc_ffn_scale"] = [0.12, 0.18, 0.09]
    return w


def _transpose(flat: list[float], rows: int, cols: int) -> list[float]:
    """Transpose a row-major ``[rows, cols]`` list into ``[cols, rows]``."""
    out = [0.0] * (rows * cols)
    for r in range(rows):
        for c in range(cols):
            out[c * rows + r] = flat[r * cols + c]
    return out


def _emit_tiny_checkpoint(out_dir: Path) -> None:
    """Write config.json + model.safetensors the Rust loader can read.

    Linears are stored in upstream ``[out, in]`` layout (the loader transposes
    them back to tnsr ``[in, out]``); ``embed`` / ``head`` / norms / ``attn_sink``
    are stored as-is; ``wo_a`` is stored upstream ``[g*o_lora, group_in]``.
    """
    import numpy as np  # local import; safetensors needs numpy arrays
    from safetensors.numpy import save_file

    t = TINY
    dim, hc = t["dim"], t["hc_mult"]
    nh, hd = t["n_heads"], t["head_dim"]
    q_lora, o_lora, o_groups = t["q_lora"], t["o_lora"], t["o_groups"]
    inter, experts = t["inter"], t["experts"]
    mix_hc = (2 + hc) * hc
    group_in = (nh // o_groups) * hd
    w = _tiny_weights()

    def arr(flat: list[float], shape: tuple[int, ...]) -> "np.ndarray":
        return np.asarray(flat, dtype=np.float32).reshape(shape)

    tensors: dict[str, "np.ndarray"] = {}
    tensors["embed.weight"] = arr(w["embed"], (t["vocab_size"], dim))
    # tnsr lm_head is [D, V]; upstream head.weight is [V, D] => transpose.
    tensors["head.weight"] = arr(_transpose(w["lm_head"], dim, t["vocab_size"]), (t["vocab_size"], dim))
    tensors["norm.weight"] = arr(w["final_norm"], (dim,))

    lp = "layers.0."
    tensors[lp + "attn_norm.weight"] = arr(w["attn_norm"], (dim,))
    tensors[lp + "ffn_norm.weight"] = arr(w["ffn_norm"], (dim,))

    # attention linears: tnsr [in, out] -> upstream [out, in].
    tensors[lp + "attn.wq_a.weight"] = arr(_transpose(w["wq_a"], dim, q_lora), (q_lora, dim))
    tensors[lp + "attn.q_norm.weight"] = arr(w["q_norm"], (q_lora,))
    tensors[lp + "attn.wq_b.weight"] = arr(_transpose(w["wq_b"], q_lora, nh * hd), (nh * hd, q_lora))
    tensors[lp + "attn.wkv.weight"] = arr(_transpose(w["wkv"], dim, hd), (hd, dim))
    tensors[lp + "attn.kv_norm.weight"] = arr(w["kv_norm"], (hd,))
    # wo_a tnsr [g, group_in, o_lora] -> upstream [g*o_lora, group_in].
    wo_a_up = [0.0] * (o_groups * o_lora * group_in)
    for g in range(o_groups):
        for i in range(group_in):
            for r in range(o_lora):
                src = (g * group_in + i) * o_lora + r  # tnsr
                dst = (g * o_lora + r) * group_in + i  # upstream row-major
                wo_a_up[dst] = w["wo_a"][src]
    tensors[lp + "attn.wo_a.weight"] = arr(wo_a_up, (o_groups * o_lora, group_in))
    tensors[lp + "attn.wo_b.weight"] = arr(
        _transpose(w["wo_b"], o_groups * o_lora, dim), (dim, o_groups * o_lora)
    )
    tensors[lp + "attn.attn_sink"] = arr(w["attn_sink"], (nh,))

    # moe gate stays row-major [experts, dim].
    tensors[lp + "ffn.gate.weight"] = arr(w["gate_weight"], (experts, dim))
    tensors[lp + "ffn.gate.bias"] = arr(w["correction_bias"], (experts,))
    for e in range(experts):
        ex = w["experts"][e]
        tensors[lp + f"ffn.experts.{e}.w1.weight"] = arr(_transpose(ex["w1"], dim, inter), (inter, dim))
        tensors[lp + f"ffn.experts.{e}.w2.weight"] = arr(_transpose(ex["w2"], inter, dim), (dim, inter))
        tensors[lp + f"ffn.experts.{e}.w3.weight"] = arr(_transpose(ex["w3"], dim, inter), (inter, dim))
    sh = w["shared_expert"]
    tensors[lp + "ffn.shared_experts.w1.weight"] = arr(_transpose(sh["w1"], dim, inter), (inter, dim))
    tensors[lp + "ffn.shared_experts.w2.weight"] = arr(_transpose(sh["w2"], inter, dim), (dim, inter))
    tensors[lp + "ffn.shared_experts.w3.weight"] = arr(_transpose(sh["w3"], dim, inter), (inter, dim))

    # hyper-connection tensors (copied as-is).
    tensors[lp + "hc_attn_fn"] = arr(w["hc_attn_fn"], (mix_hc, hc * dim))
    tensors[lp + "hc_attn_base"] = arr(w["hc_attn_base"], (mix_hc,))
    tensors[lp + "hc_attn_scale"] = arr(w["hc_attn_scale"], (3,))
    tensors[lp + "hc_ffn_fn"] = arr(w["hc_ffn_fn"], (mix_hc, hc * dim))
    tensors[lp + "hc_ffn_base"] = arr(w["hc_ffn_base"], (mix_hc,))
    tensors[lp + "hc_ffn_scale"] = arr(w["hc_ffn_scale"], (3,))

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(_tiny_config_json(), indent=2) + "\n")
    save_file(tensors, str(out_dir / "model.safetensors"))


# --------------------------------------------------------------------------
# Reference forward (shared *_ref helpers)
# --------------------------------------------------------------------------


def _reference_logits(fg) -> list[float]:
    """Compute the tiny model's last-position logits via ``*_ref`` helpers.

    Mirrors ``DeepSeekV41TextModel::try_forward_token_ids``:
    embedding -> expand_hc -> single block.forward -> hc_pre collapse ->
    rms_norm(final) -> linear(lm_head).  Weights come from ``_tiny_weights`` in
    tnsr ``[in, out]`` layout, exactly what the Rust forward runs on after load.
    """
    t = TINY
    dim, hc = t["dim"], t["hc_mult"]
    nh, hd = t["n_heads"], t["head_dim"]
    q_lora, o_lora, o_groups = t["q_lora"], t["o_lora"], t["o_groups"]
    inter, experts = t["inter"], t["experts"]
    batch, seqlen = 1, len(t["ids"])
    w = _tiny_weights()

    # embedding
    embedded: list[float] = []
    for token_id in t["ids"]:
        start = token_id * dim
        embedded.extend(w["embed"][start : start + dim])
    # expand_hc: replicate each token's D vector hc times
    h: list[float] = []
    for token in range(batch * seqlen):
        base = token * dim
        for _ in range(hc):
            h.extend(embedded[base : base + dim])
    pre_mix = [1.0, 0.0] * (batch * seqlen)  # identity_pre_mix for hc_mult=2

    cfg = {
        "mode": "sliding_window",
        "batch": batch,
        "seqlen": seqlen,
        "dim": dim,
        "hc_mult": hc,
        "hc_sinkhorn_iters": t["hc_sinkhorn_iters"],
        "hc_eps": t["hc_eps"],
        "rms_norm_eps": t["rms_eps"],
        "attention_shape": {
            "n_heads": nh,
            "head_dim": hd,
            "q_lora_rank": q_lora,
            "o_lora_rank": o_lora,
            "o_groups": o_groups,
        },
        "tokens": batch * seqlen,
        "experts": experts,
        "topk": t["topk"],
        "inter_dim": inter,
        "gate_temp": t["gate_temp"],
        "norm_topk_prob": t["norm_topk_prob"],
        "route_scale": t["route_scale"],
        "swiglu_limit": t["swiglu_limit"],
    }
    params = {
        "attn_norm": w["attn_norm"],
        "ffn_norm": w["ffn_norm"],
        "attention": {
            "wq_a": w["wq_a"],
            "q_norm": w["q_norm"],
            "wq_b": w["wq_b"],
            "wkv": w["wkv"],
            "kv_norm": w["kv_norm"],
            "wo_a": w["wo_a"],
            "wo_b": w["wo_b"],
            "attn_sink": w["attn_sink"],
        },
        "moe": {
            "gate_weight": w["gate_weight"],
            "correction_bias": w["correction_bias"],
            "experts": w["experts"],
            "shared_expert": w["shared_expert"],
        },
        "hc_attn_fn": w["hc_attn_fn"],
        "hc_attn_base": w["hc_attn_base"],
        "hc_attn_scale": w["hc_attn_scale"],
        "hc_ffn_fn": w["hc_ffn_fn"],
        "hc_ffn_base": w["hc_ffn_base"],
        "hc_ffn_scale": w["hc_ffn_scale"],
    }
    output, out_pre_mix, _state = fg.block_forward_ref(h, pre_mix, cfg, params, {})
    collapsed = fg.hc_pre_ref(output, out_pre_mix, batch, seqlen, hc, dim)
    collapsed = fg.fallback_rms_norm(collapsed, dim, w["final_norm"], t["rms_eps"])
    logits = fg.matmul_rows(collapsed, w["lm_head"], dim, t["vocab_size"])
    # last position row
    last = logits[(seqlen - 1) * t["vocab_size"] : seqlen * t["vocab_size"]]
    return last


# --------------------------------------------------------------------------
# Multimodal tiny checkpoint + reference
# --------------------------------------------------------------------------
#
# The multimodal tiny reference reuses the text `TINY` geometry (dim=4, hc_mult=2,
# 1 sliding-window layer, 2 experts, topk=1) and bolts on a tiny 1-layer ViT that
# matches the loader's `add_vision_tensors`/`tiny_vision_config_json` geometry:
#
#   vision_dim=8, n_heads=2 (head_dim=4, rope_dim=2, which `vision_cos_sin`
#   requires even), inter=3, patch_size=1 (patch_flat=3), downsample_ratio=1,
#   llm_dim=dim=4, n_layers=1.
#
# One 2x2 patch grid feeds `encode_image` and, with downsample_ratio=1, produces
# a 2x2 = 4-cell aligner output; the VL prompt fixture's span has exactly 4 IMAGE
# slots (`n_llm_h=n_llm_w=2`), so the geometry lines up with the Rust CLI path.
#
# encode_image runs the *faithful upstream* `vision.py` ViT + Aligner (pure
# torch), exactly as the vision op fixtures do, because tnsr's vision math is a
# direct port and matches upstream to tolerance.  The text seams below use the
# shared `*_ref` helpers (same reasoning as the text reference), extended here to
# apply `bias_vl` in the MoE selection on image tokens (upstream
# `Gate.forward(image_mask)`) — a term `moe_forward_ref` does not implement.

MM_VISION = dict(
    vision_dim=8,
    n_heads=2,
    inter=3,
    patch=1,
    theta=10000.0,
    n_layers=1,
    downsample=1,
    # A single 2x2 image, matching prompt_vl_fixture (n_llm_h=n_llm_w=2 -> 4 slots).
    n_vit_h=2,
    n_vit_w=2,
    n_llm_h=2,
    n_llm_w=2,
)

# The multimodal sequence: one leading text token, the 8-position image span
# (IMAGE_START, IMAGE, IMAGE, IMAGE_NEW_LINE, IMAGE, IMAGE, IMAGE_NEW_LINE,
# IMAGE_END), one trailing text token.  `token_types` uses vision_grid tags
# (TEXT=-1, IMAGE_START=0, IMAGE=1, IMAGE_NEW_LINE=2, IMAGE_END=3).  The `IMAGE`
# slots consume the 4 aligner rows in reading order.
MM_IMAGE_START = 0
MM_IMAGE = 1
MM_IMAGE_NEW_LINE = 2
MM_IMAGE_END = 3
MM_SPAN_TYPES = [MM_IMAGE_START, MM_IMAGE, MM_IMAGE, MM_IMAGE_NEW_LINE, MM_IMAGE, MM_IMAGE, MM_IMAGE_NEW_LINE, MM_IMAGE_END]
MM_IMAGE_TOKEN_ID = 4
MM_TEXT_LEAD_ID = 3
MM_TEXT_TRAIL_ID = 2
MM_IMAGE_START_POS = 1  # one leading text token


def _mm_ids_and_types() -> tuple[list[int], list[int]]:
    """The (token_ids, token_types) the Rust CLI is fed for the tiny MM parity."""
    ids = [MM_TEXT_LEAD_ID]
    types = [-1]  # TEXT
    for ty in MM_SPAN_TYPES:
        ids.append(MM_IMAGE_TOKEN_ID)  # every image-span slot carries the placeholder id
        types.append(ty)
    ids.append(MM_TEXT_TRAIL_ID)
    types.append(-1)
    return ids, types


def _mm_patches() -> list[float]:
    """Deterministic patches `[n_vit_h*n_vit_w, patch_flat]` for the one image."""
    v = MM_VISION
    n_patch = v["n_vit_h"] * v["n_vit_w"]
    patch_flat = 3 * v["patch"] * v["patch"]
    return _ramp(n_patch * patch_flat, 9, 4, 5.0)


def _mm_vision_weights() -> dict:
    """Tiny ViT + Aligner + delimiter + bias_vl weights (torch `[out, in]`)."""
    v = MM_VISION
    dim = v["vision_dim"]
    inter = v["inter"]
    patch_flat = 3 * v["patch"] * v["patch"]
    llm_dim = TINY["dim"]
    r = v["downsample"]
    in_dim = dim * r * r
    return {
        "proj_w": _ramp(dim * patch_flat, 7, 3, 8.0),  # [vision_dim, patch_flat]
        "proj_b": _ramp(dim, 5, 2, 4.0),
        "wqkv": _ramp(3 * dim * dim, 11, 5, 9.0),  # [3*vision_dim, vision_dim]
        "wqkv_b": _ramp(3 * dim, 5, 2, 6.0),
        "wo": _ramp(dim * dim, 7, 3, 7.0),  # [vision_dim, vision_dim]
        "wo_b": _ramp(dim, 5, 2, 5.0),
        "norm1": _ramp(dim, 5, 2, 8.0),
        "norm2": _ramp(dim, 7, 3, 9.0),
        "mlp_w1": _ramp(2 * inter * dim, 13, 6, 10.0),  # [2*inter, vision_dim]
        "mlp_w2": _ramp(dim * inter, 11, 5, 9.0),  # [vision_dim, inter]
        "vit_norm": _ramp(dim, 5, 2, 7.0),
        "al_w1": _ramp(llm_dim * in_dim, 11, 5, 9.0),  # [llm_dim, in_dim]
        "al_w1_b": _ramp(llm_dim, 5, 2, 6.0),
        "al_w2": _ramp(llm_dim * llm_dim, 7, 3, 7.0),  # [llm_dim, llm_dim]
        "al_w2_b": _ramp(llm_dim, 5, 2, 5.0),
        "image_start": [1.0, 2.0, 3.0, 4.0][:llm_dim],
        "image_end": [-1.0, -2.0, -3.0, -4.0][:llm_dim],
        "image_newline": [0.5, -0.5, 0.5, -0.5][:llm_dim],
        # bias_vl deliberately favours the other expert so image tokens diverge
        # from the text correction bias, exercising the masked selection path.
        "bias_vl": [-0.5, 0.5][: TINY["experts"]],
    }


def _emit_tiny_mm_checkpoint(out_dir: Path) -> None:
    """Write a vision-enabled tiny checkpoint the multimodal loader can read.

    Reuses `_emit_tiny_checkpoint` for the text tensors, then appends the vision
    tower / aligner / delimiter / `bias_vl` tensors and rewrites `config.json`
    with the vision fields enabled.
    """
    import numpy as np  # local import; safetensors needs numpy arrays
    from safetensors.numpy import load_file, save_file

    # Start from the text tiny checkpoint, then augment it.
    _emit_tiny_checkpoint(out_dir)
    v = MM_VISION
    dim = v["vision_dim"]
    inter = v["inter"]
    patch_flat = 3 * v["patch"] * v["patch"]
    llm_dim = TINY["dim"]
    r = v["downsample"]
    in_dim = dim * r * r
    vw = _mm_vision_weights()

    def arr(flat, shape):
        return np.asarray(flat, dtype=np.float32).reshape(shape)

    tensors = dict(load_file(str(out_dir / "model.safetensors")))
    tensors["vision.patch_embed.proj.weight"] = arr(vw["proj_w"], (dim, patch_flat))
    tensors["vision.patch_embed.proj.bias"] = arr(vw["proj_b"], (dim,))
    tensors["vision.blocks.0.norm1.weight"] = arr(vw["norm1"], (dim,))
    tensors["vision.blocks.0.attn.wqkv.weight"] = arr(vw["wqkv"], (3 * dim, dim))
    tensors["vision.blocks.0.attn.wqkv.bias"] = arr(vw["wqkv_b"], (3 * dim,))
    tensors["vision.blocks.0.attn.wo.weight"] = arr(vw["wo"], (dim, dim))
    tensors["vision.blocks.0.attn.wo.bias"] = arr(vw["wo_b"], (dim,))
    tensors["vision.blocks.0.norm2.weight"] = arr(vw["norm2"], (dim,))
    tensors["vision.blocks.0.mlp.w1.weight"] = arr(vw["mlp_w1"], (2 * inter, dim))
    tensors["vision.blocks.0.mlp.w2.weight"] = arr(vw["mlp_w2"], (dim, inter))
    tensors["vision.norm.weight"] = arr(vw["vit_norm"], (dim,))
    tensors["aligner.w1.weight"] = arr(vw["al_w1"], (llm_dim, in_dim))
    tensors["aligner.w1.bias"] = arr(vw["al_w1_b"], (llm_dim,))
    tensors["aligner.w2.weight"] = arr(vw["al_w2"], (llm_dim, llm_dim))
    tensors["aligner.w2.bias"] = arr(vw["al_w2_b"], (llm_dim,))
    tensors["image_start"] = arr(vw["image_start"], (llm_dim,))
    tensors["image_end"] = arr(vw["image_end"], (llm_dim,))
    tensors["image_newline"] = arr(vw["image_newline"], (llm_dim,))
    tensors["layers.0.ffn.gate.bias_vl"] = arr(vw["bias_vl"], (TINY["experts"],))
    save_file(tensors, str(out_dir / "model.safetensors"))

    cfg = _tiny_config_json()
    cfg.update(
        {
            "vision_n_layers": v["n_layers"],
            "vision_dim": dim,
            "vision_n_heads": v["n_heads"],
            "vision_inter_dim": inter,
            "vision_patch_size": v["patch"],
            "vision_rope_theta": v["theta"],
            "vision_downsample_ratio": r,
            "vision_max_n_token": 16,
            "vision_min_pixels": 1,
        }
    )
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")


def _mm_encode_image(fg, vision_module) -> list[float]:
    """Run the faithful upstream ViT + Aligner on the tiny image, returning
    aligner rows `[n_cell_h*n_cell_w, llm_dim]` flattened (reading order)."""
    import torch

    v = MM_VISION
    dim = v["vision_dim"]
    n_heads = v["n_heads"]
    inter = v["inter"]
    patch = v["patch"]
    theta = v["theta"]
    n_layers = v["n_layers"]
    downsample = v["downsample"]
    llm_dim = TINY["dim"]
    patch_flat = 3 * patch * patch
    n_h, n_w = v["n_vit_h"], v["n_vit_w"]
    n_patch = n_h * n_w
    vw = _mm_vision_weights()

    args = fg._EncodeImageArgs(dim, n_heads, inter, patch, theta, n_layers, downsample, llm_dim)

    def as_param(data, shape):
        return torch.nn.Parameter(torch.tensor(data, dtype=torch.float32).reshape(shape))

    vit = vision_module.ViT(args)
    vit.patch_embed.proj.weight = as_param(vw["proj_w"], (dim, patch_flat))
    vit.patch_embed.proj.bias = as_param(vw["proj_b"], (dim,))
    blk = vit.blocks[0]
    blk.attn.wqkv.weight = as_param(vw["wqkv"], (3 * dim, dim))
    blk.attn.wqkv.bias = as_param(vw["wqkv_b"], (3 * dim,))
    blk.attn.wo.weight = as_param(vw["wo"], (dim, dim))
    blk.attn.wo.bias = as_param(vw["wo_b"], (dim,))
    blk.norm1.weight = as_param(vw["norm1"], (dim,))
    blk.norm2.weight = as_param(vw["norm2"], (dim,))
    blk.mlp.w1.weight = as_param(vw["mlp_w1"], (2 * inter, dim))
    blk.mlp.w2.weight = as_param(vw["mlp_w2"], (dim, inter))
    vit.norm.weight = as_param(vw["vit_norm"], (dim,))

    aligner = vision_module.Aligner(args)
    in_dim = dim * downsample * downsample
    aligner.w1.weight = as_param(vw["al_w1"], (llm_dim, in_dim))
    aligner.w1.bias = as_param(vw["al_w1_b"], (llm_dim,))
    aligner.w2.weight = as_param(vw["al_w2"], (llm_dim, llm_dim))
    aligner.w2.bias = as_param(vw["al_w2_b"], (llm_dim,))

    patches_t = torch.tensor(_mm_patches(), dtype=torch.float32).reshape(n_patch, patch_flat)
    with torch.no_grad():
        vit_rows = vit(patches_t, n_h, n_w)
        aligned = aligner(vit_rows, n_h, n_w)
    return fg.flatten_nested(fg.tensor_to_nested_list(aligned))


def _mm_moe_forward_ref(fg, x, tokens, dim, cfg, moe_params, image_mask, bias_vl):
    """Image-mask-aware MoE forward.

    Mirrors `moe_forward_ref` but applies `bias_vl` to the *selection* bias on
    image tokens (`image_mask[t]`), exactly as upstream `Gate.forward(image_mask)`
    / tnsr `select_experts_masked` do.  Routing weights always come from the
    unbiased scores, so the mask changes selection only.
    """
    experts = cfg["experts"]
    topk = cfg["topk"]
    scores = fg.fallback_sqrtsoftplus_scores(
        x, moe_params["gate_weight"], cfg["gate_temp"], tokens, dim, experts
    )
    # Per-token expert selection with the effective (masked) bias.
    correction_bias = moe_params["correction_bias"]
    indices: list[int] = []
    for t in range(tokens):
        base = t * experts
        bias = bias_vl if (image_mask[t] and bias_vl is not None) else correction_bias
        ranked = sorted(range(experts), key=lambda e: (-(scores[base + e] + bias[e]), e))
        indices.extend(ranked[:topk])
    weights = fg.fallback_route_weights(
        scores, indices, tokens, experts, topk, cfg["norm_topk_prob"], cfg["route_scale"]
    )
    out = [0.0] * (tokens * dim)
    for t in range(tokens):
        row = x[t * dim : (t + 1) * dim]
        for slot in range(topk):
            expert_idx = indices[t * topk + slot]
            expert = moe_params["experts"][expert_idx]
            expert_out = fg.expert_forward_row(
                row, expert["w1"], expert["w2"], expert["w3"], dim, cfg["inter_dim"], cfg["swiglu_limit"]
            )
            for d in range(dim):
                out[t * dim + d] += weights[t * topk + slot] * expert_out[d]
        shared = moe_params["shared_expert"]
        shared_out = fg.expert_forward_row(
            row, shared["w1"], shared["w2"], shared["w3"], dim, cfg["inter_dim"], cfg["swiglu_limit"]
        )
        for d in range(dim):
            out[t * dim + d] += shared_out[d]
    return out


def _mm_block_forward_ref(fg, x, pre_mix, cfg, params, image_mask):
    """One block forward mirroring `try_forward_multimodal`'s per-layer step.

    Same attention/HC math as `block_forward_ref` (sliding-window mode), but the
    FFN MoE is image-mask-aware.  The tiny config has `engram_layer_ids: []`, so
    the engram branch (which would consume `~image_mask`) is inactive here; the
    image mask matters only for MoE selection.
    """
    batch = cfg["batch"]
    seqlen = cfg["seqlen"]
    hc_mult = cfg["hc_mult"]
    dim = cfg["dim"]
    eps = cfg["hc_eps"]
    state: dict = {"compressed_kv": None, "topk_indices": None, "consumed_sparse_indices": False}

    residual = list(x)
    attn_mix = fg.hc_mixes_ref(
        residual, params["hc_attn_fn"], params["hc_attn_scale"], params["hc_attn_base"],
        hc_mult, dim, cfg["hc_sinkhorn_iters"], eps,
    )
    attn_in = fg.hc_pre_ref(residual, pre_mix, batch, seqlen, hc_mult, dim)
    attn_in = fg.fallback_rms_norm(attn_in, dim, params["attn_norm"], cfg["rms_norm_eps"])
    attn_out = fg.attention_forward_ref(
        attn_in, batch, seqlen, dim, cfg["attention_shape"], params["attention"],
        state, cfg["rms_norm_eps"], False, False,
    )
    x = fg.hc_post_ref(attn_out, residual, attn_mix["post"], attn_mix["comb"], batch, seqlen, hc_mult, dim)

    residual = list(x)
    ffn_mix = fg.hc_mixes_ref(
        residual, params["hc_ffn_fn"], params["hc_ffn_scale"], params["hc_ffn_base"],
        hc_mult, dim, cfg["hc_sinkhorn_iters"], eps,
    )
    ffn_in = fg.hc_pre_ref(residual, attn_mix["pre"], batch, seqlen, hc_mult, dim)
    ffn_in = fg.fallback_rms_norm(ffn_in, dim, params["ffn_norm"], cfg["rms_norm_eps"])
    ffn_out = _mm_moe_forward_ref(
        fg, ffn_in, batch * seqlen, dim, cfg, params["moe"], image_mask, params["moe"].get("bias_vl")
    )
    x = fg.hc_post_ref(ffn_out, residual, ffn_mix["post"], ffn_mix["comb"], batch, seqlen, hc_mult, dim)
    return x, ffn_mix["pre"]


def _reference_multimodal_logits(fg, vision_module) -> tuple[list[float], list[int], list[int]]:
    """Compute the tiny multimodal model's last-position logits.

    Mirrors `DeepSeekV41TextModel::try_forward_multimodal`: embed the ids, overwrite
    each image span (delimiters + aligner rows) BEFORE the HC expansion, expand,
    run the (single) block with image-mask-aware MoE selection, collapse, final
    RMSNorm, and lm_head.  Returns `(last_logits, token_ids, token_types)`.
    """
    t = TINY
    dim, hc = t["dim"], t["hc_mult"]
    nh, hd = t["n_heads"], t["head_dim"]
    q_lora, o_lora, o_groups = t["q_lora"], t["o_lora"], t["o_groups"]
    inter, experts = t["inter"], t["experts"]
    w = _tiny_weights()
    vw = _mm_vision_weights()

    ids, token_types = _mm_ids_and_types()
    batch, seqlen = 1, len(ids)

    # Embedding, then overwrite the image span (mirror merge_image_embeddings).
    embedded: list[float] = []
    for token_id in ids:
        start = token_id * dim
        embedded.extend(w["embed"][start : start + dim])
    aligner_rows = _mm_encode_image(fg, vision_module)
    aligner_off = 0
    for k, ty in enumerate(MM_SPAN_TYPES):
        pos = MM_IMAGE_START_POS + k
        dst = pos * dim
        if ty == MM_IMAGE_START:
            embedded[dst : dst + dim] = vw["image_start"]
        elif ty == MM_IMAGE_END:
            embedded[dst : dst + dim] = vw["image_end"]
        elif ty == MM_IMAGE_NEW_LINE:
            embedded[dst : dst + dim] = vw["image_newline"]
        elif ty == MM_IMAGE:
            embedded[dst : dst + dim] = aligner_rows[aligner_off * dim : (aligner_off + 1) * dim]
            aligner_off += 1
    image_mask = [ty >= 0 for ty in token_types]

    # expand_hc: replicate each token's D vector hc times.
    h: list[float] = []
    for token in range(batch * seqlen):
        base = token * dim
        for _ in range(hc):
            h.extend(embedded[base : base + dim])
    pre_mix = [1.0, 0.0] * (batch * seqlen)  # identity_pre_mix for hc_mult=2

    cfg = {
        "batch": batch,
        "seqlen": seqlen,
        "dim": dim,
        "hc_mult": hc,
        "hc_sinkhorn_iters": t["hc_sinkhorn_iters"],
        "hc_eps": t["hc_eps"],
        "rms_norm_eps": t["rms_eps"],
        "attention_shape": {
            "n_heads": nh,
            "head_dim": hd,
            "q_lora_rank": q_lora,
            "o_lora_rank": o_lora,
            "o_groups": o_groups,
        },
        "tokens": batch * seqlen,
        "experts": experts,
        "topk": t["topk"],
        "inter_dim": inter,
        "gate_temp": t["gate_temp"],
        "norm_topk_prob": t["norm_topk_prob"],
        "route_scale": t["route_scale"],
        "swiglu_limit": t["swiglu_limit"],
    }
    params = {
        "attn_norm": w["attn_norm"],
        "ffn_norm": w["ffn_norm"],
        "attention": {
            "wq_a": w["wq_a"],
            "q_norm": w["q_norm"],
            "wq_b": w["wq_b"],
            "wkv": w["wkv"],
            "kv_norm": w["kv_norm"],
            "wo_a": w["wo_a"],
            "wo_b": w["wo_b"],
            "attn_sink": w["attn_sink"],
        },
        "moe": {
            "gate_weight": w["gate_weight"],
            "correction_bias": w["correction_bias"],
            "bias_vl": vw["bias_vl"],
            "experts": w["experts"],
            "shared_expert": w["shared_expert"],
        },
        "hc_attn_fn": w["hc_attn_fn"],
        "hc_attn_base": w["hc_attn_base"],
        "hc_attn_scale": w["hc_attn_scale"],
        "hc_ffn_fn": w["hc_ffn_fn"],
        "hc_ffn_base": w["hc_ffn_base"],
        "hc_ffn_scale": w["hc_ffn_scale"],
    }
    output, _out_pre_mix = _mm_block_forward_ref(fg, h, pre_mix, cfg, params, image_mask)
    collapsed = fg.hc_pre_ref(output, _out_pre_mix, batch, seqlen, hc, dim)
    collapsed = fg.fallback_rms_norm(collapsed, dim, w["final_norm"], t["rms_eps"])
    logits = fg.matmul_rows(collapsed, w["lm_head"], dim, t["vocab_size"])
    last = logits[(seqlen - 1) * t["vocab_size"] : seqlen * t["vocab_size"]]
    return last, ids, token_types


def _write_image_patches_json(path: Path) -> None:
    """Write the `--image-patches` JSON the Rust CLI consumes for the tiny image."""
    v = MM_VISION
    record = {
        "start": MM_IMAGE_START_POS,
        "n_vit_h": v["n_vit_h"],
        "n_vit_w": v["n_vit_w"],
        "n_llm_h": v["n_llm_h"],
        "n_llm_w": v["n_llm_w"],
        "patches": _mm_patches(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([record]) + "\n")


# --------------------------------------------------------------------------
# DSpark tiny reference (composed forward_spec)
# --------------------------------------------------------------------------
#
# The DSpark tiny reference reuses the *exact* math of
# `deepseek_v41_fixture_gen.generate_dspark_model` (the same helper composition
# the Rust `DeepSeekV41DsparkHead::forward_spec` is checked against in
# `deepseek_v41_model_test.rs`).  Rather than duplicate that composition here, we
# invoke the fixture generator into a scratch dir, read back the tiny model
# fixture it wrote, and re-serialize the `output_ids` / `logits` / `confidence`
# in the parity JSON schema the comparator consumes.  This keeps a single source
# of numeric truth for the DSpark forward.


def _dspark_reference_payload(fg, prompt: str) -> dict:
    """Compute the tiny DSpark reference via ``generate_dspark_model``.

    Returns the parity payload ``{output_ids, logits, confidence, ...}``.  The
    numbers are the same ones ``dspark_tiny_model_fixture.json`` carries, so the
    Rust ``forward_spec`` parity test and this Python reference share one source
    of truth.
    """
    import tempfile

    backend, kernel_patch = _record_upstream_backend(_repo_root_from_here(), fg)
    with tempfile.TemporaryDirectory(prefix="dsv41_dspark_ref_") as tmp:
        out_dir = Path(tmp)
        # `generate_dspark_model` re-derives the tiny stage from the shared
        # `*_ref` helpers and writes `dspark_tiny_model_fixture.json`.
        fg.generate_dspark_model(out_dir, None, "reference-recompute")
        fixture = json.loads((out_dir / "dspark_tiny_model_fixture.json").read_text())

    expected = fixture["expected"]
    inp = fixture["input"]
    return {
        "token_ids": list(inp["input_ids"]),
        "prompt": prompt,
        "vocab_size": inp["vocab_size"],
        "model_type": "deepseek_v41_dspark",
        "logits": [float(v) for v in expected["logits"]],
        "logits_shape": list(expected["logits_shape"]),
        "output_ids": list(expected["output_ids"]),
        "confidence": [float(v) for v in expected["confidence"]],
        "reference_backend": backend,
        "kernel_patch": kernel_patch,
    }


def _detect_cuda_backend() -> str:
    """Record whether a CUDA torch runtime is available (B200 path).

    The tiny reference runs the same math on CPU or GPU; this only records the
    detected device in metadata so a GPU run is provenance-distinguishable.
    """
    try:
        import torch

        if torch.cuda.is_available():
            return f"cuda:{torch.cuda.get_device_name(0)}"
        return "cpu"
    except Exception:  # noqa: BLE001
        return "cpu-no-torch"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, default=None)
    p.add_argument(
        "--emit-tiny-checkpoint",
        type=Path,
        default=None,
        help="write a tiny converted-style checkpoint (config.json + model.safetensors) here",
    )
    p.add_argument(
        "--emit-tiny-mm-checkpoint",
        type=Path,
        default=None,
        help="write a tiny vision-enabled checkpoint the multimodal loader can read here",
    )
    p.add_argument(
        "--emit-image-patches",
        type=Path,
        default=None,
        help="write the tiny --image-patches JSON the Rust CLI consumes here",
    )
    p.add_argument(
        "--multimodal",
        action="store_true",
        help="compute the tiny multimodal reference logits instead of text-only",
    )
    p.add_argument(
        "--dspark",
        action="store_true",
        help="compute the tiny DSpark forward_spec reference (output_ids/logits/confidence)",
    )
    p.add_argument("--out", type=Path, default=None, help="write reference logits JSON here")
    p.add_argument(
        "--token-ids",
        type=str,
        default=None,
        help="comma-separated ids to record (default: tiny fixture ids)",
    )
    p.add_argument("--prompt", type=str, default="")
    p.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="real checkpoint dir (NOT supported: this tiny reference only "
        "reproduces the tiny helper-based logits)",
    )
    args = p.parse_args()

    if args.model_dir is not None:
        # Real-checkpoint reference logits require an upstream GPU forward that
        # is not available on this CPU host; fail loudly rather than silently
        # emitting tiny logits that would not match a real checkpoint.
        sys.stderr.write(
            "error: --model-dir is not supported by the tiny reference; "
            "real-checkpoint parity is out of scope on this host.\n"
        )
        return 3

    repo_root = (args.repo_root or _repo_root_from_here()).resolve()
    fg = _import_fixture_gen(repo_root)

    if args.emit_tiny_checkpoint is not None:
        _emit_tiny_checkpoint(args.emit_tiny_checkpoint.resolve())
        print(f"wrote tiny checkpoint to {args.emit_tiny_checkpoint}")

    if args.emit_tiny_mm_checkpoint is not None:
        _emit_tiny_mm_checkpoint(args.emit_tiny_mm_checkpoint.resolve())
        print(f"wrote tiny multimodal checkpoint to {args.emit_tiny_mm_checkpoint}")

    if args.emit_image_patches is not None:
        _write_image_patches_json(args.emit_image_patches.resolve())
        print(f"wrote tiny image patches to {args.emit_image_patches}")

    if args.out is None:
        return 0

    if args.multimodal:
        vision_py = (
            repo_root
            / "ferric_continuum/tnsr/third_party/deepseek_v41/upstream/inference/vision.py"
        )
        vision_module, vision_status = fg.import_upstream_vision_by_path(vision_py)
        if vision_module is None:
            sys.stderr.write(
                f"error: could not import upstream vision.py ({vision_status}); "
                "multimodal reference requires a faithful upstream ViT/Aligner forward.\n"
            )
            return 4
        cuda_backend = _detect_cuda_backend()
        logits, ids, token_types = _reference_multimodal_logits(fg, vision_module)
        payload = {
            "token_ids": ids,
            "token_types": token_types,
            "prompt": args.prompt,
            "vocab_size": TINY["vocab_size"],
            "model_type": "deepseek_v41_multimodal",
            "logits": logits,
            "reference_backend": {
                "device": cuda_backend,
                "vision_import": vision_status,
            },
            "kernel_patch": "none:pure-torch-vit-aligner",
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload) + "\n")
        print(
            f"wrote multimodal reference logits ({len(logits)}) to {args.out} "
            f"[device={cuda_backend} vision={vision_status}]"
        )
        return 0

    if args.token_ids is not None:
        ids = [int(x) for x in args.token_ids.split(",") if x.strip()]
    else:
        ids = list(TINY["ids"])

    if args.dspark:
        payload = _dspark_reference_payload(fg, args.prompt)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload) + "\n")
        print(
            f"wrote dspark reference (output_ids={payload['output_ids']}, "
            f"{len(payload['logits'])} logits, {len(payload['confidence'])} "
            f"confidence) to {args.out} [backend={payload['reference_backend']}]"
        )
        return 0

    backend, kernel_patch = _record_upstream_backend(repo_root, fg)
    logits = _reference_logits(fg)
    payload = {
        "token_ids": ids,
        "prompt": args.prompt,
        "vocab_size": TINY["vocab_size"],
        "model_type": "deepseek_v41_text",
        "logits": logits,
        "reference_backend": backend,
        "kernel_patch": kernel_patch,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload) + "\n")
    print(f"wrote reference logits ({len(logits)}) to {args.out} [backend={backend}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
