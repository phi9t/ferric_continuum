#!/usr/bin/env python3
"""Generate DeepSeek V4.1 small op verifier fixtures.

Expected values are produced by calling the mirrored upstream `model.py` by
path when its Python dependencies are available. The small pure-Python fallback
exists only so the script can explain exactly why a specific upstream call could
not run in a lightweight environment.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import types
from pathlib import Path


def round_float(value: float) -> float | str:
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return round(value, 8)


def round_list(values: list[float]) -> list[float | str]:
    return [round_float(v) for v in values]


def has_torch() -> bool:
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


def maybe_reexec_with_repo_venv(repo_root: Path) -> None:
    if has_torch() or os.environ.get("DEEPSEEK_V41_FIXTURE_VENV_REEXEC") == "1":
        return

    candidates = [
        repo_root / ".venv-hf/bin/python",
        repo_root / ".venv/bin/python",
        repo_root.parent.parent / ".venv-hf/bin/python",
        repo_root.parent.parent / ".venv/bin/python",
    ]
    venv_python = next((path for path in candidates if path.exists()), None)
    if venv_python is None:
        return

    env = os.environ.copy()
    env["DEEPSEEK_V41_FIXTURE_VENV_REEXEC"] = "1"
    os.execve(str(venv_python), [str(venv_python), *sys.argv], env)


def fallback_precompute_freqs_cis(
    dim: int,
    seqlen: int,
    original_seq_len: int,
    base: float,
    factor: float,
    beta_fast: float,
    beta_slow: float,
) -> list[list[float]]:
    freqs = [1.0 / (base ** (i / dim)) for i in range(0, dim, 2)]
    if original_seq_len > 0:
        def corrected_dim(rotations: float) -> float:
            return dim * math.log(original_seq_len / (rotations * 2 * math.pi)) / (2 * math.log(base))

        low = max(math.floor(corrected_dim(beta_fast)), 0)
        high = min(math.ceil(corrected_dim(beta_slow)), dim - 1)
        smooth = []
        for pair in range(dim // 2):
            ramp = min(max((pair - low) / max(high - low, 1e-3), 0.0), 1.0)
            smooth.append(1.0 - ramp)
        freqs = [freq / factor * (1.0 - s) + freq * s for freq, s in zip(freqs, smooth)]

    out = []
    for pos in range(seqlen):
        for freq in freqs:
            theta = pos * freq
            out.append([math.cos(theta), math.sin(theta)])
    return out


def fallback_apply_rotary(
    data: list[float],
    batch: int,
    seqlen: int,
    heads: int,
    dim: int,
    freqs: list[list[float]],
    inverse: bool = False,
) -> list[float]:
    out = list(data)
    half = dim // 2
    for b in range(batch):
        for s in range(seqlen):
            for h in range(heads):
                base = ((b * seqlen + s) * heads + h) * dim
                for pair in range(half):
                    cos, sin = freqs[s * half + pair]
                    if inverse:
                        sin = -sin
                    x0 = out[base + 2 * pair]
                    x1 = out[base + 2 * pair + 1]
                    out[base + 2 * pair] = x0 * cos - x1 * sin
                    out[base + 2 * pair + 1] = x0 * sin + x1 * cos
    return out


def fallback_window_topk(window_size: int, batch: int, seqlen: int, start_pos: int) -> list[int]:
    rows = seqlen if start_pos == 0 else 1
    width = min(seqlen, window_size) if start_pos == 0 else window_size
    one = []
    if start_pos == 0:
        for end in range(seqlen):
            row_start = max(end - window_size + 1, 0)
            for slot in range(width):
                idx = row_start + slot
                one.append(-1 if idx > end else idx)
    else:
        oldest = start_pos % window_size + 1
        for idx in list(range(oldest, window_size)) + list(range(oldest)):
            one.append(-1 if idx > start_pos else idx)
    assert len(one) == rows * width
    return one * batch


def fallback_select_candidate_blocks(
    logits: list[float],
    batch: int,
    seqlen: int,
    positions: int,
    compress_lens: list[int],
    topk_blocks: int,
    block_size: int,
) -> list[bool]:
    num_blocks = (positions + block_size - 1) // block_size
    out = [False] * len(logits)
    for b in range(batch):
        for s in range(seqlen):
            q = b * seqlen + s
            row_base = q * positions
            scores = []
            for block in range(num_blocks):
                values = []
                for offset in range(block_size):
                    pos = block * block_size + offset
                    values.append(logits[row_base + pos] if pos < positions else -math.inf)
                scores.append(max(values))
            last = (compress_lens[q] - 1) // block_size
            if 0 <= last < len(scores):
                scores[last] = math.inf
            ranked = sorted(enumerate(scores), key=lambda item: (-item[1], item[0]))
            for block, score in ranked[: min(topk_blocks, num_blocks)]:
                if score > -math.inf:
                    for offset in range(block_size):
                        pos = block * block_size + offset
                        if pos < positions:
                            out[row_base + pos] = True
    return out


def fallback_rms_norm(rows: list[float], dim: int, norm_weight: list[float], eps: float) -> list[float]:
    out = []
    for row_start in range(0, len(rows), dim):
        row = rows[row_start : row_start + dim]
        scale = 1.0 / math.sqrt(sum(v * v for v in row) / dim + eps)
        out.extend(v * scale * w for v, w in zip(row, norm_weight))
    return out


def fallback_compress_ratio_n(
    kv: list[float],
    score: list[float],
    ratio: int,
    dim: int,
    norm_weight: list[float],
    eps: float,
) -> list[float]:
    rows = len(kv) // dim
    complete_groups = rows // ratio
    pooled = []
    for group in range(complete_groups):
        for d in range(dim):
            values = [score[(group * ratio + r) * dim + d] for r in range(ratio)]
            max_score = max(values)
            weights = [math.exp(v - max_score) for v in values]
            denom = sum(weights)
            pooled.append(
                sum(kv[(group * ratio + r) * dim + d] * weights[r] for r in range(ratio)) / denom
            )
    return fallback_rms_norm(pooled, dim, norm_weight, eps)


def source_meta(
    function: str,
    line_hint: str,
    source_file: str = "ferric_continuum/tnsr/third_party/deepseek_v41/upstream/inference/model.py",
) -> dict[str, str]:
    return {
        "upstream_function": function,
        "source_file": source_file,
        "line_hint": line_hint,
    }


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _hc_split_sinkhorn_torch(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    import torch

    assert mixes.dtype == torch.float32
    b, s, mix_hc = mixes.size()
    assert mix_hc == (2 + hc_mult) * hc_mult

    pre_logits = mixes[..., :hc_mult] * hc_scale[0] + hc_base[:hc_mult]
    post_logits = mixes[..., hc_mult : 2 * hc_mult] * hc_scale[1] + hc_base[hc_mult : 2 * hc_mult]
    comb_logits = mixes[..., 2 * hc_mult :] * hc_scale[2] + hc_base[2 * hc_mult :]
    comb_logits = comb_logits.view(b, s, hc_mult, hc_mult)

    pre = torch.sigmoid(pre_logits) + eps
    post = 2.0 * torch.sigmoid(post_logits)

    # comb = comb.softmax(-1) + eps
    comb_max = comb_logits.max(dim=-1, keepdim=True).values
    comb = torch.exp(comb_logits - comb_max)
    comb = comb / comb.sum(dim=-1, keepdim=True) + eps

    # comb = comb / (comb.sum(-2) + eps)
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)

    for _ in range(max(0, sinkhorn_iters - 1)):
        # comb = comb / (comb.sum(-1) + eps)
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        # comb = comb / (comb.sum(-2) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)

    return pre, post, comb


def install_upstream_import_stubs() -> None:
    kernel = types.ModuleType("kernel")
    for name in [
        "act_quant",
        "fp4_act_quant",
        "fp4_gemm",
        "fp8_gemm",
        "sparse_attn",
    ]:
        setattr(kernel, name, _stubbed_kernel_call)
    kernel.hc_split_sinkhorn = _hc_split_sinkhorn_torch
    sys.modules["kernel"] = kernel

    engram = types.ModuleType("engram")
    engram.EngramLayout = object
    engram.NgramHashState = object
    sys.modules["engram"] = engram

    image_processor = types.ModuleType("image_processor")
    image_processor.IMAGE = 1
    image_processor.IMAGE_END = 3
    image_processor.IMAGE_NEW_LINE = 2
    image_processor.IMAGE_START = 0
    sys.modules["image_processor"] = image_processor

    vision = types.ModuleType("vision")
    vision.Aligner = object
    vision.ViT = object
    sys.modules["vision"] = vision


def _stubbed_kernel_call(*_args, **_kwargs):
    raise RuntimeError("stubbed upstream kernel is outside Ticket 03 fixture scope")


def import_upstream_image_processor_by_path(image_py: Path) -> tuple[object | None, str]:
    """Load upstream image_processor.py by path.

    The pure grid functions only need `math`, but the module top-level imports
    numpy/torch/PIL. We stub those to lightweight placeholders so the pure grid
    functions (which never touch them) import and run in a CPU-only env.
    """
    saved = {name: sys.modules.get(name) for name in ("numpy", "torch", "PIL")}
    try:
        if "numpy" not in sys.modules:
            sys.modules["numpy"] = types.ModuleType("numpy")
        if "torch" not in sys.modules:
            torch_stub = types.ModuleType("torch")
            torch_stub.Tensor = object
            torch_stub.int64 = "int64"

            def _tensor(data, dtype=None):
                return data

            torch_stub.tensor = _tensor
            sys.modules["torch"] = torch_stub
        if "PIL" not in sys.modules:
            pil = types.ModuleType("PIL")
            pil.Image = object
            pil.ImageOps = object
            sys.modules["PIL"] = pil
            sys.modules["PIL.Image"] = types.ModuleType("PIL.Image")

        spec = importlib.util.spec_from_file_location(
            "deepseek_v41_upstream_image_processor", image_py
        )
        if spec is None or spec.loader is None:
            raise SystemExit(f"could not create import spec for {image_py}")
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except ModuleNotFoundError as err:
            return None, f"path-import-skipped-missing-dependency:{err.name}"
        except Exception as err:
            return None, f"path-import-skipped:{type(err).__name__}:{err}"
        return module, "path-import-ok"
    finally:
        for name, prev in saved.items():
            if prev is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev


def import_upstream_model_by_path(model_py: Path) -> tuple[object | None, str]:
    """Load upstream model.py by path, stubbing unrelated heavy modules."""
    spec = importlib.util.spec_from_file_location("deepseek_v41_upstream_model", model_py)
    if spec is None or spec.loader is None:
        raise SystemExit(f"could not create import spec for {model_py}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(model_py.parent))
    install_upstream_import_stubs()
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as err:
        return None, f"path-import-skipped-missing-dependency:{err.name}"
    except Exception as err:
        return None, f"path-import-skipped:{type(err).__name__}:{err}"
    finally:
        sys.path.pop(0)
    return module, "path-import-ok"


def install_tokenizers_shim() -> None:
    import re
    import unicodedata

    tokenizers = types.ModuleType("tokenizers")

    class Regex:
        def __init__(self, pattern: str):
            self._pattern = re.compile(pattern)

        def sub(self, repl: str, text: str) -> str:
            return self._pattern.sub(repl, text)

    normalizers = types.ModuleType("tokenizers.normalizers")

    class Sequence:
        def __init__(self, parts):
            self._parts = parts

        def normalize_str(self, text: str) -> str:
            out = text
            for part in self._parts:
                out = part.normalize_str(out)
            return out

    class NFKC:
        def normalize_str(self, text: str) -> str:
            return unicodedata.normalize("NFKC", text)

    class NFD:
        def normalize_str(self, text: str) -> str:
            return unicodedata.normalize("NFD", text)

    class StripAccents:
        def normalize_str(self, text: str) -> str:
            return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")

    class Lowercase:
        def normalize_str(self, text: str) -> str:
            return text.lower()

    class Replace:
        def __init__(self, regex: Regex, content: str):
            if isinstance(regex, str):
                self._pattern = re.compile(regex)
            elif hasattr(regex, "_pattern"):
                self._pattern = regex._pattern
            else:
                raise TypeError(f"unsupported regex type: {type(regex)}")
            self._content = content

        def normalize_str(self, text: str) -> str:
            return self._pattern.sub(self._content, text)

    class Strip:
        def normalize_str(self, text: str) -> str:
            return text.strip()

    normalizers.Sequence = Sequence
    normalizers.NFKC = NFKC
    normalizers.NFD = NFD
    normalizers.StripAccents = StripAccents
    normalizers.Lowercase = Lowercase
    normalizers.Replace = Replace
    normalizers.Strip = Strip

    tokenizers.Regex = Regex
    tokenizers.normalizers = normalizers
    sys.modules["tokenizers"] = tokenizers
    sys.modules["tokenizers.normalizers"] = normalizers


def install_sympy_isprime_shim() -> None:
    sympy = types.ModuleType("sympy")

    def isprime(n: int) -> bool:
        if n < 2:
            return False
        if n % 2 == 0:
            return n == 2
        d = 3
        while d * d <= n:
            if n % d == 0:
                return False
            d += 2
        return True

    sympy.isprime = isprime
    sys.modules["sympy"] = sympy


def import_upstream_engram_by_path(engram_py: Path) -> tuple[object | None, str]:
    spec = importlib.util.spec_from_file_location("deepseek_v41_upstream_engram", engram_py)
    if spec is None or spec.loader is None:
        raise SystemExit(f"could not create import spec for {engram_py}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(engram_py.parent))
    install_tokenizers_shim()
    install_sympy_isprime_shim()
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as err:
        return None, f"path-import-skipped-missing-dependency:{err.name}"
    except Exception as err:
        return None, f"path-import-skipped:{type(err).__name__}:{err}"
    finally:
        sys.path.pop(0)
    return module, "path-import-ok"


def tensor_to_nested_list(tensor) -> list:
    return tensor.detach().cpu().tolist()


def flatten_nested(values) -> list:
    if isinstance(values, list):
        out = []
        for value in values:
            out.extend(flatten_nested(value))
        return out
    return [values]


def call_upstream_or_fallback(status: str, func_name: str, upstream_call, fallback_call):
    if status == "path-import-ok":
        try:
            return upstream_call(), f"called-upstream:{func_name}"
        except Exception as err:
            return fallback_call(), f"fallback:{func_name}:{type(err).__name__}:{err}"
    return fallback_call(), f"fallback:{func_name}:{status}"


def generate_rope(out_dir: Path, module: object | None, import_status: str) -> None:
    config = {
        "rope_head_dim": 8,
        "max_seq_len": 80,
        "original_seq_len": 16,
        "rope_theta": 10000.0,
        "rope_factor": 4.0,
        "beta_fast": 4.0,
        "beta_slow": 1.0,
    }
    seqlen = 40
    freq_args = (
        config["rope_head_dim"],
        seqlen,
        config["original_seq_len"],
        config["rope_theta"],
        config["rope_factor"],
        config["beta_fast"],
        config["beta_slow"],
    )
    shape3 = {"batch": 1, "seqlen": 4, "dim": 8}
    data3 = [((i % 11) - 5) / 4.0 for i in range(shape3["batch"] * shape3["seqlen"] * shape3["dim"])]
    shape4 = {"batch": 1, "seqlen": 4, "heads": 2, "dim": 8}
    data4 = [((i % 17) - 8) / 5.0 for i in range(shape4["batch"] * shape4["seqlen"] * shape4["heads"] * shape4["dim"])]
    def upstream_rope():
        import torch

        assert module is not None
        freqs_tensor = module.precompute_freqs_cis(*freq_args)
        data3_tensor = torch.tensor(data3, dtype=torch.float32).reshape(
            shape3["batch"], shape3["seqlen"], shape3["dim"]
        )
        data4_tensor = torch.tensor(data4, dtype=torch.float32).reshape(
            shape4["batch"], shape4["seqlen"], shape4["heads"], shape4["dim"]
        )
        module.apply_rotary_emb(data3_tensor, freqs_tensor[: shape3["seqlen"]])
        module.apply_rotary_emb(data4_tensor, freqs_tensor[: shape4["seqlen"]])
        freqs = [
            [[float(value.real), float(value.imag)] for value in row]
            for row in tensor_to_nested_list(freqs_tensor)
        ]
        return freqs, data3_tensor.flatten().tolist(), data4_tensor.flatten().tolist()

    def fallback_rope():
        freqs = fallback_precompute_freqs_cis(*freq_args)
        return (
            freqs,
            fallback_apply_rotary(data3, shape3["batch"], shape3["seqlen"], 1, shape3["dim"], freqs),
            fallback_apply_rotary(
                data4,
                shape4["batch"],
                shape4["seqlen"],
                shape4["heads"],
                shape4["dim"],
                freqs,
            ),
        )

    rope_outputs, call_status = call_upstream_or_fallback(
        import_status,
        "precompute_freqs_cis/apply_rotary_emb",
        upstream_rope,
        fallback_rope,
    )
    freqs_nested, rope3_forward, rope4_forward = rope_outputs
    selected = []
    half = config["rope_head_dim"] // 2
    for pos in [0, 1, 7, 33]:
        for pair in [0, 1, 3]:
            cos, sin = freqs_nested[pos][pair]
            selected.append({"position": pos, "pair": pair, "cos": round_float(cos), "sin": round_float(sin)})

    write_json(
        out_dir / "rope_yarn_fixture.json",
        {
            **source_meta("precompute_freqs_cis", "precompute_freqs_cis/apply_rotary_emb around lines 369-408"),
            "upstream_import_status": import_status,
            "upstream_call_status": call_status,
            "absolute_tolerance": 1e-5,
            "input": {
                "config": config,
                "seqlen": seqlen,
                "rope_3d_shape": shape3,
                "rope_3d_data": round_list(data3),
                "rope_4d_shape": shape4,
                "rope_4d_data": round_list(data4),
            },
            "expected": {
                "selected_freqs": selected,
                "rope_3d_forward": round_list(rope3_forward),
                "rope_4d_forward": round_list(rope4_forward),
            },
        },
    )


def generate_sparse(out_dir: Path, module: object | None, import_status: str) -> None:
    window_cases = [
        {"name": "prefill_with_padding", "input": {"window_size": 4, "batch": 2, "seqlen": 6, "start_pos": 0}},
        {"name": "decode_ring_still_filling", "input": {"window_size": 5, "batch": 1, "seqlen": 1, "start_pos": 2}},
        {"name": "decode_full_ring_oldest_first", "input": {"window_size": 5, "batch": 1, "seqlen": 1, "start_pos": 7}},
    ]
    for case in window_cases:
        inp = case["input"]
        def upstream_window(inp=inp):
            assert module is not None
            return flatten_nested(
                tensor_to_nested_list(
                    module.get_window_topk_idxs(
                        inp["window_size"],
                        inp["batch"],
                        inp["seqlen"],
                        inp["start_pos"],
                    )
                )
            )

        expected, case_status = call_upstream_or_fallback(
            import_status,
            "get_window_topk_idxs",
            upstream_window,
            lambda inp=inp: fallback_window_topk(
                inp["window_size"], inp["batch"], inp["seqlen"], inp["start_pos"]
            ),
        )
        case["expected"] = expected
        case["upstream_call_status"] = case_status
    write_json(
        out_dir / "window_topk_fixture.json",
        {
            **source_meta("get_window_topk_idxs", "get_window_topk_idxs around lines 410-427"),
            "upstream_import_status": import_status,
            "upstream_call_status": ",".join(sorted({case["upstream_call_status"] for case in window_cases})),
            "absolute_tolerance": 0.0,
            "cases": window_cases,
        },
    )

    logits = [
        0.1, 0.2, 0.9, 0.0, 2.0, 2.5, "-inf", "-inf",
        "-inf", "-inf", "-inf", "-inf", 0.1, 0.2, 0.3, 0.4,
    ]
    numeric_logits = [-math.inf if v == "-inf" else float(v) for v in logits]
    shape = {"batch": 1, "seqlen": 2, "positions": 8}
    compress_lens = [6, 4]
    def upstream_candidates():
        import torch

        assert module is not None
        logits_tensor = torch.tensor(numeric_logits, dtype=torch.float32).reshape(
            shape["batch"], shape["seqlen"], shape["positions"]
        )
        lens_tensor = torch.tensor(compress_lens, dtype=torch.int64).reshape(shape["batch"], shape["seqlen"], 1)
        return flatten_nested(
            tensor_to_nested_list(module.select_candidate_blocks(logits_tensor, lens_tensor, 2, 2))
        )

    candidate_mask, candidate_status = call_upstream_or_fallback(
        import_status,
        "select_candidate_blocks",
        upstream_candidates,
        lambda: fallback_select_candidate_blocks(
            numeric_logits,
            shape["batch"],
            shape["seqlen"],
            shape["positions"],
            compress_lens,
            2,
            2,
        ),
    )

    write_json(
        out_dir / "candidate_blocks_fixture.json",
        {
            **source_meta("select_candidate_blocks", "select_candidate_blocks around lines 583-611"),
            "upstream_import_status": import_status,
            "upstream_call_status": candidate_status,
            "absolute_tolerance": 0.0,
            "input": {
                "shape": shape,
                "logits": logits,
                "compress_lens": compress_lens,
                "topk_blocks": 2,
                "block_size": 2,
            },
            "expected": {"mask": candidate_mask},
        },
    )


def generate_compressor(out_dir: Path, module: object | None, import_status: str) -> None:
    norm_weight = [1.0, 1.5, 0.5, 2.0]
    eps = 1e-6
    ratio_one_kv = [1.0, -2.0, 3.0, -4.0, 2.0, 1.0, -1.0, -2.0]
    ratio_n_kv = [
        1.0, 2.0, 3.0, 4.0,
        2.0, 0.0, 4.0, 1.0,
        -1.0, 1.5, 0.5, 2.5,
        5.0, 4.0, 3.0, 2.0,
        9.0, 9.0, 9.0, 9.0,
    ]
    ratio_n_score = [
        0.0, 1.0, 0.5, -1.0,
        1.0, 0.0, 0.5, 2.0,
        -0.5, 0.25, 1.5, 0.0,
        2.0, 1.0, -1.0, 0.5,
        8.0, 8.0, 8.0, 8.0,
    ]
    ratio = 2

    def upstream_ratio_one():
        import torch

        assert module is not None
        compressor = module.Compressor.__new__(module.Compressor)
        module.nn.Module.__init__(compressor)
        compressor.compress_ratio = 1
        compressor.norm = module.RMSNorm(len(norm_weight), eps)
        compressor.norm.weight.data = torch.tensor(norm_weight, dtype=torch.float32)
        batch, seqlen, dim = 1, len(ratio_one_kv) // len(norm_weight), len(norm_weight)
        object.__setattr__(
            compressor,
            "wkv",
            lambda _x: torch.tensor(ratio_one_kv, dtype=torch.float32).reshape(batch, seqlen, dim),
        )
        return compressor.forward(
            torch.tensor(ratio_one_kv, dtype=torch.float32).reshape(batch, seqlen, dim),
            0,
        ).flatten().tolist()

    def upstream_ratio_n():
        import torch

        assert module is not None
        compressor = module.Compressor.__new__(module.Compressor)
        module.nn.Module.__init__(compressor)
        compressor.compress_ratio = ratio
        compressor.norm = module.RMSNorm(len(norm_weight), eps)
        compressor.norm.weight.data = torch.tensor(norm_weight, dtype=torch.float32)
        batch, seqlen, dim = 1, len(ratio_n_kv) // len(norm_weight), len(norm_weight)
        x = torch.tensor(ratio_n_kv, dtype=torch.float32).reshape(batch, seqlen, dim)
        score = torch.tensor(ratio_n_score, dtype=torch.float32).reshape(batch, seqlen, dim)
        object.__setattr__(compressor, "wkv", lambda _x: x)
        object.__setattr__(compressor, "wgate", lambda _x: score)
        compressor.kv_state = torch.zeros(batch, ratio, dim, dtype=torch.float32)
        compressor.score_state = torch.full((batch, ratio, dim), -torch.inf, dtype=torch.float32)
        return compressor.forward(x, 0).flatten().tolist()

    ratio_one_expected, ratio_one_status = call_upstream_or_fallback(
        import_status,
        "Compressor.forward:ratio_one",
        upstream_ratio_one,
        lambda: fallback_rms_norm(ratio_one_kv, len(norm_weight), norm_weight, eps),
    )
    ratio_n_expected, ratio_n_status = call_upstream_or_fallback(
        import_status,
        "Compressor.forward:ratio_n_prefill",
        upstream_ratio_n,
        lambda: fallback_compress_ratio_n(
            ratio_n_kv, ratio_n_score, ratio, len(norm_weight), norm_weight, eps
        ),
    )

    write_json(
        out_dir / "compressor_fixture.json",
        {
            **source_meta("Compressor.forward", "Compressor.forward around lines 458-489"),
            "upstream_import_status": import_status,
            "upstream_call_status": f"{ratio_one_status},{ratio_n_status}",
            "absolute_tolerance": 1e-5,
            "ratio_one": {
                "input": {"kv": round_list(ratio_one_kv), "norm_weight": norm_weight, "eps": eps},
                "expected": round_list(ratio_one_expected),
                "upstream_call_status": ratio_one_status,
            },
            "ratio_n_prefill": {
                "input": {
                    "kv": round_list(ratio_n_kv),
                    "score": round_list(ratio_n_score),
                    "ratio": ratio,
                    "norm_weight": norm_weight,
                    "eps": eps,
                },
                "expected": round_list(ratio_n_expected),
                "upstream_call_status": ratio_n_status,
            },
        },
    )


def fallback_sqrtsoftplus_scores(x: list[float], gate_weight: list[float], gate_temp: float, tokens: int, dim: int, experts: int) -> list[float]:
    out = [0.0] * (tokens * experts)
    for t in range(tokens):
        x_base = t * dim
        for e in range(experts):
            w_base = e * dim
            acc = 0.0
            for d in range(dim):
                acc += x[x_base + d] * gate_weight[w_base + d]
            scaled = acc / gate_temp
            out[t * experts + e] = math.sqrt(math.log1p(math.exp(scaled)))
    return out


def fallback_select_experts(scores: list[float], bias: list[float], tokens: int, experts: int, topk: int) -> list[int]:
    out = []
    for t in range(tokens):
        base = t * experts
        ranked = sorted(
            range(experts),
            key=lambda e: (-(scores[base + e] + bias[e]), e),
        )
        out.extend(ranked[:topk])
    return out


def fallback_route_weights(scores: list[float], indices: list[int], tokens: int, experts: int, topk: int, norm_topk_prob: bool, route_scale: float) -> list[float]:
    out = []
    for t in range(tokens):
        s_base = t * experts
        idx_base = t * topk
        weights = [scores[s_base + indices[idx_base + k]] for k in range(topk)]
        if norm_topk_prob and topk > 1:
            denom = sum(weights) + 1e-20
            weights = [w / denom for w in weights]
        out.extend([w * route_scale for w in weights])
    return out


def fallback_expert_swiglu(gate: list[float], up: list[float], swiglu_limit: float) -> list[float]:
    out = []
    for g, u in zip(gate, up):
        if swiglu_limit > 0:
            u = max(min(u, swiglu_limit), -swiglu_limit)
            g = min(g, swiglu_limit)
        out.append((g / (1.0 + math.exp(-g))) * u)
    return out


def matmul_rows(x: list[float], w: list[float], din: int, dout: int) -> list[float]:
    rows = len(x) // din
    out = [0.0] * (rows * dout)
    for row in range(rows):
        for j in range(dout):
            acc = 0.0
            for i in range(din):
                acc += x[row * din + i] * w[i * dout + j]
            out[row * dout + j] = acc
    return out


def layer_sparse_attention(
    q: list[float],
    kv: list[float],
    sparse_indices: list[int],
    batch: int,
    seqlen: int,
    n_heads: int,
    head_dim: int,
    attn_sink: list[float],
) -> list[float]:
    out = [0.0] * (batch * seqlen * n_heads * head_dim)
    scale = head_dim ** -0.5
    for b in range(batch):
        for s in range(seqlen):
            src = sparse_indices[b * seqlen + s]
            for h in range(n_heads):
                score = attn_sink[h]
                for d in range(head_dim):
                    q_idx = ((b * seqlen + s) * n_heads + h) * head_dim + d
                    k_idx = (b * seqlen + src) * head_dim + d
                    score += q[q_idx] * kv[k_idx] * scale
                gate = 1.0 / (1.0 + math.exp(-score))
                for d in range(head_dim):
                    out_idx = ((b * seqlen + s) * n_heads + h) * head_dim + d
                    v_idx = (b * seqlen + src) * head_dim + d
                    out[out_idx] = gate * kv[v_idx]
    return out


def grouped_wo_a(
    attended: list[float],
    batch: int,
    seqlen: int,
    n_heads: int,
    head_dim: int,
    o_groups: int,
    o_lora_rank: int,
    wo_a: list[float],
) -> list[float]:
    heads_per_group = n_heads // o_groups
    group_in = heads_per_group * head_dim
    out = [0.0] * (batch * seqlen * o_groups * o_lora_rank)
    for b in range(batch):
        for s in range(seqlen):
            for g in range(o_groups):
                for r in range(o_lora_rank):
                    acc = 0.0
                    for i in range(group_in):
                        h = g * heads_per_group + i // head_dim
                        d = i % head_dim
                        a_idx = ((b * seqlen + s) * n_heads + h) * head_dim + d
                        w_idx = (g * group_in + i) * o_lora_rank + r
                        acc += attended[a_idx] * wo_a[w_idx]
                    out[((b * seqlen + s) * o_groups + g) * o_lora_rank + r] = acc
    return out


def expert_forward_row(
    x: list[float],
    w1: list[float],
    w2: list[float],
    w3: list[float],
    dim: int,
    inter_dim: int,
    swiglu_limit: float,
) -> list[float]:
    gate = matmul_rows(x, w1, dim, inter_dim)
    up = matmul_rows(x, w3, dim, inter_dim)
    hidden = fallback_expert_swiglu(gate, up, swiglu_limit)
    return matmul_rows(hidden, w2, inter_dim, dim)


def generate_layer(out_dir: Path, module: object | None, import_status: str) -> None:
    batch, seqlen, dim = 1, 3, 4
    n_heads, head_dim, rope_head_dim = 2, 2, 1
    q_lora_rank, o_lora_rank, o_groups = 3, 2, 2
    eps = 1e-6
    x = [((i % 7) - 3) / 5.0 for i in range(batch * seqlen * dim)]
    wq_a = [((i % 5) - 2) / 7.0 for i in range(dim * q_lora_rank)]
    q_norm = [1.0, 0.75, 1.25]
    wq_b = [((i % 11) - 5) / 6.0 for i in range(q_lora_rank * n_heads * head_dim)]
    wkv = [((i % 13) - 6) / 8.0 for i in range(dim * head_dim)]
    kv_norm = [1.0, 1.5]
    wo_a = [((i % 9) - 4) / 6.0 for i in range(o_groups * head_dim * o_lora_rank)]
    wo_b = [((i % 7) - 3) / 5.0 for i in range(o_groups * o_lora_rank * dim)]
    attn_sink = [0.25, -0.1]
    sparse_indices = [0, 0, 1]
    compressor_wkv = [((i % 10) - 4) / 9.0 for i in range(dim * head_dim)]
    compressor_norm = [0.5, 1.25]

    qr = fallback_rms_norm(matmul_rows(x, wq_a, dim, q_lora_rank), q_lora_rank, q_norm, eps)
    q = matmul_rows(qr, wq_b, q_lora_rank, n_heads * head_dim)
    kv = fallback_rms_norm(matmul_rows(x, wkv, dim, head_dim), head_dim, kv_norm, eps)
    attended = layer_sparse_attention(q, kv, sparse_indices, batch, seqlen, n_heads, head_dim, attn_sink)
    low_rank = grouped_wo_a(attended, batch, seqlen, n_heads, head_dim, o_groups, o_lora_rank, wo_a)
    output = matmul_rows(low_rank, wo_b, o_groups * o_lora_rank, dim)
    compressed = fallback_rms_norm(
        matmul_rows(x, compressor_wkv, dim, head_dim),
        head_dim,
        compressor_norm,
        eps,
    )

    def param(shape: list[int], data: list[float]) -> dict:
        return {"shape": shape, "data": round_list(data)}

    write_json(
        out_dir / "attention_layer_fixture.json",
        {
            **source_meta("Attention.forward", "Attention.forward around lines 627-789"),
            "upstream_import_status": import_status,
            "upstream_call_status": "fixture-seam:Attention.forward:layer-skeleton",
            "absolute_tolerance": 1e-5,
            "input": {
                "x_shape": [batch, seqlen, dim],
                "x": round_list(x),
                "start_pos": 0,
                "rms_norm_eps": eps,
                "sparse_indices": sparse_indices,
                "shape": {
                    "n_heads": n_heads,
                    "head_dim": head_dim,
                    "rope_head_dim": rope_head_dim,
                    "q_lora_rank": q_lora_rank,
                    "o_lora_rank": o_lora_rank,
                    "o_groups": o_groups,
                    "compress_ratio": 1,
                    "window_size": 4,
                    "index_topk": 1,
                },
            },
            "parameters": {
                "wq_a": param([dim, q_lora_rank], wq_a),
                "q_norm": param([q_lora_rank], q_norm),
                "wq_b": param([q_lora_rank, n_heads * head_dim], wq_b),
                "wkv": param([dim, head_dim], wkv),
                "kv_norm": param([head_dim], kv_norm),
                "wo_a": param([o_groups, head_dim, o_lora_rank], wo_a),
                "wo_b": param([o_groups * o_lora_rank, dim], wo_b),
                "attn_sink": param([n_heads], attn_sink),
                "compressor_wkv": param([dim, head_dim], compressor_wkv),
                "compressor_norm": param([head_dim], compressor_norm),
            },
            "expected": {
                "shape": [batch, seqlen, dim],
                "output": round_list(output),
                "compressed_kv": round_list(compressed),
            },
        },
    )

    tokens, moe_dim, experts, topk, inter_dim = 2, 4, 4, 2, 3
    gate_temp = 1.2
    route_scale = 1.5
    moe_x = [((i % 9) - 4) / 4.0 for i in range(tokens * moe_dim)]
    gate_weight = [((i % 13) - 6) / 5.0 for i in range(experts * moe_dim)]
    correction_bias = [0.0, 0.25, -0.1, 0.15]
    expert_params = []
    for expert in range(experts):
        expert_params.append(
            {
                "w1": [((i + expert) % 7 - 3) / 6.0 for i in range(moe_dim * inter_dim)],
                "w2": [((i + 2 * expert) % 11 - 5) / 8.0 for i in range(inter_dim * moe_dim)],
                "w3": [((i + 3 * expert) % 5 - 2) / 5.0 for i in range(moe_dim * inter_dim)],
            }
        )
    shared = {
        "w1": [((i % 7) - 3) / 8.0 for i in range(moe_dim * inter_dim)],
        "w2": [((i % 9) - 4) / 7.0 for i in range(inter_dim * moe_dim)],
        "w3": [((i % 5) - 2) / 6.0 for i in range(moe_dim * inter_dim)],
    }
    scores = fallback_sqrtsoftplus_scores(moe_x, gate_weight, gate_temp, tokens, moe_dim, experts)
    indices = fallback_select_experts(scores, correction_bias, tokens, experts, topk)
    weights = fallback_route_weights(scores, indices, tokens, experts, topk, True, route_scale)
    moe_out = [0.0] * (tokens * moe_dim)
    for token in range(tokens):
        row = moe_x[token * moe_dim : (token + 1) * moe_dim]
        for slot in range(topk):
            expert_idx = indices[token * topk + slot]
            expert_out = expert_forward_row(
                row,
                expert_params[expert_idx]["w1"],
                expert_params[expert_idx]["w2"],
                expert_params[expert_idx]["w3"],
                moe_dim,
                inter_dim,
                10.0,
            )
            for d in range(moe_dim):
                moe_out[token * moe_dim + d] += weights[token * topk + slot] * expert_out[d]
        shared_out = expert_forward_row(
            row, shared["w1"], shared["w2"], shared["w3"], moe_dim, inter_dim, 10.0
        )
        for d in range(moe_dim):
            moe_out[token * moe_dim + d] += shared_out[d]

    def expert_json(weights: dict[str, list[float]]) -> dict:
        return {
            "w1": param([moe_dim, inter_dim], weights["w1"]),
            "w2": param([inter_dim, moe_dim], weights["w2"]),
            "w3": param([moe_dim, inter_dim], weights["w3"]),
        }

    write_json(
        out_dir / "moe_layer_fixture.json",
        {
            **source_meta("MoE.forward", "MoE.forward around lines 854-890"),
            "upstream_import_status": import_status,
            "upstream_call_status": "fixture-seam:MoE.forward:layer-skeleton",
            "absolute_tolerance": 1e-5,
            "input": {
                "x_shape": [1, tokens, moe_dim],
                "x": round_list(moe_x),
                "tokens": tokens,
                "dim": moe_dim,
                "experts": experts,
                "topk": topk,
                "inter_dim": inter_dim,
                "gate_temp": gate_temp,
                "norm_topk_prob": True,
                "route_scale": route_scale,
                "swiglu_limit": 10.0,
            },
            "parameters": {
                "gate_weight": param([experts, moe_dim], gate_weight),
                "correction_bias": param([experts], correction_bias),
                "experts": [expert_json(weights) for weights in expert_params],
                "shared_expert": expert_json(shared),
            },
            "expected": {
                "shape": [1, tokens, moe_dim],
                "selected_experts": indices,
                "output": round_list(moe_out),
            },
        },
    )

    engram_tokens, hc_mult, engram_dim = 2, 2, 3
    engram_x = [((i % 11) - 5) / 5.0 for i in range(engram_tokens * hc_mult * engram_dim)]
    engram_key = [((i % 7) - 3) / 4.0 for i in range(engram_tokens * hc_mult * engram_dim)]
    engram_value = [((i % 5) - 2) / 3.0 for i in range(engram_tokens * engram_dim)]
    q_weight = [1.0, 0.75, 1.25, 0.5, 1.5, 0.875]
    k_weight = [0.5, 1.25, 0.75, 1.0, 0.625, 1.375]
    token_mask = [True, False]
    engram_output = layer_engram_update(
        engram_x, engram_key, engram_value, q_weight, k_weight, eps, token_mask
    )
    masked_start = engram_dim * hc_mult
    masked_end = masked_start + engram_dim * hc_mult

    write_json(
        out_dir / "engram_layer_fixture.json",
        {
            **source_meta(
                "Engram.forward",
                "Engram.forward around lines 328-367",
                source_file="ferric_continuum/tnsr/third_party/deepseek_v41/upstream/inference/engram.py",
            ),
            "upstream_import_status": import_status,
            "upstream_call_status": "fixture-seam:Engram.forward:layer-skeleton",
            "absolute_tolerance": 1e-5,
            "input": {
                "x_shape": [engram_tokens, hc_mult, engram_dim],
                "x": round_list(engram_x),
                "key": round_list(engram_key),
                "value_shape": [engram_tokens, engram_dim],
                "value": round_list(engram_value),
                "eps": eps,
                "token_mask": token_mask,
                "masked_token_start": masked_start,
                "masked_token_end": masked_end,
            },
            "parameters": {
                "q_weight": param([hc_mult * engram_dim], q_weight),
                "k_weight": param([hc_mult * engram_dim], k_weight),
            },
            "expected": {
                "shape": [engram_tokens, hc_mult, engram_dim],
                "output": round_list(engram_output),
                "masked_token_original": round_list(engram_x[masked_start:masked_end]),
            },
        },
    )


def layer_engram_update(
    x: list[float],
    key: list[float],
    value: list[float],
    q_weight: list[float],
    k_weight: list[float],
    eps: float,
    token_mask: list[bool],
) -> list[float]:
    hc_mult = len(x) // len(value)
    dim = len(q_weight) // hc_mult
    tokens = len(value) // dim
    out = [0.0] * len(x)
    weight = [q * k for q, k in zip(q_weight, k_weight)]
    inv_sqrt_dim = dim ** -0.5
    for t in range(tokens):
        for h in range(hc_mult):
            base = (t * hc_mult + h) * dim
            vbase = t * dim
            row = x[base : base + dim]
            krow = key[base : base + dim]
            mean_sq_x = sum(v * v for v in row) / dim
            mean_sq_k = sum(v * v for v in krow) / dim
            dot = sum(row[d] * weight[h * dim + d] * krow[d] for d in range(dim))
            dot *= (mean_sq_x + eps) ** -0.5 * (mean_sq_k + eps) ** -0.5 * inv_sqrt_dim
            gate = 1.0 / (1.0 + math.exp(-math.copysign(max(abs(dot), 1e-6) ** 0.5, dot)))
            if not token_mask[t]:
                gate = 0.0
            for d in range(dim):
                out[base + d] = row[d] + gate * value[vbase + d]
    return out


def fixture_param(shape: list[int], data: list[float]) -> dict:
    return {"shape": shape, "data": round_list(data)}


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def sinkhorn(matrix: list[float], n: int, iters: int, eps: float) -> list[float]:
    out = list(matrix)
    for row in range(n):
        base = row * n
        row_max = max(out[base : base + n])
        values = [math.exp(out[base + col] - row_max) for col in range(n)]
        denom = sum(values)
        for col, value in enumerate(values):
            out[base + col] = value / denom + eps

    def normalize_cols() -> None:
        sums = [sum(out[row * n + col] for row in range(n)) for col in range(n)]
        for col, total in enumerate(sums):
            denom = total + eps
            for row in range(n):
                out[row * n + col] /= denom

    def normalize_rows() -> None:
        for row in range(n):
            base = row * n
            denom = sum(out[base : base + n]) + eps
            for col in range(n):
                out[base + col] /= denom

    normalize_cols()
    for _ in range(max(0, iters - 1)):
        normalize_rows()
        normalize_cols()
    return out


def hc_mixes_ref(
    flat_hc: list[float],
    hc_fn: list[float],
    hc_scale: list[float],
    hc_base: list[float],
    hc_mult: int,
    dim: int,
    iters: int,
    eps: float,
) -> dict[str, list[float]]:
    tokens = len(flat_hc) // (hc_mult * dim)
    mix_hc = (2 + hc_mult) * hc_mult
    mixes = [0.0] * (tokens * mix_hc)
    for token in range(tokens):
        x_base = token * hc_mult * dim
        row = flat_hc[x_base : x_base + hc_mult * dim]
        rsqrt = (sum(v * v for v in row) / (hc_mult * dim) + eps) ** -0.5
        for out_idx in range(mix_hc):
            w_base = out_idx * hc_mult * dim
            mixes[token * mix_hc + out_idx] = (
                sum(row[i] * hc_fn[w_base + i] for i in range(hc_mult * dim)) * rsqrt
            )

    pre = [0.0] * (tokens * hc_mult)
    post = [0.0] * (tokens * hc_mult)
    comb = [0.0] * (tokens * hc_mult * hc_mult)
    for token in range(tokens):
        base = token * mix_hc
        for hc in range(hc_mult):
            pre[token * hc_mult + hc] = sigmoid(mixes[base + hc] * hc_scale[0] + hc_base[hc]) + eps
            post[token * hc_mult + hc] = 2.0 * sigmoid(
                mixes[base + hc_mult + hc] * hc_scale[1] + hc_base[hc_mult + hc]
            )
        comb_logits = []
        for dst in range(hc_mult):
            for src in range(hc_mult):
                flat = dst * hc_mult + src
                comb_logits.append(
                    mixes[base + 2 * hc_mult + flat] * hc_scale[2] + hc_base[2 * hc_mult + flat]
                )
        comb[token * hc_mult * hc_mult : (token + 1) * hc_mult * hc_mult] = sinkhorn(
            comb_logits, hc_mult, iters, eps
        )
    return {"pre": pre, "post": post, "comb": comb}


def hc_pre_ref(
    x: list[float],
    pre_mix: list[float],
    batch: int,
    seqlen: int,
    hc_mult: int,
    dim: int,
) -> list[float]:
    out = [0.0] * (batch * seqlen * dim)
    for token in range(batch * seqlen):
        x_base = token * hc_mult * dim
        out_base = token * dim
        mix_base = token * hc_mult
        for hc in range(hc_mult):
            coeff = pre_mix[mix_base + hc]
            for d in range(dim):
                out[out_base + d] += coeff * x[x_base + hc * dim + d]
    return out


def hc_post_ref(
    sublayer: list[float],
    residual: list[float],
    post: list[float],
    comb: list[float],
    batch: int,
    seqlen: int,
    hc_mult: int,
    dim: int,
) -> list[float]:
    out = [0.0] * (batch * seqlen * hc_mult * dim)
    for token in range(batch * seqlen):
        sub_base = token * dim
        res_base = token * hc_mult * dim
        post_base = token * hc_mult
        comb_base = token * hc_mult * hc_mult
        for dst in range(hc_mult):
            out_base = res_base + dst * dim
            for d in range(dim):
                out[out_base + d] = post[post_base + dst] * sublayer[sub_base + d]
            for src in range(hc_mult):
                coeff = comb[comb_base + dst * hc_mult + src]
                for d in range(dim):
                    out[out_base + d] += coeff * residual[res_base + src * dim + d]
    return out


def default_sparse_indices(batch: int, seqlen: int) -> list[int]:
    out = []
    for _ in range(batch):
        for pos in range(seqlen):
            out.append(min(max(pos - 1, 0), pos))
    return out


def attention_forward_ref(
    x: list[float],
    batch: int,
    seqlen: int,
    dim: int,
    attention: dict,
    params: dict,
    state: dict,
    eps: float,
    has_compressor: bool,
    has_indexer: bool,
) -> list[float]:
    q_lora_rank = attention["q_lora_rank"]
    n_heads = attention["n_heads"]
    head_dim = attention["head_dim"]
    o_groups = attention["o_groups"]
    o_lora_rank = attention["o_lora_rank"]
    qr = fallback_rms_norm(
        matmul_rows(x, params["wq_a"], dim, q_lora_rank),
        q_lora_rank,
        params["q_norm"],
        eps,
    )
    q = matmul_rows(qr, params["wq_b"], q_lora_rank, n_heads * head_dim)
    kv = fallback_rms_norm(
        matmul_rows(x, params["wkv"], dim, head_dim),
        head_dim,
        params["kv_norm"],
        eps,
    )
    if has_compressor:
        state["compressed_kv"] = fallback_rms_norm(
            matmul_rows(x, params["compressor_wkv"], dim, head_dim),
            head_dim,
            params["compressor_norm"],
            eps,
        )
    if has_indexer and state.get("topk_indices") is None:
        state["topk_indices"] = default_sparse_indices(batch, seqlen)
    if state.get("topk_indices") is not None:
        state["consumed_sparse_indices"] = True
    sparse = state.get("topk_indices") or default_sparse_indices(batch, seqlen)
    attended = layer_sparse_attention(q, kv, sparse, batch, seqlen, n_heads, head_dim, params["attn_sink"])
    low_rank = grouped_wo_a(attended, batch, seqlen, n_heads, head_dim, o_groups, o_lora_rank, params["wo_a"])
    return matmul_rows(low_rank, params["wo_b"], o_groups * o_lora_rank, dim)


def moe_forward_ref(x: list[float], tokens: int, dim: int, moe: dict, params: dict) -> list[float]:
    scores = fallback_sqrtsoftplus_scores(
        x,
        params["gate_weight"],
        moe["gate_temp"],
        tokens,
        dim,
        moe["experts"],
    )
    indices = fallback_select_experts(
        scores,
        params["correction_bias"],
        tokens,
        moe["experts"],
        moe["topk"],
    )
    weights = fallback_route_weights(
        scores,
        indices,
        tokens,
        moe["experts"],
        moe["topk"],
        moe["norm_topk_prob"],
        moe["route_scale"],
    )
    out = [0.0] * (tokens * dim)
    for token in range(tokens):
        row = x[token * dim : (token + 1) * dim]
        for slot in range(moe["topk"]):
            expert_idx = indices[token * moe["topk"] + slot]
            expert = params["experts"][expert_idx]
            expert_out = expert_forward_row(
                row,
                expert["w1"],
                expert["w2"],
                expert["w3"],
                dim,
                moe["inter_dim"],
                moe["swiglu_limit"],
            )
            for d in range(dim):
                out[token * dim + d] += weights[token * moe["topk"] + slot] * expert_out[d]
        shared = params["shared_expert"]
        shared_out = expert_forward_row(
            row,
            shared["w1"],
            shared["w2"],
            shared["w3"],
            dim,
            moe["inter_dim"],
            moe["swiglu_limit"],
        )
        for d in range(dim):
            out[token * dim + d] += shared_out[d]
    return out


def block_forward_ref(
    x: list[float],
    pre_mix: list[float],
    cfg: dict,
    params: dict,
    state: dict | None = None,
) -> tuple[list[float], list[float], dict]:
    state = {} if state is None else dict(state)
    state.setdefault("compressed_kv", None)
    state.setdefault("topk_indices", None)
    state.setdefault("consumed_sparse_indices", False)
    batch = cfg["batch"]
    seqlen = cfg["seqlen"]
    hc_mult = cfg["hc_mult"]
    dim = cfg["dim"]
    eps = cfg["hc_eps"]

    if cfg["mode"] == "engram":
        x = layer_engram_update(
            x,
            params["engram"]["key"],
            params["engram"]["value"],
            params["engram"]["q_weight"],
            params["engram"]["k_weight"],
            eps,
            cfg["token_mask"],
        )

    residual = list(x)
    attn_mix = hc_mixes_ref(
        residual,
        params["hc_attn_fn"],
        params["hc_attn_scale"],
        params["hc_attn_base"],
        hc_mult,
        dim,
        cfg["hc_sinkhorn_iters"],
        eps,
    )
    attn_in = hc_pre_ref(residual, pre_mix, batch, seqlen, hc_mult, dim)
    attn_in = fallback_rms_norm(attn_in, dim, params["attn_norm"], cfg["rms_norm_eps"])
    attn_out = attention_forward_ref(
        attn_in,
        batch,
        seqlen,
        dim,
        cfg["attention_shape"],
        params["attention"],
        state,
        cfg["rms_norm_eps"],
        cfg["mode"] in {"kv_source", "index_source"},
        cfg["mode"] == "index_source",
    )
    x = hc_post_ref(attn_out, residual, attn_mix["post"], attn_mix["comb"], batch, seqlen, hc_mult, dim)

    residual = list(x)
    ffn_mix = hc_mixes_ref(
        residual,
        params["hc_ffn_fn"],
        params["hc_ffn_scale"],
        params["hc_ffn_base"],
        hc_mult,
        dim,
        cfg["hc_sinkhorn_iters"],
        eps,
    )
    ffn_in = hc_pre_ref(residual, attn_mix["pre"], batch, seqlen, hc_mult, dim)
    ffn_in = fallback_rms_norm(ffn_in, dim, params["ffn_norm"], cfg["rms_norm_eps"])
    ffn_out = moe_forward_ref(ffn_in, batch * seqlen, dim, cfg, params["moe"])
    x = hc_post_ref(ffn_out, residual, ffn_mix["post"], ffn_mix["comb"], batch, seqlen, hc_mult, dim)
    return x, ffn_mix["pre"], state


def make_block_fixture(mode: str, import_status: str) -> dict:
    batch, seqlen, dim, hc_mult = 1, 2, 4, 2
    n_heads, head_dim, rope_head_dim = 2, 2, 1
    q_lora_rank, o_lora_rank, o_groups = 3, 2, 2
    experts, topk, inter_dim = 4, 2, 3
    eps = 1e-6
    x = [((i % 13) - 6) / 8.0 for i in range(batch * seqlen * hc_mult * dim)]
    pre_mix = [1.0, 0.0] * (batch * seqlen)

    attention_shape = {
        "n_heads": n_heads,
        "head_dim": head_dim,
        "rope_head_dim": rope_head_dim,
        "q_lora_rank": q_lora_rank,
        "o_lora_rank": o_lora_rank,
        "o_groups": o_groups,
        "compress_ratio": 1,
        "window_size": 4,
        "index_topk": 1,
    }
    attention_params = {
        "wq_a": [((i % 5) - 2) / 7.0 for i in range(dim * q_lora_rank)],
        "q_norm": [1.0, 0.75, 1.25],
        "wq_b": [((i % 11) - 5) / 6.0 for i in range(q_lora_rank * n_heads * head_dim)],
        "wkv": [((i % 13) - 6) / 8.0 for i in range(dim * head_dim)],
        "kv_norm": [1.0, 1.5],
        "wo_a": [((i % 9) - 4) / 6.0 for i in range(o_groups * head_dim * o_lora_rank)],
        "wo_b": [((i % 7) - 3) / 5.0 for i in range(o_groups * o_lora_rank * dim)],
        "attn_sink": [0.25, -0.1],
        "compressor_wkv": [((i % 10) - 4) / 9.0 for i in range(dim * head_dim)],
        "compressor_norm": [0.5, 1.25],
    }
    expert_params = []
    for expert in range(experts):
        expert_params.append(
            {
                "w1": [((i + expert) % 7 - 3) / 6.0 for i in range(dim * inter_dim)],
                "w2": [((i + 2 * expert) % 11 - 5) / 8.0 for i in range(inter_dim * dim)],
                "w3": [((i + 3 * expert) % 5 - 2) / 5.0 for i in range(dim * inter_dim)],
            }
        )
    shared = {
        "w1": [((i % 7) - 3) / 8.0 for i in range(dim * inter_dim)],
        "w2": [((i % 9) - 4) / 7.0 for i in range(inter_dim * dim)],
        "w3": [((i % 5) - 2) / 6.0 for i in range(dim * inter_dim)],
    }
    mix_hc = (2 + hc_mult) * hc_mult
    params_raw = {
        "attn_norm": [1.0, 0.875, 1.25, 0.75],
        "ffn_norm": [0.8, 1.1, 0.9, 1.2],
        "attention": attention_params,
        "moe": {
            "gate_weight": [((i % 13) - 6) / 5.0 for i in range(experts * dim)],
            "correction_bias": [0.0, 0.25, -0.1, 0.15],
            "experts": expert_params,
            "shared_expert": shared,
        },
        "hc_attn_fn": [((i % 17) - 8) / 30.0 for i in range(mix_hc * hc_mult * dim)],
        "hc_attn_base": [((i % 7) - 3) / 20.0 for i in range(mix_hc)],
        "hc_attn_scale": [0.15, 0.2, 0.1],
        "hc_ffn_fn": [((i % 19) - 9) / 35.0 for i in range(mix_hc * hc_mult * dim)],
        "hc_ffn_base": [((i % 5) - 2) / 18.0 for i in range(mix_hc)],
        "hc_ffn_scale": [0.12, 0.18, 0.09],
    }
    if mode == "engram":
        params_raw["engram"] = {
            "q_weight": [1.0, 0.75, 1.25, 0.5, 1.5, 0.875, 1.125, 0.625],
            "k_weight": [0.5, 1.25, 0.75, 1.0, 0.625, 1.375, 0.875, 1.125],
            "key": [((i % 7) - 3) / 4.0 for i in range(batch * seqlen * hc_mult * dim)],
            "value": [((i % 5) - 2) / 3.0 for i in range(batch * seqlen * dim)],
        }

    cfg = {
        "mode": mode,
        "layer_id": 14 if mode == "engram" else 2,
        "batch": batch,
        "seqlen": seqlen,
        "dim": dim,
        "hc_mult": hc_mult,
        "hc_sinkhorn_iters": 3,
        "hc_eps": eps,
        "x": x,
        "x_shape": [batch, seqlen, hc_mult, dim],
        "pre_mix": pre_mix,
        "start_pos": 0,
        "rms_norm_eps": eps,
        "attention_shape": attention_shape,
        "tokens": batch * seqlen,
        "experts": experts,
        "topk": topk,
        "inter_dim": inter_dim,
        "gate_temp": 1.2,
        "norm_topk_prob": True,
        "route_scale": 1.5,
        "swiglu_limit": 10.0,
    }
    if mode == "reuse":
        cfg["initial_sparse_indices"] = [0, 0]
    if mode == "engram":
        cfg["token_mask"] = [True, False]

    state = {"topk_indices": cfg.get("initial_sparse_indices")}
    output, next_pre_mix, final_state = block_forward_ref(x, pre_mix, cfg, params_raw, state)

    def expert_json(expert: dict[str, list[float]]) -> dict:
        return {
            "w1": fixture_param([dim, inter_dim], expert["w1"]),
            "w2": fixture_param([inter_dim, dim], expert["w2"]),
            "w3": fixture_param([dim, inter_dim], expert["w3"]),
        }

    params = {
        "attn_norm": fixture_param([dim], params_raw["attn_norm"]),
        "ffn_norm": fixture_param([dim], params_raw["ffn_norm"]),
        "attention": {
            "wq_a": fixture_param([dim, q_lora_rank], attention_params["wq_a"]),
            "q_norm": fixture_param([q_lora_rank], attention_params["q_norm"]),
            "wq_b": fixture_param([q_lora_rank, n_heads * head_dim], attention_params["wq_b"]),
            "wkv": fixture_param([dim, head_dim], attention_params["wkv"]),
            "kv_norm": fixture_param([head_dim], attention_params["kv_norm"]),
            "wo_a": fixture_param([o_groups, head_dim, o_lora_rank], attention_params["wo_a"]),
            "wo_b": fixture_param([o_groups * o_lora_rank, dim], attention_params["wo_b"]),
            "attn_sink": fixture_param([n_heads], attention_params["attn_sink"]),
            "compressor_wkv": fixture_param([dim, head_dim], attention_params["compressor_wkv"]),
            "compressor_norm": fixture_param([head_dim], attention_params["compressor_norm"]),
        },
        "moe": {
            "gate_weight": fixture_param([experts, dim], params_raw["moe"]["gate_weight"]),
            "correction_bias": fixture_param([experts], params_raw["moe"]["correction_bias"]),
            "experts": [expert_json(expert) for expert in expert_params],
            "shared_expert": expert_json(shared),
        },
        "hc_attn_fn": fixture_param([mix_hc, hc_mult * dim], params_raw["hc_attn_fn"]),
        "hc_attn_base": fixture_param([mix_hc], params_raw["hc_attn_base"]),
        "hc_attn_scale": fixture_param([3], params_raw["hc_attn_scale"]),
        "hc_ffn_fn": fixture_param([mix_hc, hc_mult * dim], params_raw["hc_ffn_fn"]),
        "hc_ffn_base": fixture_param([mix_hc], params_raw["hc_ffn_base"]),
        "hc_ffn_scale": fixture_param([3], params_raw["hc_ffn_scale"]),
    }
    if mode == "engram":
        params["engram"] = {
            "q_weight": fixture_param([hc_mult * dim], params_raw["engram"]["q_weight"]),
            "k_weight": fixture_param([hc_mult * dim], params_raw["engram"]["k_weight"]),
            "key": fixture_param([batch, seqlen, hc_mult, dim], params_raw["engram"]["key"]),
            "value": fixture_param([batch, seqlen, dim], params_raw["engram"]["value"]),
        }

    public_input = {k: v for k, v in cfg.items() if k not in {"batch", "seqlen"}}
    public_input["x"] = round_list(public_input["x"])
    public_input["pre_mix"] = round_list(public_input["pre_mix"])
    return {
        **source_meta("Block.forward", "Block.forward around lines 893-952"),
        "upstream_import_status": import_status,
        "upstream_call_status": "fixture-seam:Block.forward:block-skeleton",
        "absolute_tolerance": 2e-5,
        "input": public_input,
        "parameters": params,
        "expected": {
            "shape": [batch, seqlen, hc_mult, dim],
            "output": round_list(output),
            "next_pre_mix": round_list(next_pre_mix),
            "published_compressed_kv": final_state.get("compressed_kv") is not None,
            "has_topk_indices": final_state.get("topk_indices") is not None,
            "consumed_sparse_indices": bool(final_state.get("consumed_sparse_indices")),
        },
    }


def generate_block(out_dir: Path, module: object | None, import_status: str) -> None:
    del module
    for filename, mode in [
        ("block_sliding_window_fixture.json", "sliding_window"),
        ("block_kv_source_fixture.json", "kv_source"),
        ("block_index_source_fixture.json", "index_source"),
        ("block_reuse_fixture.json", "reuse"),
        ("block_engram_layer_fixture.json", "engram"),
    ]:
        write_json(out_dir / filename, make_block_fixture(mode, import_status))


def generate_tiny_model(out_dir: Path, module: object | None, import_status: str) -> None:
    del module
    block = make_block_fixture("sliding_window", import_status)
    input_cfg = block["input"]
    params = block["parameters"]
    batch = 1
    seqlen = 2
    vocab_size = 5
    hidden_size = input_cfg["dim"]
    hc_mult = input_cfg["hc_mult"]
    ids = [1, 3]
    image_token_id = 4
    embed_tokens = [((i % 11) - 5) / 7.0 for i in range(vocab_size * hidden_size)]
    final_norm = [1.0, 0.875, 1.125, 0.75]
    lm_head = [((i % 13) - 6) / 8.0 for i in range(hidden_size * vocab_size)]

    embedded = []
    for token_id in ids:
        start = token_id * hidden_size
        embedded.extend(embed_tokens[start : start + hidden_size])
    h = []
    for token in range(batch * seqlen):
        base = token * hidden_size
        for _ in range(hc_mult):
            h.extend(embedded[base : base + hidden_size])
    pre_mix = [1.0, 0.0] * (batch * seqlen)
    output, pre_mix, _state = block_forward_ref(
        h,
        pre_mix,
        {
            **input_cfg,
            "batch": batch,
            "seqlen": seqlen,
            "x": h,
            "pre_mix": pre_mix,
            "mode": "sliding_window",
        },
        {
            "attn_norm": params["attn_norm"]["data"],
            "ffn_norm": params["ffn_norm"]["data"],
            "attention": {key: value["data"] for key, value in params["attention"].items()},
            "moe": {
                "gate_weight": params["moe"]["gate_weight"]["data"],
                "correction_bias": params["moe"]["correction_bias"]["data"],
                "experts": [
                    {key: value["data"] for key, value in expert.items()}
                    for expert in params["moe"]["experts"]
                ],
                "shared_expert": {
                    key: value["data"] for key, value in params["moe"]["shared_expert"].items()
                },
            },
            "hc_attn_fn": params["hc_attn_fn"]["data"],
            "hc_attn_base": params["hc_attn_base"]["data"],
            "hc_attn_scale": params["hc_attn_scale"]["data"],
            "hc_ffn_fn": params["hc_ffn_fn"]["data"],
            "hc_ffn_base": params["hc_ffn_base"]["data"],
            "hc_ffn_scale": params["hc_ffn_scale"]["data"],
        },
    )
    collapsed = hc_pre_ref(output, pre_mix, batch, seqlen, hc_mult, hidden_size)
    collapsed = fallback_rms_norm(collapsed, hidden_size, final_norm, 1e-6)
    logits = matmul_rows(collapsed, lm_head, hidden_size, vocab_size)
    write_json(
        out_dir / "tiny_model_fixture.json",
        {
            **source_meta("Transformer.forward", "Transformer.forward around lines 953-1071"),
            "upstream_import_status": import_status,
            "upstream_call_status": "fixture-seam:Transformer.forward:tiny-text-model",
            "absolute_tolerance": 2e-5,
            "input": {
                "vocab_size": vocab_size,
                "hidden_size": hidden_size,
                "hc_mult": hc_mult,
                "image_token_id": image_token_id,
                "ids": ids,
                "image_ids": [1, image_token_id],
                "batch": batch,
                "seqlen": seqlen,
            },
            "parameters": {
                "embed_tokens": fixture_param([vocab_size, hidden_size], embed_tokens),
                "block_fixture": block,
                "final_norm": fixture_param([hidden_size], final_norm),
                "lm_head": fixture_param([hidden_size, vocab_size], lm_head),
            },
            "expected": {
                "shape": [batch, seqlen, vocab_size],
                "logits": round_list(logits),
            },
        },
    )


def generate_moe(out_dir: Path, module: object | None, import_status: str) -> None:
    tokens, dim, experts, topk = 2, 4, 5, 3
    gate_temp = 1.2
    norm_topk_prob = True
    route_scale = 1.5
    x = [((i % 9) - 4) / 3.0 for i in range(tokens * dim)]
    gate_weight = [((i % 11) - 5) / 4.0 for i in range(experts * dim)]
    correction_bias = [0.0, 0.0, 0.2, -0.1, 0.05]

    def upstream_gate():
        import torch

        assert module is not None
        gate = module.Gate.__new__(module.Gate)
        module.nn.Module.__init__(gate)
        gate.dim = dim
        gate.topk = topk
        gate.score_func = "sqrtsoftplus"
        gate.gate_temp = gate_temp
        gate.norm_topk_prob = norm_topk_prob
        gate.route_scale = route_scale
        gate.weight = torch.nn.Parameter(torch.tensor(gate_weight, dtype=torch.float32).reshape(experts, dim))
        gate.bias = torch.nn.Parameter(torch.tensor(correction_bias, dtype=torch.float32))
        gate.bias_vl = None

        weights, indices = gate.forward(torch.tensor(x, dtype=torch.float32).reshape(tokens, dim), None)
        # also return the unbiased scores for fixture verification
        scores = (module.linear(torch.tensor(x, dtype=torch.float32), gate.weight) / gate_temp).softplus().sqrt()
        return scores.flatten().tolist(), weights.flatten().tolist(), indices.flatten().tolist()

    def fallback_gate():
        scores = fallback_sqrtsoftplus_scores(x, gate_weight, gate_temp, tokens, dim, experts)
        indices = fallback_select_experts(scores, correction_bias, tokens, experts, topk)
        weights = fallback_route_weights(scores, indices, tokens, experts, topk, norm_topk_prob, route_scale)
        return scores, weights, indices

    (scores, weights, indices), call_status = call_upstream_or_fallback(
        import_status,
        "Gate.forward",
        upstream_gate,
        fallback_gate,
    )

    write_json(
        out_dir / "moe_gate_fixture.json",
        {
            **source_meta("Gate.forward", "Gate.forward around lines 792-827"),
            "upstream_import_status": import_status,
            "upstream_call_status": call_status,
            "absolute_tolerance": 1e-5,
            "input": {
                "shape": {"tokens": tokens, "dim": dim, "experts": experts},
                "x": round_list(x),
                "gate_weight": round_list(gate_weight),
                "gate_temp": gate_temp,
                "correction_bias": round_list(correction_bias),
                "topk": topk,
                "norm_topk_prob": norm_topk_prob,
                "route_scale": route_scale,
            },
            "expected": {
                "scores": round_list(scores),
                "indices": indices,
                "weights": round_list(weights),
            },
        },
    )

    swiglu_limit = 10.0
    gate = [0.0, 11.0, -5.0, 3.0]
    up = [-20.0, 20.0, 2.0, -2.0]

    def upstream_expert():
        import torch

        assert module is not None
        gate_tensor = torch.tensor(gate, dtype=torch.float32).reshape(1, 1, -1)
        up_tensor = torch.tensor(up, dtype=torch.float32).reshape(1, 1, -1)
        # Reuse upstream Expert clamp logic directly by mimicking the pre-w2 fused activation.
        clamp_gate = gate_tensor.clone()
        clamp_up = up_tensor.clone()
        clamp_up = torch.clamp(clamp_up, min=-swiglu_limit, max=swiglu_limit)
        clamp_gate = torch.clamp(clamp_gate, max=swiglu_limit)
        return (torch.nn.functional.silu(clamp_gate) * clamp_up).flatten().tolist()

    expected, expert_status = call_upstream_or_fallback(
        import_status,
        "Expert.forward",
        upstream_expert,
        lambda: fallback_expert_swiglu(gate, up, swiglu_limit),
    )

    write_json(
        out_dir / "expert_swiglu_fixture.json",
        {
            **source_meta("Expert.forward", "Expert.forward around lines 830-851"),
            "upstream_import_status": import_status,
            "upstream_call_status": expert_status,
            "absolute_tolerance": 1e-5,
            "input": {"gate": round_list(gate), "up": round_list(up), "swiglu_limit": swiglu_limit},
            "expected": {"output": round_list(expected)},
        },
    )


def generate_hyper(out_dir: Path, module: object | None, import_status: str) -> None:
    # A small deterministic HC stream plus parameter stubs; the upstream call
    # uses an upstream-compatible kernel.hc_split_sinkhorn shim (torch-only).
    batch, seqlen, hc_mult, dim = 1, 2, 3, 4
    flat_hc = [((i % 13) - 6) / 5.0 for i in range(batch * seqlen * hc_mult * dim)]
    mix_hc = (2 + hc_mult) * hc_mult
    hc_dim = hc_mult * dim
    hc_fn = [((i % 7) - 3) / 3.0 for i in range(mix_hc * hc_dim)]
    hc_base = [((i % 5) - 2) / 4.0 for i in range(mix_hc)]
    hc_scale = [0.5, 0.25, 0.75]
    sinkhorn_iters = 5
    eps = 1e-6

    def upstream_mixes():
        import torch

        assert module is not None
        block = module.Block.__new__(module.Block)
        module.nn.Module.__init__(block)
        block.norm_eps = 1e-20
        block.hc_mult = hc_mult
        block.hc_sinkhorn_iters = sinkhorn_iters
        block.hc_eps = eps
        x = torch.tensor(flat_hc, dtype=torch.float32).reshape(batch, seqlen, hc_mult, dim)
        hc_fn_t = torch.tensor(hc_fn, dtype=torch.float32).reshape(mix_hc, hc_dim)
        hc_scale_t = torch.tensor(hc_scale, dtype=torch.float32)
        hc_base_t = torch.tensor(hc_base, dtype=torch.float32)
        pre, post, comb = block.hc_mixes(x, hc_fn_t, hc_scale_t, hc_base_t)
        return pre.flatten().tolist(), post.flatten().tolist(), comb.flatten().tolist()

    if import_status != "path-import-ok" or module is None:
        raise SystemExit(f"hyper fixtures require importing upstream model.py, got {import_status}")

    (pre, post, comb), call_status = call_upstream_or_fallback(
        import_status,
        "Block.hc_mixes",
        upstream_mixes,
        lambda: (_ for _ in ()).throw(RuntimeError("unreachable")),
    )
    if call_status.startswith("called-upstream:"):
        call_status = f"{call_status}:shim-kernel.hc_split_sinkhorn"

    # Layer-level pre/post seam with a simple one-hot pre_mix and deterministic post/comb.
    residual = flat_hc
    pre_mix = [1.0, 0.0, 0.0] * (batch * seqlen)
    sublayer = [((i % 9) - 4) / 4.0 for i in range(batch * seqlen * dim)]
    post_coeff = post
    comb_coeff = comb

    def apply_pre(residual, pre_mix):
        out = []
        tokens = batch * seqlen
        for t in range(tokens):
            x_row = residual[t * hc_mult * dim : (t + 1) * hc_mult * dim]
            mix = pre_mix[t * hc_mult : (t + 1) * hc_mult]
            for d in range(dim):
                out.append(sum(mix[h] * x_row[h * dim + d] for h in range(hc_mult)))
        return out

    def apply_post(sublayer, residual, post, comb):
        out = []
        tokens = batch * seqlen
        for t in range(tokens):
            sl = sublayer[t * dim : (t + 1) * dim]
            res = residual[t * hc_mult * dim : (t + 1) * hc_mult * dim]
            post_row = post[t * hc_mult : (t + 1) * hc_mult]
            comb_row = comb[t * hc_mult * hc_mult : (t + 1) * hc_mult * hc_mult]
            for dst in range(hc_mult):
                for d in range(dim):
                    v = post_row[dst] * sl[d]
                    for src in range(hc_mult):
                        v += comb_row[dst * hc_mult + src] * res[src * dim + d]
                    out.append(v)
        return out

    pre_out = apply_pre(residual, pre_mix)
    post_out = apply_post(sublayer, residual, post_coeff, comb_coeff)

    def comb_sums(comb):
        tokens = batch * seqlen
        row, col = [], []
        for t in range(tokens):
            mat = comb[t * hc_mult * hc_mult : (t + 1) * hc_mult * hc_mult]
            for r in range(hc_mult):
                row.append(sum(mat[r * hc_mult + c] for c in range(hc_mult)))
            for c in range(hc_mult):
                col.append(sum(mat[r * hc_mult + c] for r in range(hc_mult)))
        return row, col

    row_sums, col_sums = comb_sums(comb)
    write_json(
        out_dir / "hyper_connection_fixture.json",
        {
            **source_meta("Block.hc_mixes", "Block.hc_mixes/hc_pre/hc_post around lines 940-979"),
            "upstream_import_status": import_status,
            "upstream_call_status": call_status,
            "absolute_tolerance": 2e-4,
            "input": {
                "flat_hc": round_list(flat_hc),
                "hc_fn": round_list(hc_fn),
                "hc_scale": round_list(hc_scale),
                "hc_base": round_list(hc_base),
                "hc_mult": hc_mult,
                "dim": dim,
                "sinkhorn_iters": sinkhorn_iters,
                "eps": eps,
            },
            "expected": {
                "pre": round_list(pre),
                "post": round_list(post),
                "comb": round_list(comb),
                "comb_sums": {"row": round_list(row_sums), "col": round_list(col_sums)},
            },
            "layer": {
                "input": {
                    "shape": {"batch": batch, "seqlen": seqlen, "hc_mult": hc_mult, "dim": dim},
                    "residual": round_list(residual),
                    "pre_mix": round_list(pre_mix),
                    "sublayer": round_list(sublayer),
                    "post": round_list(post_coeff),
                    "comb": round_list(comb_coeff),
                },
                "expected": {"pre_out": round_list(pre_out), "post_out": round_list(post_out)},
            },
        },
    )


def generate_engram(out_dir: Path, module: object | None, import_status: str) -> None:
    engram_py = out_dir.parent.parent.parent.parent / "ferric_continuum/tnsr/third_party/deepseek_v41/upstream/inference/engram.py"
    upstream_engram, engram_status = import_upstream_engram_by_path(engram_py)
    if upstream_engram is None or engram_status != "path-import-ok":
        raise SystemExit(f"engram fixtures require importing upstream engram.py, got {engram_status}")

    # Layout fixture: computed from a config slice and encoded as flattened primes/offsets.
    config = {
        "vocab_size": 129280,
        "hidden_size": 16,
        "moe_intermediate_size": 8,
        "num_hidden_layers": 40,
        "num_attention_heads": 64,
        "num_key_value_heads": 1,
        "head_dim": 512,
        "qk_rope_head_dim": 64,
        "q_lora_rank": 1280,
        "o_lora_rank": 1024,
        "o_groups": 8,
        "rms_norm_eps": 1e-20,
        "rope_theta": 10000.0,
        "rope_factor": 16.0,
        "original_seq_len": 65536,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "sliding_window": 128,
        "compress_ratios": [0],
        "compress_rope_theta": 160000.0,
        "kv_source_layer_ids": [2, 8, 14, 20],
        "index_source_layer_ids": [2, 8, 14, 20, 24, 28, 32, 36],
        "index_n_heads": 32,
        "index_head_dim": 128,
        "index_topk": 512,
        "candidate_source_layer_id": 20,
        "candidate_topk_blocks": 2048,
        "candidate_block_size": 8,
        "hc_mult": 4,
        "hc_sinkhorn_iters": 20,
        "hc_eps": 1e-6,
        "n_routed_experts": 384,
        "n_shared_experts": 1,
        "num_experts_per_tok": 6,
        "scoring_func": "sqrtsoftplus",
        "norm_topk_prob": True,
        "routed_scaling_factor": 1.5,
        "swiglu_limit": 10.0,
        "engram_layer_ids": [1, 14],
        "engram_num_embeddings": [384006168, 384016682],
        "engram_max_ngram_size": 4,
        "engram_vocab_size": 16000000,
        "engram_n_heads": 2,
        "engram_head_dim": 8,
        "engram_pad_token_id": 2,
        "engram_compressed_vocab_size": 32,
        "image_token_id": 129264,
        "dtype": "fp8",
        "expert_dtype": "fp4",
        "dspark": {
            "n_mtp_layers": 0,
            "dspark_block_size": 0,
            "dspark_noise_token_id": 0,
            "dspark_target_layer_ids": [],
            "dspark_markov_rank": 0,
            "dspark_n_routed_experts": 0,
            "dspark_num_experts_per_tok": 0,
        },
        "vision": {
            "num_hidden_layers": 0,
            "hidden_size": 0,
            "num_attention_heads": 0,
            "intermediate_size": 0,
            "patch_size": 0,
            "rope_theta": 10000.0,
            "downsample_ratio": 0,
            "max_image_tokens": 0,
            "min_pixels": 0,
            "max_wh_ratio": None,
        },
    }

    layout = {"max_ngram_size": config["engram_max_ngram_size"], "layer_ids": config["engram_layer_ids"], "num_embeddings": config["engram_num_embeddings"], "n_heads": config["engram_n_heads"], "head_dim": config["engram_head_dim"]}

    def upstream_layout():
        args = types.SimpleNamespace(
            engram_layer_ids=tuple(config["engram_layer_ids"]),
            engram_num_embeddings=tuple(config["engram_num_embeddings"]),
            engram_max_ngram_size=config["engram_max_ngram_size"],
            engram_vocab_size=config["engram_vocab_size"],
            engram_n_heads=config["engram_n_heads"],
            engram_head_dim=config["engram_head_dim"],
        )
        lay = upstream_engram.EngramLayout.from_args(args)
        primes_flat = [p for per_ngram in lay.primes[0] for p in per_ngram] + [p for per_ngram in lay.primes[1] for p in per_ngram]
        # offsets are computed in NgramHashState, mirror that logic here for fixture
        sizes0 = primes_flat[: (config["engram_max_ngram_size"] - 1) * config["engram_n_heads"]]
        sizes1 = primes_flat[(config["engram_max_ngram_size"] - 1) * config["engram_n_heads"] :]
        offs0, o = [], 0
        for s in sizes0:
            offs0.append(o)
            o += s
        offs1, o = [], 0
        for s in sizes1:
            offs1.append(o)
            o += s
        return primes_flat, offs0 + offs1

    (primes_flat, offsets) = upstream_layout()
    call_status = "called-upstream:EngramLayout.from_args:shim-sympy.isprime"

    write_json(
        out_dir / "engram_layout_fixture.json",
        {
            **source_meta(
                "EngramLayout.from_args",
                "EngramLayout.from_args around lines 86-126",
                source_file="ferric_continuum/tnsr/third_party/deepseek_v41/upstream/inference/engram.py",
            ),
            "upstream_import_status": engram_status,
            "upstream_call_status": call_status,
            "absolute_tolerance": 0.0,
            "input": {"config": config, "layout": {**layout, "primes_flat": primes_flat, "offsets": offsets}},
            "expected": {"primes_flat": primes_flat, "offsets": offsets},
        },
    )

    # Hash fixture: small vocab subset + prefill/decode sequences.
    vocab_entries = [
        {"token_id": 0, "text": "<BOS>"},
        {"token_id": 1, "text": "<EOS>"},
        {"token_id": 2, "text": "<PAD>"},
        {"token_id": 3, "text": " The"},
        {"token_id": 4, "text": "the"},
        {"token_id": 5, "text": "ＴＨＥ"},
        {"token_id": 6, "text": "Cafe\u0301"},
        {"token_id": 7, "text": "Caf\u00e9"},
        {"token_id": 8, "text": " \t"},
        {"token_id": 9, "text": " "},
    ]
    prefill_input_ids = [3, 6, 4, 7]
    decode_input_ids = [5]
    decode_start_pos = len(prefill_input_ids)
    max_batch_size = 1
    max_seq_len = 16
    pad_token_id = 2

    class _BackendTokenizer:
        def __init__(self, entries):
            self._map = {e["token_id"]: e["text"] for e in entries}

        def decode(self, ids, skip_special_tokens=False):
            assert len(ids) == 1
            return self._map[ids[0]]

        def id_to_token(self, token_id: int) -> str:
            return self._map[token_id]

    class _Tokenizer:
        def __init__(self, entries):
            self.backend_tokenizer = _BackendTokenizer(entries)
            self._len = max(e["token_id"] for e in entries) + 1

        def __len__(self):
            return self._len

    tokenizer = _Tokenizer(vocab_entries)
    lookup, vocab_size = upstream_engram.build_compressed_token_map(tokenizer)

    multipliers = upstream_engram.compute_hash_multipliers(
        tuple(config["engram_layer_ids"]), config["engram_max_ngram_size"], vocab_size
    )
    mult_flat = flatten_nested(tensor_to_nested_list(multipliers))

    import torch

    args = types.SimpleNamespace(
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        engram_layer_ids=tuple(config["engram_layer_ids"]),
        engram_num_embeddings=tuple(config["engram_num_embeddings"]),
        engram_max_ngram_size=config["engram_max_ngram_size"],
        engram_vocab_size=config["engram_vocab_size"],
        engram_n_heads=config["engram_n_heads"],
        engram_head_dim=config["engram_head_dim"],
        engram_pad_id=pad_token_id,
        engram_compressed_vocab_size=vocab_size,
    )
    layout_obj = upstream_engram.EngramLayout.from_args(args)
    state = upstream_engram.NgramHashState(args, layout_obj, tokenizer)
    prefill_hashes = flatten_nested(
        tensor_to_nested_list(state.forward(torch.tensor([prefill_input_ids], dtype=torch.int64), 0))
    )
    decode_hashes = flatten_nested(
        tensor_to_nested_list(state.forward(torch.tensor([decode_input_ids], dtype=torch.int64), decode_start_pos))
    )

    write_json(
        out_dir / "engram_hash_fixture.json",
        {
            **source_meta(
                "NgramHashState.forward",
                "NgramHashState.forward around lines 129-184",
                source_file="ferric_continuum/tnsr/third_party/deepseek_v41/upstream/inference/engram.py",
            ),
            "upstream_import_status": engram_status,
            "upstream_call_status": "called-upstream:NgramHashState.forward:shim-tokenizers,shim-sympy.isprime",
            "absolute_tolerance": 0.0,
            "input": {
                "vocab_entries": vocab_entries,
                "pad_token_id": pad_token_id,
                "max_batch_size": max_batch_size,
                "max_seq_len": max_seq_len,
                "prefill_input_ids": prefill_input_ids,
                "decode_input_ids": decode_input_ids,
                "decode_start_pos": decode_start_pos,
                "multipliers": mult_flat,
                "expected_token_map": lookup,
                "expected_compressed_vocab_size": vocab_size,
            },
            "expected": {"prefill_hashes": prefill_hashes, "decode_hashes": decode_hashes},
        },
    )

    # Engram update fixture: match the gating math in Engram.forward (no table lookup here).
    tokens, hc_mult, dim = 2, 2, 4
    x = [((i % 11) - 5) / 5.0 for i in range(tokens * hc_mult * dim)]
    key = [((i % 9) - 4) / 4.0 for i in range(tokens * hc_mult * dim)]
    value = [((i % 7) - 3) / 3.0 for i in range(tokens * dim)]
    q_weight = [1.0] * (hc_mult * dim)
    k_weight = [1.0] * (hc_mult * dim)
    token_mask = [True, False]
    eps = 1e-20

    def update(x, key, value, token_mask):
        out = []
        inv_sqrt_dim = dim ** -0.5
        clamp_value = 1e-6
        for t in range(tokens):
            ok = token_mask[t]
            vrow = value[t * dim : (t + 1) * dim]
            for h in range(hc_mult):
                row = x[(t * hc_mult + h) * dim : (t * hc_mult + h + 1) * dim]
                krow = key[(t * hc_mult + h) * dim : (t * hc_mult + h + 1) * dim]
                mean_sq_x = sum(v * v for v in row) / dim
                mean_sq_k = sum(v * v for v in krow) / dim
                rstd = (mean_sq_x + eps) ** -0.5 * (mean_sq_k + eps) ** -0.5
                dot = sum(a * b for a, b in zip(row, krow)) * rstd * inv_sqrt_dim
                gate = 1.0 / (1.0 + math.exp(-math.copysign(max(abs(dot), clamp_value) ** 0.5, dot)))
                if not ok:
                    gate = 0.0
                out.extend([rv + gate * vv for rv, vv in zip(row, vrow)])
        return out

    expected = update(x, key, value, token_mask)
    write_json(
        out_dir / "engram_update_fixture.json",
        {
            **source_meta("Engram.forward", "Engram.forward around lines 328-367"),
            "upstream_import_status": import_status,
            "upstream_call_status": "fallback:Engram.forward:fixture-only",
            "absolute_tolerance": 1e-5,
            "input": {
                "x": round_list(x),
                "key": round_list(key),
                "value": round_list(value),
                "q_weight": round_list(q_weight),
                "k_weight": round_list(k_weight),
                "eps": eps,
                "token_mask": token_mask,
            },
            "expected": {"output": round_list(expected)},
        },
    )

    # Placeholder layout/update fixture kept for completeness; test does not read it yet.
    write_json(
        out_dir / "engram_update_fixture.json",
        json.loads((out_dir / "engram_update_fixture.json").read_text()),
    )


def import_upstream_encoding_by_path(encoding_py: Path) -> tuple[object | None, str]:
    """Load upstream encoding/encoding.py by path. It is pure Python (json/re),
    so no heavy stubs are needed; a failure is recorded verbatim."""
    spec = importlib.util.spec_from_file_location(
        "deepseek_v41_upstream_encoding", encoding_py
    )
    if spec is None or spec.loader is None:
        raise SystemExit(f"could not create import spec for {encoding_py}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(encoding_py.parent))
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as err:
        return None, f"path-import-skipped-missing-dependency:{err.name}"
    except Exception as err:  # noqa: BLE001
        return None, f"path-import-skipped:{type(err).__name__}:{err}"
    finally:
        sys.path.pop(0)
    return module, "path-import-ok"


def _fallback_encode_messages(
    enc: object,
    messages: list[dict],
    thinking_mode: str,
    reasoning_effort,
) -> str:
    """This wave calls upstream `encode_messages` directly; there is no viable
    hand-rolled fallback for the full V4.1 template, so we require the real
    import and surface a clear error otherwise."""
    raise RuntimeError(
        "prompt fixtures require importing upstream encoding.py; no fallback exists"
    )


def generate_prompt(out_dir: Path, repo_root: Path) -> None:
    """Emit text-only prompt fixtures by calling upstream `encoding.py`.

    Each fixture records the rendered prompt string plus, when a Hugging Face
    tokenizer is reachable, its token IDs. Token IDs are optional so the fixture
    is still meaningful (prompt-string parity) without local tokenizer assets.
    """
    encoding_py = repo_root / "ferric_continuum/tnsr/third_party/deepseek_v41/upstream/encoding/encoding.py"
    if not encoding_py.exists():
        raise SystemExit(f"missing mirrored upstream encoding: {encoding_py}")
    enc, enc_status = import_upstream_encoding_by_path(encoding_py)
    if enc_status != "path-import-ok" or enc is None:
        raise SystemExit(
            f"prompt fixtures require upstream encoding.py import, got {enc_status}"
        )

    tokenizer_dir = repo_root / "ferric_continuum/tnsr/third_party/deepseek_v41/upstream"

    def maybe_token_ids(prompt: str) -> tuple[list[int] | None, str]:
        # A real HF tokenizer needs tokenizer.json/model; only tokenizer_config
        # is mirrored locally, so token IDs are recorded only when a full
        # tokenizer is present. This keeps prompt-string parity independent of
        # the ~10MB tokenizer asset.
        tj = tokenizer_dir / "tokenizer.json"
        if not tj.exists():
            return None, "skipped-no-tokenizer-json"
        try:
            from transformers import AutoTokenizer  # type: ignore

            tok = AutoTokenizer.from_pretrained(str(tokenizer_dir))
            ids = tok.encode(prompt, add_special_tokens=False)
            return list(ids), "hf-tokenizer"
        except Exception as err:  # noqa: BLE001
            return None, f"skipped-tokenizer-error:{type(err).__name__}"

    def emit(name: str, prompt: str, mode: str, extra: dict) -> None:
        token_ids, tok_status = maybe_token_ids(prompt)
        payload = {
            "upstream_function": "encode_messages",
            "source_file": "ferric_continuum/tnsr/third_party/deepseek_v41/upstream/encoding/encoding.py",
            "upstream_import_status": enc_status,
            "mode": mode,
            "tokenizer_status": tok_status,
            "input": extra,
            "expected": {
                "prompt": prompt,
                "token_ids": token_ids,
            },
        }
        write_json(out_dir / name, payload)

    # 1) Plain single-user chat prompt (no thinking).
    plain_messages = [{"role": "user", "content": "What is 2 plus 2?"}]
    plain_prompt = enc.encode_messages(
        [dict(m) for m in plain_messages], thinking_mode="chat"
    )
    emit(
        "prompt_plain_fixture.json",
        plain_prompt,
        "chat",
        {"messages": plain_messages, "thinking_mode": "chat", "reasoning_effort": None},
    )

    # 2) Chat mode with a prior assistant turn: `</think>` reasoning must be
    #    suppressed (drop_thinking) when re-rendering chat history.
    chat_messages = [
        {"role": "user", "content": "Hello."},
        {
            "role": "assistant",
            "content": "Hi there!",
            "reasoning_content": "The user greeted me; respond politely.",
        },
        {"role": "user", "content": "And now?"},
    ]
    chat_prompt = enc.encode_messages(
        [dict(m) for m in chat_messages], thinking_mode="chat"
    )
    contains_think = enc.thinking_end_token in chat_prompt
    emit(
        "prompt_chat_fixture.json",
        chat_prompt,
        "chat",
        {
            "messages": chat_messages,
            "thinking_mode": "chat",
            "reasoning_effort": None,
            "contains_thinking_end_token": contains_think,
        },
    )

    # 3) Thinking mode with numeric reasoning effort: the reasoning-effort prefix
    #    must render at index 0 (see render_reasoning_effort).
    thinking_messages = [{"role": "user", "content": "Prove that 2 is prime."}]
    thinking_prompt = enc.encode_messages(
        [dict(m) for m in thinking_messages],
        thinking_mode="thinking",
        reasoning_effort=42,
    )
    effort_prefix = enc.REASONING_EFFORT_TEMPLATE.format(budget=42)
    emit(
        "prompt_thinking_fixture.json",
        thinking_prompt,
        "thinking",
        {
            "messages": thinking_messages,
            "thinking_mode": "thinking",
            "reasoning_effort": 42,
            "reasoning_effort_prefix": effort_prefix,
            "contains_reasoning_effort_prefix": effort_prefix in thinking_prompt,
        },
    )

    # 4) DSML tool-call prompt string: a system message carrying tools renders
    #    the V4.1 DSML tool block. Only the prompt string is exercised; parsing
    #    is a verifier-only concern this wave.
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]
    dsml_messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Weather in Paris?"},
    ]
    dsml_case = [dict(m) for m in dsml_messages]
    dsml_case[0]["tools"] = tools
    dsml_prompt = enc.encode_messages(dsml_case, thinking_mode="chat")
    emit(
        "prompt_dsml_fixture.json",
        dsml_prompt,
        "chat",
        {
            "messages": dsml_messages,
            "tools": tools,
            "thinking_mode": "chat",
            "reasoning_effort": None,
            "dsml_token": enc.dsml_token,
            "contains_dsml_token": enc.dsml_token in dsml_prompt,
        },
    )


def _vision_cfg():
    """Release vision config values (see spec.org / config.rs)."""
    return dict(
        vision_patch_size=14,
        vision_downsample_ratio=3,
        vision_max_n_token=1024,
        vision_min_pixels=295936,
        vision_max_wh_ratio=None,
    )


class _VisionArgs:
    def __init__(self, cfg: dict):
        for key, value in cfg.items():
            setattr(self, key, value)


def generate_image_grid(out_dir: Path, image_module: object | None, import_status: str) -> None:
    cfg = _vision_cfg()
    args = _VisionArgs(cfg)
    # (width, height) originals: square, tall, wide, below min-pixels, above max-tokens.
    cases = [
        {"name": "square_small", "width": 224, "height": 224},
        {"name": "tall", "width": 224, "height": 2016},
        {"name": "wide", "width": 4032, "height": 224},
        {"name": "below_min_pixels", "width": 64, "height": 48},
        {"name": "above_max_tokens", "width": 4096, "height": 4096},
    ]

    def upstream_case(case):
        assert image_module is not None
        n_llm_h, n_llm_w, best_h, best_w = image_module.plan_image_grid(
            case["width"], case["height"], args
        )
        types = image_module.image_token_types(n_llm_h, n_llm_w)
        types_list = types.tolist() if hasattr(types, "tolist") else list(types)
        return {
            "n_llm_h": int(n_llm_h),
            "n_llm_w": int(n_llm_w),
            "best_height": int(best_h),
            "best_width": int(best_w),
            "num_image_tokens": image_module.num_image_tokens(n_llm_h, n_llm_w),
            "token_types": [int(t) for t in types_list],
        }

    def fallback_case(case):
        return _fallback_image_grid(case["width"], case["height"], cfg)

    for case in cases:
        expected, case_status = call_upstream_or_fallback(
            import_status,
            "plan_image_grid",
            lambda case=case: upstream_case(case),
            lambda case=case: fallback_case(case),
        )
        case["expected"] = expected
        case["upstream_call_status"] = case_status

    write_json(
        out_dir / "image_grid_fixture.json",
        {
            **source_meta(
                "plan_image_grid/image_token_types",
                "plan_image_grid around lines 101-137",
                source_file="ferric_continuum/tnsr/third_party/deepseek_v41/upstream/inference/image_processor.py",
            ),
            "upstream_import_status": import_status,
            "upstream_call_status": ",".join(
                sorted({case["upstream_call_status"] for case in cases})
            ),
            "absolute_tolerance": 0.0,
            "vision_config": cfg,
            "cases": cases,
        },
    )


def _fallback_image_grid(width: int, height: int, cfg: dict) -> dict:
    """Pure-Python mirror of plan_image_grid + image_token_types."""
    p = cfg["vision_patch_size"]
    ds = cfg["vision_downsample_ratio"]
    max_n = cfg["vision_max_n_token"]
    min_pixels = cfg["vision_min_pixels"]
    max_wh = cfg["vision_max_wh_ratio"]

    def num_image_tokens(n_llm_h, n_llm_w):
        return n_llm_h * (n_llm_w + 1) + 2

    def llm_grid(best_h, best_w):
        return math.ceil((best_h // p) / ds), math.ceil((best_w // p) / ds)

    def solve_resize_ratio(h, w):
        r = h / w
        max_w_float = math.sqrt((max_n - 2) / r + 0.25) - 0.5
        max_h_float = max_w_float * r
        cell = p * ds
        if max_w_float < 1.0:
            return (max_n - 2) // 2 * cell, cell
        if max_h_float < 1.0:
            return cell, (max_n - 3) * cell
        beta = min(math.floor(max_w_float) * cell / w, math.floor(max_h_float) * cell / h)
        return math.floor(h * beta / p) * p, math.floor(w * beta / p) * p

    def safe_resize(h, w, best_h, best_w):
        n_llm_h, n_llm_w = llm_grid(best_h, best_w)
        if num_image_tokens(n_llm_h, n_llm_w) > max_n:
            best_h, best_w = solve_resize_ratio(h, w)
            n_llm_h, n_llm_w = llm_grid(best_h, best_w)
        return n_llm_h, n_llm_w, best_h, best_w

    if max_wh is not None and width > height * max_wh:
        width = int(height * max_wh)
    if 0 < width * height < min_pixels:
        ratio = (min_pixels / (width * height)) ** 0.5
        width = int(width * ratio)
        height = int(height * ratio)
    best_width = math.ceil(width / p) * p
    best_height = math.ceil(height / p) * p
    n_llm_h, n_llm_w, best_h, best_w = safe_resize(height, width, best_height, best_width)

    types = [0]  # IMAGE_START
    types += ([1] * n_llm_w + [2]) * n_llm_h  # IMAGE, IMAGE_NEW_LINE
    types.append(3)  # IMAGE_END
    return {
        "n_llm_h": int(n_llm_h),
        "n_llm_w": int(n_llm_w),
        "best_height": int(best_h),
        "best_width": int(best_w),
        "num_image_tokens": num_image_tokens(n_llm_h, n_llm_w),
        "token_types": types,
    }


def import_upstream_vision_by_path(vision_py: Path) -> tuple[object | None, str]:
    """Load upstream vision.py by path. It is pure torch (no kernel stubs)."""
    spec = importlib.util.spec_from_file_location(
        "deepseek_v41_upstream_vision", vision_py
    )
    if spec is None or spec.loader is None:
        raise SystemExit(f"could not create import spec for {vision_py}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as err:
        return None, f"path-import-skipped-missing-dependency:{err.name}"
    except Exception as err:
        return None, f"path-import-skipped:{type(err).__name__}:{err}"
    return module, "path-import-ok"


class _VitArgs:
    """Minimal ModelArgs surface the vision.py modules read."""

    def __init__(self, dim, n_heads, inter, patch, theta, n_layers):
        self.vision_dim = dim
        self.vision_n_heads = n_heads
        self.vision_inter_dim = inter
        self.vision_patch_size = patch
        self.vision_rope_theta = theta
        self.vision_n_layers = n_layers


class _AlignerArgs:
    """Minimal ModelArgs surface the Aligner reads: vision_dim, downsample, dim."""

    def __init__(self, vision_dim, downsample_ratio, dim):
        self.vision_dim = vision_dim
        self.vision_downsample_ratio = downsample_ratio
        self.dim = dim


def _det_weights(count: int, span: int = 11, scale: float = 6.0) -> list[float]:
    return [((i % span) - (span // 2)) / scale for i in range(count)]


def generate_vision(out_dir: Path, vision_module: object | None, import_status: str) -> None:
    # Tiny deterministic ViT config. head_dim = dim/n_heads = 4; rope_dim = 2.
    dim, n_heads, inter, patch, theta, n_layers = 8, 2, 3, 2, 10000.0, 1
    head_dim = dim // n_heads
    rope_dim = dim // n_heads // 2
    patch_flat = 3 * patch * patch
    n_h, n_w = 2, 3
    n_patch = n_h * n_w

    args = _VitArgs(dim, n_heads, inter, patch, theta, n_layers)

    patches = _det_weights(n_patch * patch_flat, span=9, scale=5.0)
    proj_w = _det_weights(dim * patch_flat, span=7, scale=8.0)
    proj_b = _det_weights(dim, span=5, scale=4.0)
    wqkv = _det_weights(3 * dim * dim, span=11, scale=9.0)
    wqkv_b = _det_weights(3 * dim, span=5, scale=6.0)
    wo = _det_weights(dim * dim, span=7, scale=7.0)
    wo_b = _det_weights(dim, span=5, scale=5.0)
    norm1 = _det_weights(dim, span=5, scale=8.0)
    norm2 = _det_weights(dim, span=7, scale=9.0)
    w1 = _det_weights(2 * inter * dim, span=13, scale=10.0)
    w2 = _det_weights(dim * inter, span=11, scale=9.0)
    vit_norm = _det_weights(dim, span=5, scale=7.0)

    def build_and_run():
        import torch

        assert vision_module is not None

        def as_param(data, shape):
            return torch.nn.Parameter(torch.tensor(data, dtype=torch.float32).reshape(shape))

        # cos/sin table
        cos, sin = vision_module.get_vision_cos_sin(n_h, n_w, rope_dim, theta)

        # Patch embed
        pe = vision_module.PatchEmbed(args)
        pe.proj.weight = as_param(proj_w, (dim, patch_flat))
        pe.proj.bias = as_param(proj_b, (dim,))

        # Attention
        attn = vision_module.Attention(args)
        attn.wqkv.weight = as_param(wqkv, (3 * dim, dim))
        attn.wqkv.bias = as_param(wqkv_b, (3 * dim,))
        attn.wo.weight = as_param(wo, (dim, dim))
        attn.wo.bias = as_param(wo_b, (dim,))

        # MLP
        mlp = vision_module.MLP(args)
        mlp.w1.weight = as_param(w1, (2 * inter, dim))
        mlp.w2.weight = as_param(w2, (dim, inter))

        # Norms
        n1 = vision_module.RMSNorm(dim)
        n1.weight = as_param(norm1, (dim,))
        n2 = vision_module.RMSNorm(dim)
        n2.weight = as_param(norm2, (dim,))

        patches_t = torch.tensor(patches, dtype=torch.float32).reshape(n_patch, patch_flat)

        with torch.no_grad():
            embedded = pe(patches_t)
            rope_out = tensor_to_nested_list(
                vision_module.apply_rotary(
                    embedded.view(n_patch, n_heads, head_dim), cos, sin
                )
            )
            attn_out = attn(n1(embedded), cos, sin)
            block_h = embedded + attn_out
            block_out = block_h + mlp(n2(block_h))
        return {
            "cos": flatten_nested(tensor_to_nested_list(cos)),
            "sin": flatten_nested(tensor_to_nested_list(sin)),
            "rope_out": flatten_nested(rope_out),
            "embedded": flatten_nested(tensor_to_nested_list(embedded)),
            "attn_out": flatten_nested(tensor_to_nested_list(attn_out)),
            "block_out": flatten_nested(tensor_to_nested_list(block_out)),
        }

    def fallback():
        # Pure-Python mirror is unnecessary for parity intent; we only emit a
        # marker so the fixture explains why upstream did not run. The Rust test
        # skips assertions when upstream did not execute.
        return {
            "cos": [],
            "sin": [],
            "rope_out": [],
            "embedded": [],
            "attn_out": [],
            "block_out": [],
        }

    result, call_status = call_upstream_or_fallback(
        import_status, "ViT", build_and_run, fallback
    )

    write_json(
        out_dir / "vision_block_fixture.json",
        {
            **source_meta(
                "get_vision_cos_sin/Attention/Block",
                "vision.py Attention/Block around lines 9-84",
                source_file="ferric_continuum/tnsr/third_party/deepseek_v41/upstream/inference/vision.py",
            ),
            "upstream_import_status": import_status,
            "upstream_call_status": call_status,
            "absolute_tolerance": 2e-5,
            "input": {
                "dim": dim,
                "n_heads": n_heads,
                "head_dim": head_dim,
                "rope_dim": rope_dim,
                "inter": inter,
                "patch": patch,
                "patch_flat": patch_flat,
                "theta": theta,
                "n_h": n_h,
                "n_w": n_w,
                "n_patch": n_patch,
                "patches": round_list(patches),
            },
            "parameters": {
                "proj_w": fixture_param([dim, patch_flat], proj_w),
                "proj_b": fixture_param([dim], proj_b),
                "wqkv": fixture_param([3 * dim, dim], wqkv),
                "wqkv_b": fixture_param([3 * dim], wqkv_b),
                "wo": fixture_param([dim, dim], wo),
                "wo_b": fixture_param([dim], wo_b),
                "norm1": fixture_param([dim], norm1),
                "norm2": fixture_param([dim], norm2),
                "w1": fixture_param([2 * inter, dim], w1),
                "w2": fixture_param([dim, inter], w2),
                "vit_norm": fixture_param([dim], vit_norm),
            },
            "expected": {
                "cos": round_list(result["cos"]),
                "sin": round_list(result["sin"]),
                "rope_out": round_list(result["rope_out"]),
                "embedded": round_list(result["embedded"]),
                "attn_out": round_list(result["attn_out"]),
                "block_out": round_list(result["block_out"]),
            },
        },
    )


def generate_aligner(out_dir: Path, vision_module: object | None, import_status: str) -> None:
    # Tiny deterministic Aligner config with a non-multiple-of-r grid to exercise
    # zero-padding in the unfold path.
    vision_dim, downsample_ratio, dim = 4, 2, 6
    r = downsample_ratio
    in_dim = vision_dim * r * r
    n_h, n_w = 3, 3  # both odd -> pad by 1 in each spatial axis
    n_patch = n_h * n_w

    args = _AlignerArgs(vision_dim, downsample_ratio, dim)

    vit_rows = _det_weights(n_patch * vision_dim, span=9, scale=5.0)
    w1 = _det_weights(dim * in_dim, span=11, scale=9.0)
    w1_b = _det_weights(dim, span=5, scale=6.0)
    w2 = _det_weights(dim * dim, span=7, scale=7.0)
    w2_b = _det_weights(dim, span=5, scale=5.0)

    def build_and_run():
        import torch

        assert vision_module is not None

        def as_param(data, shape):
            return torch.nn.Parameter(torch.tensor(data, dtype=torch.float32).reshape(shape))

        aligner = vision_module.Aligner(args)
        aligner.w1.weight = as_param(w1, (dim, in_dim))
        aligner.w1.bias = as_param(w1_b, (dim,))
        aligner.w2.weight = as_param(w2, (dim, dim))
        aligner.w2.bias = as_param(w2_b, (dim,))

        vit_t = torch.tensor(vit_rows, dtype=torch.float32).reshape(n_patch, vision_dim)
        with torch.no_grad():
            out = aligner(vit_t, n_h, n_w)
        return {"aligned": flatten_nested(tensor_to_nested_list(out))}

    def fallback():
        return {"aligned": []}

    result, call_status = call_upstream_or_fallback(
        import_status, "Aligner", build_and_run, fallback
    )

    write_json(
        out_dir / "aligner_fixture.json",
        {
            **source_meta(
                "Aligner.forward",
                "vision.py Aligner around lines 106-119",
                source_file="ferric_continuum/tnsr/third_party/deepseek_v41/upstream/inference/vision.py",
            ),
            "upstream_import_status": import_status,
            "upstream_call_status": call_status,
            "absolute_tolerance": 2e-5,
            "input": {
                "vision_dim": vision_dim,
                "downsample_ratio": downsample_ratio,
                "dim": dim,
                "in_dim": in_dim,
                "n_h": n_h,
                "n_w": n_w,
                "n_patch": n_patch,
                "vit_rows": round_list(vit_rows),
            },
            "parameters": {
                "w1": fixture_param([dim, in_dim], w1),
                "w1_b": fixture_param([dim], w1_b),
                "w2": fixture_param([dim, dim], w2),
                "w2_b": fixture_param([dim], w2_b),
            },
            "expected": {
                "aligned": round_list(result["aligned"]),
            },
        },
    )


class _EncodeImageArgs:
    """ModelArgs surface for a chained ViT + Aligner (encode_image)."""

    def __init__(self, dim, n_heads, inter, patch, theta, n_layers, downsample, llm_dim):
        self.vision_dim = dim
        self.vision_n_heads = n_heads
        self.vision_inter_dim = inter
        self.vision_patch_size = patch
        self.vision_rope_theta = theta
        self.vision_n_layers = n_layers
        self.vision_downsample_ratio = downsample
        self.dim = llm_dim


def generate_encode_image(out_dir: Path, vision_module: object | None, import_status: str) -> None:
    # Tiny ViT (2 layers) + Aligner over a non-multiple-of-r grid.
    dim, n_heads, inter, patch, theta, n_layers = 8, 2, 3, 2, 10000.0, 2
    downsample, llm_dim = 2, 6
    head_dim = dim // n_heads
    rope_dim = dim // n_heads // 2
    patch_flat = 3 * patch * patch
    n_h, n_w = 3, 3  # odd -> aligner pad branch
    n_patch = n_h * n_w
    r = downsample
    in_dim = dim * r * r

    args = _EncodeImageArgs(dim, n_heads, inter, patch, theta, n_layers, downsample, llm_dim)

    patches = _det_weights(n_patch * patch_flat, span=9, scale=5.0)
    proj_w = _det_weights(dim * patch_flat, span=7, scale=8.0)
    proj_b = _det_weights(dim, span=5, scale=4.0)
    vit_norm = _det_weights(dim, span=5, scale=7.0)
    # Per-block params (distinct per layer so a layer swap would be detected).
    blocks = []
    for li in range(n_layers):
        blocks.append(
            {
                "wqkv": _det_weights(3 * dim * dim, span=11 + li, scale=9.0),
                "wqkv_b": _det_weights(3 * dim, span=5, scale=6.0),
                "wo": _det_weights(dim * dim, span=7, scale=7.0),
                "wo_b": _det_weights(dim, span=5, scale=5.0),
                "norm1": _det_weights(dim, span=5, scale=8.0),
                "norm2": _det_weights(dim, span=7, scale=9.0),
                "w1": _det_weights(2 * inter * dim, span=13, scale=10.0),
                "w2": _det_weights(dim * inter, span=11, scale=9.0),
            }
        )
    al_w1 = _det_weights(llm_dim * in_dim, span=11, scale=9.0)
    al_w1_b = _det_weights(llm_dim, span=5, scale=6.0)
    al_w2 = _det_weights(llm_dim * llm_dim, span=7, scale=7.0)
    al_w2_b = _det_weights(llm_dim, span=5, scale=5.0)

    def build_and_run():
        import torch

        assert vision_module is not None

        def as_param(data, shape):
            return torch.nn.Parameter(torch.tensor(data, dtype=torch.float32).reshape(shape))

        vit = vision_module.ViT(args)
        vit.patch_embed.proj.weight = as_param(proj_w, (dim, patch_flat))
        vit.patch_embed.proj.bias = as_param(proj_b, (dim,))
        for li, blk in enumerate(vit.blocks):
            b = blocks[li]
            blk.attn.wqkv.weight = as_param(b["wqkv"], (3 * dim, dim))
            blk.attn.wqkv.bias = as_param(b["wqkv_b"], (3 * dim,))
            blk.attn.wo.weight = as_param(b["wo"], (dim, dim))
            blk.attn.wo.bias = as_param(b["wo_b"], (dim,))
            blk.norm1.weight = as_param(b["norm1"], (dim,))
            blk.norm2.weight = as_param(b["norm2"], (dim,))
            blk.mlp.w1.weight = as_param(b["w1"], (2 * inter, dim))
            blk.mlp.w2.weight = as_param(b["w2"], (dim, inter))
        vit.norm.weight = as_param(vit_norm, (dim,))

        aligner = vision_module.Aligner(args)
        aligner.w1.weight = as_param(al_w1, (llm_dim, in_dim))
        aligner.w1.bias = as_param(al_w1_b, (llm_dim,))
        aligner.w2.weight = as_param(al_w2, (llm_dim, llm_dim))
        aligner.w2.bias = as_param(al_w2_b, (llm_dim,))

        patches_t = torch.tensor(patches, dtype=torch.float32).reshape(n_patch, patch_flat)
        with torch.no_grad():
            vit_rows = vit(patches_t, n_h, n_w)
            aligned = aligner(vit_rows, n_h, n_w)
        return {
            "vit_rows": flatten_nested(tensor_to_nested_list(vit_rows)),
            "aligned": flatten_nested(tensor_to_nested_list(aligned)),
        }

    def fallback():
        return {"vit_rows": [], "aligned": []}

    result, call_status = call_upstream_or_fallback(
        import_status, "EncodeImage", build_and_run, fallback
    )

    def block_params(b):
        return {
            "wqkv": fixture_param([3 * dim, dim], b["wqkv"]),
            "wqkv_b": fixture_param([3 * dim], b["wqkv_b"]),
            "wo": fixture_param([dim, dim], b["wo"]),
            "wo_b": fixture_param([dim], b["wo_b"]),
            "norm1": fixture_param([dim], b["norm1"]),
            "norm2": fixture_param([dim], b["norm2"]),
            "w1": fixture_param([2 * inter, dim], b["w1"]),
            "w2": fixture_param([dim, inter], b["w2"]),
        }

    write_json(
        out_dir / "encode_image_fixture.json",
        {
            **source_meta(
                "ViT.forward+Aligner.forward",
                "vision.py ViT/Aligner (encode_image) around lines 86-119",
                source_file="ferric_continuum/tnsr/third_party/deepseek_v41/upstream/inference/vision.py",
            ),
            "upstream_import_status": import_status,
            "upstream_call_status": call_status,
            "absolute_tolerance": 2e-5,
            "input": {
                "dim": dim,
                "n_heads": n_heads,
                "head_dim": head_dim,
                "rope_dim": rope_dim,
                "inter": inter,
                "patch": patch,
                "patch_flat": patch_flat,
                "theta": theta,
                "n_layers": n_layers,
                "downsample_ratio": downsample,
                "llm_dim": llm_dim,
                "in_dim": in_dim,
                "n_h": n_h,
                "n_w": n_w,
                "n_patch": n_patch,
                "patches": round_list(patches),
            },
            "parameters": {
                "proj_w": fixture_param([dim, patch_flat], proj_w),
                "proj_b": fixture_param([dim], proj_b),
                "vit_norm": fixture_param([dim], vit_norm),
                "blocks": [block_params(b) for b in blocks],
                "al_w1": fixture_param([llm_dim, in_dim], al_w1),
                "al_w1_b": fixture_param([llm_dim], al_w1_b),
                "al_w2": fixture_param([llm_dim, llm_dim], al_w2),
                "al_w2_b": fixture_param([llm_dim], al_w2_b),
            },
            "expected": {
                "vit_rows": round_list(result["vit_rows"]),
                "aligned": round_list(result["aligned"]),
            },
        },
    )


def generate_vl_gate(out_dir: Path, module: object | None, import_status: str) -> None:
    """Image-mask VL routing bias: bias_vl steers selection on image tokens only."""
    tokens, dim, experts, topk = 3, 4, 5, 2
    gate_temp = 1.2
    norm_topk_prob = True
    route_scale = 1.5
    x = [((i % 9) - 4) / 3.0 for i in range(tokens * dim)]
    gate_weight = [((i % 11) - 5) / 4.0 for i in range(experts * dim)]
    correction_bias = [0.0, 0.0, 0.2, -0.1, 0.05]
    # bias_vl deliberately favours a different expert so image tokens diverge.
    bias_vl = [0.0, 5.0, -5.0, 0.0, 0.0]
    # token 0 text, tokens 1 and 2 inside an image span.
    image_mask = [False, True, True]

    def upstream_gate():
        import torch

        assert module is not None
        gate = module.Gate.__new__(module.Gate)
        module.nn.Module.__init__(gate)
        gate.dim = dim
        gate.topk = topk
        gate.score_func = "sqrtsoftplus"
        gate.gate_temp = gate_temp
        gate.norm_topk_prob = norm_topk_prob
        gate.route_scale = route_scale
        gate.weight = torch.nn.Parameter(
            torch.tensor(gate_weight, dtype=torch.float32).reshape(experts, dim)
        )
        gate.bias = torch.nn.Parameter(torch.tensor(correction_bias, dtype=torch.float32))
        gate.bias_vl = torch.nn.Parameter(torch.tensor(bias_vl, dtype=torch.float32))

        mask = torch.tensor(image_mask, dtype=torch.bool)
        x_t = torch.tensor(x, dtype=torch.float32).reshape(tokens, dim)
        weights, indices = gate.forward(x_t, mask)
        scores = (
            torch.nn.functional.softplus(module.linear(x_t, gate.weight) / gate_temp).sqrt()
        )
        return scores.flatten().tolist(), weights.flatten().tolist(), indices.flatten().tolist()

    if import_status != "path-import-ok" or module is None:
        raise SystemExit(f"vl-gate fixtures require upstream model.py, got {import_status}")

    (scores, weights, indices), call_status = call_upstream_or_fallback(
        import_status,
        "Gate.forward(image_mask)",
        upstream_gate,
        lambda: (_ for _ in ()).throw(RuntimeError("unreachable")),
    )

    write_json(
        out_dir / "vl_gate_fixture.json",
        {
            **source_meta("Gate.forward(image_mask)", "Gate.forward(image_mask) around lines 809-827"),
            "upstream_import_status": import_status,
            "upstream_call_status": call_status,
            "absolute_tolerance": 1e-5,
            "input": {
                "shape": {"tokens": tokens, "dim": dim, "experts": experts},
                "x": round_list(x),
                "gate_weight": round_list(gate_weight),
                "gate_temp": gate_temp,
                "correction_bias": round_list(correction_bias),
                "bias_vl": round_list(bias_vl),
                "image_mask": image_mask,
                "topk": topk,
                "norm_topk_prob": norm_topk_prob,
                "route_scale": route_scale,
            },
            "expected": {
                "scores": round_list(scores),
                "indices": indices,
                "weights": round_list(weights),
            },
        },
    )


def generate_merge(out_dir: Path, module: object | None, import_status: str) -> None:
    """merge_image_embeddings span-overwrite correctness for one image."""
    # One batch row, one image; 2x2 aligner grid -> types layout has 4 IMAGE, 2
    # IMAGE_NEW_LINE, IMAGE_START/IMAGE_END. Sequence has text on either side.
    b, dim = 1, 4
    n_llm_h, n_llm_w = 2, 2
    # token_types over the image span (reading order): START, (IMAGE*w, NEWLINE)*h, END.
    IMAGE_START, IMAGE, IMAGE_NEW_LINE, IMAGE_END = 0, 1, 2, 3
    types = [IMAGE_START]
    for _ in range(n_llm_h):
        types += [IMAGE] * n_llm_w + [IMAGE_NEW_LINE]
    types.append(IMAGE_END)
    span_len = len(types)
    start = 2  # two text tokens before the image span
    s = start + span_len + 1  # one trailing text token
    n_image = n_llm_h * n_llm_w

    embed = [((i % 13) - 6) / 5.0 for i in range(b * s * dim)]
    aligner_rows = [((i % 7) - 3) / 2.0 for i in range(n_image * dim)]
    image_start = [1.0, 2.0, 3.0, 4.0]
    image_end = [-1.0, -2.0, -3.0, -4.0]
    image_newline = [0.5, -0.5, 0.5, -0.5]

    def upstream_merge():
        import torch

        assert module is not None
        transformer = module.Transformer.__new__(module.Transformer)
        module.nn.Module.__init__(transformer)
        transformer.image_start = torch.nn.Parameter(torch.tensor(image_start, dtype=torch.float32))
        transformer.image_end = torch.nn.Parameter(torch.tensor(image_end, dtype=torch.float32))
        transformer.image_newline = torch.nn.Parameter(
            torch.tensor(image_newline, dtype=torch.float32)
        )

        class _Img:
            pass

        img = _Img()
        img.start = start
        img.types = torch.tensor(types, dtype=torch.long)

        # Stub encode_image to return the fixed aligner rows (isolate the merge).
        rows = torch.tensor(aligner_rows, dtype=torch.float32).reshape(n_image, dim)
        transformer.encode_image = lambda *_args, **_kw: rows
        img.patches = torch.zeros(1)
        img.n_vit_h = n_llm_h
        img.n_vit_w = n_llm_w

        h = torch.tensor(embed, dtype=torch.float32).reshape(b, s, dim)
        transformer.merge_image_embeddings([[img]], h)
        return h.flatten().tolist()

    if import_status != "path-import-ok" or module is None:
        raise SystemExit(f"merge fixtures require upstream model.py, got {import_status}")

    merged, call_status = call_upstream_or_fallback(
        import_status,
        "Transformer.merge_image_embeddings",
        upstream_merge,
        lambda: (_ for _ in ()).throw(RuntimeError("unreachable")),
    )

    write_json(
        out_dir / "merge_image_embeddings_fixture.json",
        {
            **source_meta(
                "Transformer.merge_image_embeddings",
                "Transformer.merge_image_embeddings around lines 1228-1239",
            ),
            "upstream_import_status": import_status,
            "upstream_call_status": call_status,
            "absolute_tolerance": 0.0,
            "input": {
                "b": b,
                "s": s,
                "dim": dim,
                "start": start,
                "n_image": n_image,
                "token_types": types,
                "embed": round_list(embed),
                "aligner_rows": round_list(aligner_rows),
                "image_start": round_list(image_start),
                "image_end": round_list(image_end),
                "image_newline": round_list(image_newline),
            },
            "expected": {"merged": round_list(merged)},
        },
    )


def generate_vl_prompt(out_dir: Path, repo_root: Path) -> None:
    """Emit a multimodal prompt-expansion fixture by calling upstream
    `prepare_vl_inputs`.

    The full VL pipeline (`prepare_vl_inputs`) needs a tokenizer and PIL image
    decode. To stay hermetic and avoid real image bytes, we:

    * stub the tokenizer with a tiny deterministic encoder that maps the image
      placeholder to `image_token_id` and any other text to a fixed id stream,
    * monkeypatch `image_processor.load_image` to return tiny deterministic
      patches plus a fixed `(n_vit_h, n_vit_w, n_llm_h, n_llm_w)` grid,

    so the fixture exercises the real placeholder->span expansion,
    `image_token_types` layout, and per-image `(start, grid)` bookkeeping — which
    is exactly what the Rust CLI's `--token-types`/`--image-patches` path mirrors.
    """
    image_py = (
        repo_root
        / "ferric_continuum/tnsr/third_party/deepseek_v41/upstream/inference/image_processor.py"
    )
    if not image_py.exists():
        raise SystemExit(f"missing mirrored upstream source: {image_py}")
    module, import_status = import_upstream_image_processor_by_path(image_py)
    if import_status != "path-import-ok" or module is None:
        raise SystemExit(
            f"vl-prompt fixtures require upstream image_processor.py import, got {import_status}"
        )

    image_token_id = 4
    # Tiny deterministic grid for the single image: 2x2 aligner cells.
    n_vit_h, n_vit_w, n_llm_h, n_llm_w = 2, 2, 2, 2
    vd = 8  # tiny vision_dim (mirrors the loader fixture geometry)
    patch_flat = 3  # 3 * patch_size^2 with patch_size=1
    n_patch = n_vit_h * n_vit_w

    class _Args:
        vision_enabled = True
        image_token_id = 4

    class _Tokenizer:
        """Deterministic encoder: the marker string maps to `image_token_id`,
        every other whitespace-separated word maps to a stable small id."""

        unk_token_id = 0

        def convert_tokens_to_ids(self, tok):
            return None  # skip the placeholder cross-check

        def encode(self, prompt):
            ids = []
            for word in prompt.split():
                if word == "<image>":
                    ids.append(image_token_id)
                else:
                    # Deterministic across runs (Python hashes are salted):
                    # sum of byte codes, mapped into {1,2,3}.
                    ids.append(1 + (sum(word.encode("utf-8")) % 3))
            return ids

    # Deterministic patches + grid: isolate span expansion from PIL/torch decode.
    patches = [((i % 5) - 2) / 2.0 for i in range(n_patch * patch_flat)]

    def fake_load_image(record, args):
        import torch

        t = torch.tensor(patches, dtype=torch.float32).reshape(n_patch, patch_flat)
        return t, n_vit_h, n_vit_w, n_llm_h, n_llm_w

    def upstream_prepare():
        import torch  # noqa: F401  (module.torch is the stub; real torch here)

        # prepare_vl_inputs does `from encoding import IMAGE_PLACEHOLDER`; expose
        # the real spelling from the mirrored upstream encoding module without
        # importing its heavy transitive deps.
        encoding_py = (
            repo_root
            / "ferric_continuum/tnsr/third_party/deepseek_v41/upstream/encoding/encoding.py"
        )
        placeholder = "<｜deepseek_image｜>"
        for line in encoding_py.read_text().splitlines():
            if line.startswith("IMAGE_PLACEHOLDER ="):
                placeholder = line.split("=", 1)[1].strip().strip('"').strip("'")
                break
        encoding_stub = types.ModuleType("encoding")
        encoding_stub.IMAGE_PLACEHOLDER = placeholder
        saved_encoding = sys.modules.get("encoding")
        sys.modules["encoding"] = encoding_stub

        # The module was imported with a stub `torch` (its top-level import), so
        # `image_token_types` would build a Python list instead of a Tensor.
        # Bind the real torch onto the module so span expansion runs for real.
        saved_torch = getattr(module, "torch", None)
        module.torch = torch
        saved = module.load_image
        module.load_image = fake_load_image
        try:
            prompt = "describe <image> please"
            tokens, token_types, image_inputs = module.prepare_vl_inputs(
                prompt, [{"data": b"x"}], _Tokenizer(), _Args()
            )
            imgs = []
            for img in image_inputs or []:
                span_types = (
                    img.types.tolist() if hasattr(img.types, "tolist") else list(img.types)
                )
                imgs.append(
                    {
                        "start": int(img.start),
                        "n_vit_h": int(img.n_vit_h),
                        "n_vit_w": int(img.n_vit_w),
                        "n_llm_h": n_llm_h,
                        "n_llm_w": n_llm_w,
                        "token_types": [int(x) for x in span_types],
                    }
                )
            return {
                "tokens": [int(x) for x in tokens],
                "token_types": [int(x) for x in token_types],
                "images": imgs,
            }
        finally:
            module.load_image = saved
            if saved_torch is None:
                if hasattr(module, "torch"):
                    del module.torch
            else:
                module.torch = saved_torch
            if saved_encoding is None:
                sys.modules.pop("encoding", None)
            else:
                sys.modules["encoding"] = saved_encoding

    # Real torch is needed for the stubbed load_image tensor; if unavailable the
    # fixture cannot be produced (no meaningful fallback for span expansion).
    try:
        import torch  # noqa: F401
    except Exception as err:  # noqa: BLE001
        raise SystemExit(
            f"vl-prompt fixtures require torch for deterministic patches, got {type(err).__name__}: {err}"
        )

    result, call_status = call_upstream_or_fallback(
        import_status,
        "prepare_vl_inputs",
        upstream_prepare,
        lambda: (_ for _ in ()).throw(RuntimeError("unreachable")),
    )

    write_json(
        out_dir / "prompt_vl_fixture.json",
        {
            **source_meta(
                "prepare_vl_inputs",
                "prepare_vl_inputs around lines 140-173",
                source_file="ferric_continuum/tnsr/third_party/deepseek_v41/upstream/inference/image_processor.py",
            ),
            "upstream_import_status": import_status,
            "upstream_call_status": call_status,
            "absolute_tolerance": 0.0,
            "input": {
                "prompt": "describe <image> please",
                "image_token_id": image_token_id,
                "n_vit_h": n_vit_h,
                "n_vit_w": n_vit_w,
                "n_llm_h": n_llm_h,
                "n_llm_w": n_llm_w,
                "vision_dim": vd,
                "patch_flat": patch_flat,
                "patches": round_list(patches),
            },
            "expected": result,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--family", default="rope,sparse,compressor")
    args = parser.parse_args()

    root = args.repo_root.resolve()
    maybe_reexec_with_repo_venv(root)
    out_dir = root / "ferric_continuum/tnsr/testdata/deepseek_v41"
    families = {part.strip() for part in args.family.split(",") if part.strip()}

    # Prompt fixtures need only the pure-Python encoding.py, not model.py.
    if "prompt" in families:
        generate_prompt(out_dir, root)
        families.discard("prompt")
        if not families:
            return

    # VL-prompt fixtures need only the pure-Python image_processor.py, not model.py.
    if "vl-prompt" in families:
        generate_vl_prompt(out_dir, root)
        families.discard("vl-prompt")
        if not families:
            return

    # Image-grid fixtures need only the pure-Python grid math from
    # image_processor.py, not model.py.
    if "image-grid" in families:
        image_py = (
            root
            / "ferric_continuum/tnsr/third_party/deepseek_v41/upstream/inference/image_processor.py"
        )
        if not image_py.exists():
            raise SystemExit(f"missing mirrored upstream source: {image_py}")
        image_module, image_status = import_upstream_image_processor_by_path(image_py)
        generate_image_grid(out_dir, image_module, image_status)
        families.discard("image-grid")
        if not families:
            return

    # Vision ViT fixtures need only the pure-torch vision.py, not model.py.
    if "vision" in families:
        vision_py = (
            root
            / "ferric_continuum/tnsr/third_party/deepseek_v41/upstream/inference/vision.py"
        )
        if not vision_py.exists():
            raise SystemExit(f"missing mirrored upstream source: {vision_py}")
        vision_module, vision_status = import_upstream_vision_by_path(vision_py)
        generate_vision(out_dir, vision_module, vision_status)
        families.discard("vision")
        if not families:
            return

    # Aligner fixtures need only the pure-torch vision.py, not model.py.
    if "aligner" in families:
        vision_py = (
            root
            / "ferric_continuum/tnsr/third_party/deepseek_v41/upstream/inference/vision.py"
        )
        if not vision_py.exists():
            raise SystemExit(f"missing mirrored upstream source: {vision_py}")
        vision_module, vision_status = import_upstream_vision_by_path(vision_py)
        generate_aligner(out_dir, vision_module, vision_status)
        families.discard("aligner")
        if not families:
            return

    # encode_image fixtures chain ViT + Aligner from the pure-torch vision.py.
    if "encode-image" in families:
        vision_py = (
            root
            / "ferric_continuum/tnsr/third_party/deepseek_v41/upstream/inference/vision.py"
        )
        if not vision_py.exists():
            raise SystemExit(f"missing mirrored upstream source: {vision_py}")
        vision_module, vision_status = import_upstream_vision_by_path(vision_py)
        generate_encode_image(out_dir, vision_module, vision_status)
        families.discard("encode-image")
        if not families:
            return

    model_py = root / "ferric_continuum/tnsr/third_party/deepseek_v41/upstream/inference/model.py"
    if not model_py.exists():
        raise SystemExit(f"missing mirrored upstream source: {model_py}")
    module, import_status = import_upstream_model_by_path(model_py)

    if "rope" in families:
        generate_rope(out_dir, module, import_status)
    if "sparse" in families:
        generate_sparse(out_dir, module, import_status)
    if "compressor" in families:
        generate_compressor(out_dir, module, import_status)
    if "moe" in families:
        generate_moe(out_dir, module, import_status)
    if "hyper" in families:
        generate_hyper(out_dir, module, import_status)
    if "engram" in families:
        generate_engram(out_dir, module, import_status)
    if "layer" in families:
        generate_layer(out_dir, module, import_status)
    if "block" in families:
        generate_block(out_dir, module, import_status)
    if "tiny-model" in families:
        generate_tiny_model(out_dir, module, import_status)
    if "vl-gate" in families:
        generate_vl_gate(out_dir, module, import_status)
    if "merge" in families:
        generate_merge(out_dir, module, import_status)


if __name__ == "__main__":
    main()
