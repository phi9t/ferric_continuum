# DeepSeek-V4.1-Flash Upstream Architecture Notes

Date: 2026-09-10

Scope: read-only research for a possible `tnsr` Rust loader and forward path.
Primary upstream: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/main.

## Upstream Files Read

- `README.md`: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/README.md
- `DeepSeek_V41_Tech_Report.pdf`: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf
- `config.json`: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/config.json
- `tokenizer_config.json`: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/tokenizer_config.json
- `model.safetensors.index.json`: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/model.safetensors.index.json
- `inference/README.md`: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/README.md
- `inference/config.json`: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/config.json
- `inference/model.py`: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/model.py
- `inference/generate.py`: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/generate.py
- `inference/convert.py`: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/convert.py
- `inference/kernel.py`: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/kernel.py
- `inference/engram.py`: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/engram.py
- `inference/vision.py`: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/vision.py
- `inference/image_processor.py`: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/image_processor.py
- `encoding/README.md`: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/encoding/README.md
- `encoding/encoding.py`: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/encoding/encoding.py

Local line-numbered copies used for citations live under
`ferric_continuum/tnsr/third_party/deepseek_v41/upstream/`. I did not download the 48 large model
weight shards; I used the upstream tree API and `model.safetensors.index.json`.

## Architecture And Config

The HF config declares `architectures: ["DeepseekV41ForCausalLM"]`,
`model_type: "deepseek_v41"`, top-level `dtype: "bfloat16"`, token ids
`bos=0`, `eos=1`, `pad=2`, `image_token_id=129264`, and an FP8 quantization
config with dynamic activations, `[32, 32]` weight blocks, `ue8m0` scale format,
and FP4 experts (`config.json:2-20`).

The text stack is `deepseek_v41_text`: `vocab_size=129280`, `hidden_size=5120`,
`moe_intermediate_size=2304`, `num_hidden_layers=40`,
`num_attention_heads=64`, `num_key_value_heads=1`, `head_dim=512`,
`qk_rope_head_dim=64`, `q_lora_rank=1280`, `o_lora_rank=1024`,
`o_groups=8`, `silu`, `swiglu_limit=10.0`, `rms_norm_eps=1e-20`, no attention
bias/dropout, untied embeddings, `max_position_embeddings=1048576`,
`rope_theta=10000`, and YaRN scaling factor 16 from original 65536 tokens
(`config.json:22-51`).

The MoE fields are `n_routed_experts=384`, `n_shared_experts=1`,
`num_experts_per_tok=6`, `scoring_func="sqrtsoftplus"`,
`topk_method="noaux_tc"`, `norm_topk_prob=true`, and
`routed_scaling_factor=1.5` (`config.json:52-58`). The release README describes
the model as a multimodal MoE with 552B backbone parameters, CED organization as
20 causal-encoder layers followed by 20 decoder layers, 8B active parameters per
token during prefill, and 16B during decode (`README.md:45-51`). The extracted
tech report also states 552B backbone plus 196B Engram parameters
(`DeepSeek_V41_Tech_Report.txt:557-564`).

The sparse-attention config includes `sliding_window=128`,
`compress_ratios` for 43 layer slots, `compress_rope_theta=160000`,
`kv_source_layer_ids=[2,8,14,20]`,
`index_source_layer_ids=[2,8,14,20,24,28,32,36]`,
`index_n_heads=32`, `index_head_dim=128`, `index_topk=512`,
`candidate_source_layer_id=20`, `candidate_topk_blocks=2048`,
`candidate_block_size=8`, and hyper-connection fields `hc_mult=4`,
`hc_sinkhorn_iters=20`, `hc_eps=1e-6` (`config.json:59-130`).

The model also includes Engram, DSpark, and vision fields: Engram layers
`[1,14]`, table row counts `384006168` and `384016682`, max n-gram size 4,
compressed vocab size 99092; three next-token prediction layers; DSpark target
layers `[37,38,39]`; and a vision tower with 32 layers, hidden size 1024,
16 heads, patch size 14, 3x downsample, and up to 1024 image tokens
(`config.json:131-169`).

The reference inference implementation has a separate `ModelArgs` dataclass
whose field names match `inference/config.json`, not the nested HF
`text_config` names. The large-release values in `inference/config.json` mirror
the HF text and vision fields (`inference/config.json:1-65`), while the
dataclass defaults are intentionally a small runnable self-test model
(`inference/model.py:44-137`).

## Attention, MLA, CSA2, And KV

The released code does not define a class named `MLA`; it implements latent
attention directly in `Attention`. The path is:

- `wq_a: dim -> q_lora_rank`, `q_norm`, then column-parallel `wq_b:
  q_lora_rank -> n_heads * head_dim` (`inference/model.py:639-643`,
  `inference/model.py:770-772`).
- `wkv: dim -> head_dim`, then `kv_norm`; this is one latent KV vector per
  token, with only the last `rope_head_dim` rotated (`inference/model.py:643-645`,
  `inference/model.py:700-707`).
- `wo_a` is grouped and low-rank, then `wo_b` maps back to `dim`; the forward
  path uses an explicit grouped `einsum` for `wo_a`
  (`inference/model.py:645-650`, `inference/model.py:780-789`).
- An `attn_sink` parameter is passed into `sparse_attn` with the softmax scale
  `head_dim ** -0.5` (`inference/model.py:639-651`, `inference/model.py:780`).

RoPE is interleaved in the text reference implementation: `apply_rotary_emb`
views adjacent element pairs as complex values (`inference/model.py:392-406`).
With compression enabled, the text attention uses YaRN over
`compress_rope_theta`; with pure sliding-window attention it disables YaRN and
uses base `rope_theta` (`inference/model.py:680-698`).

CSA2 combines a sliding-window ring cache with optional compressed KV. Window
indices are causal and ring-buffered over `window_size` (`inference/model.py:409-426`,
`inference/model.py:700-720`). For compressed KV, source layers pool
`compress_ratio` tokens through `Compressor`; ratio 1 is a plain projection,
while higher ratios use a learned softmax gate and state for partial decode
groups (`inference/model.py:429-485`). A compressed layer concatenates window KV
and compressed KV, then calls `sparse_attn` once with concatenated top-k indices
(`inference/model.py:765-780`).

The indexer is a side attention over compressed positions. KV-source layers can
project compressed latents into index keys; index-source layers compute Top-K
compressed positions; later layers can reuse the published `shared_attn`
indices. The candidate source layer creates a block-level candidate pool, and
later indexers mask to that pool (`inference/model.py:488-580`,
`inference/model.py:583-610`, `inference/model.py:1166-1180`). This matches the
README description of CSA2 Full/Reindex/Reuse modes and hierarchical sparse
indexing (`README.md:49-51`) and the tech report's statements that CSA2 shares
main KV, indexer K, and Top-K indices across layers (`DeepSeek_V41_Tech_Report.txt:767-785`).

The DeepSeek V4.1 attention note should not be read as a DeepSeek V4 runtime
contract. The report contrasts V4's `CSA-HCA` hybrid with V4.1 Flash's pure
CSA2 design (`DeepSeek_V41_Tech_Report.txt:739-742`): this codebase preserves
that contrast by implementing V4.1's Full/Reindex/Reuse sharing semantics, not
older overlapping CSA source-entry or absolute-position compressor assumptions.

The FP4 main-KV cache is a runtime quantization path, not just checkpoint
storage. The reference says compressed KV uses groups of 16 with E4M3 scales,
while the indexer uses groups of 32 with E8M0 scales (`inference/model.py:758-760`).
`kernel.py` implements FP8 activation quantization and FP4 activation
quantization with E8M0 or E4M3 scales (`inference/kernel.py:40-124`,
`inference/kernel.py:127-204`). The tech report states the main KV cache uses
FP4 with one E4M3 scale per 16 channels and quantizes both non-RoPE and RoPE
components (`DeepSeek_V41_Tech_Report.txt:998-1024`).

## MoE, Hyper-Connections, Engram, DSpark

MoE routing uses a learned gate weight plus correction bias. `sqrtsoftplus`
means `sqrt(softplus(linear(x) / gate_temp))`; expert selection uses
`scores + bias`, while actual routing weights are gathered from the unbiased
scores, normalized across Top-K with `+1e-20`, and multiplied by route scale
(`inference/model.py:792-827`). Vision/image tokens may use `bias_vl`
(`inference/model.py:806-820`).

Each routed expert and the shared expert are SwiGLU MLPs with `w1`, `w3`, and
`w2`; `w3` is clamped to `[-swiglu_limit, swiglu_limit]`, and `w1` is clamped
only above before `silu(w1) * w3` (`inference/model.py:830-851`). The MoE has
one shared expert and shard-local routed experts; outputs are all-reduced across
tensor-parallel ranks before adding the shared expert (`inference/model.py:854-904`).

Hyper-Connections carry `hc_mult` parallel residual copies. Each block has
learned `hc_attn_fn/base/scale` and `hc_ffn_fn/base/scale`; `hc_mixes` projects
the flattened residual stream, normalizes it, and applies a Sinkhorn split into
pre, post, and combination coefficients (`inference/model.py:907-995`).

Engram computes hashed n-gram ids from a tokenizer-compressed id map. The
compressed-token map normalizes decoded tokens with NFKC, accent stripping,
lowercasing, whitespace normalization, and a sentinel for a single-space token
(`inference/engram.py:17-61`). The layout uses disjoint prime-sized buckets for
each layer, n-gram size, and head (`inference/engram.py:86-126`). Runtime hashing
caches compressed token ids across prefill/decode and prevents n-grams from
crossing image spans (`inference/engram.py:129-184`). The Engram module looks up
FP8 rows, projects to per-HC keys and a value, and gates updates into the
residual with a normalized signed-square-root dot product
(`inference/model.py:296-365`).

DSpark appears under the `mtp.*` checkpoint namespace and is implemented as
three extra draft layers. The reference forward can seed DSpark caches during
prefill, generate a draft block during decode, apply a Markov head and a
confidence head, but `generate.py` does not call `forward_spec`
(`inference/model.py:1020-1156`, `inference/model.py:1274-1282`). The
dataclass comment explicitly says the speculative-decoding loop is out of scope
for the reference repo (`inference/model.py:129-136`).

## Tokenizer And Generation Quirks

The model card says there is no Jinja chat template; prompt encoding is defined
by `encoding/encoding.py`, with a Rust/Python `deepseek-recipe` toolkit
recommended for production (`README.md:158-162`). Tokenizer config disables
automatic BOS/EOS insertion even though the format encoder prepends BOS itself:
`add_bos_token=false`, `add_eos_token=false`, `model_max_length=1048576`;
the pad token content is the EOS string (`tokenizer_config.json:1-33`).

Important prompt-format details:

- Special strings include `<｜begin▁of▁sentence｜>`,
  `<｜end▁of▁sentence｜>`, `<｜User｜>`, `<｜Assistant｜>`,
  `<｜latest_reminder｜>`, `<think>`, `</think>`, `｜DSML｜`, and
  `<｜deepseek_image｜>` (`encoding/encoding.py:29-49`,
  `encoding/README.md:107-120`).
- V4.1 changed DSML tool-call tags to include a leading space in tag names,
  e.g. `<｜DSML｜ calls>`, `<｜DSML｜ invoke>`, and
  `<｜DSML｜ parameter>` (`encoding/README.md:8-27`,
  `encoding/encoding.py:8-17`).
- Reasoning effort is numeric 1-100, with `low=50`, `high=75`, `max=100`;
  it is rendered only for `thinking_mode="thinking"` at the start of the
  conversation (`encoding/README.md:17-23`, `encoding/README.md:260-266`).
- In chat mode, `</think>` is appended immediately after `<|Assistant|>` so the
  model generates normal content (`encoding/README.md:141-149`,
  `encoding/encoding.py:720-729`).
- `encode_messages` merges tool messages, sorts tool results by assistant
  call order, optionally drops older reasoning, and prepends BOS only when no
  context is supplied (`encoding/encoding.py:759-833`).
- The parser expects well-formed model output with EOS unless tool parsing
  consumes a valid tool block; it intentionally does not recover malformed
  output (`encoding/encoding.py:935-974`).

Generation is plain autoregressive sampling. The first forward pass processes
up to the shortest prompt length, subsequent calls decode one token at a time,
and prompt tokens override model predictions while prompts are still being
consumed (`inference/generate.py:28-87`). `sample` uses argmax when
`temperature == 0`; otherwise it uses Gumbel-max after softmax
(`inference/model.py:1285-1292`). The model card recommends
`temperature=1.0`, `top_p=0.95 or 1.0`, 1M context, and `max_tokens >= 256K`,
but the reference `generate.py` does not implement top-p filtering
(`README.md:168-175`, `inference/generate.py:28-87`).

Vision prompts require expanding each `<｜deepseek_image｜>` placeholder to an
image token span. Every image-span position carries `image_token_id`; a
separate token-type vector distinguishes start, image patches, row separators,
and end (`inference/image_processor.py:1-10`, `inference/image_processor.py:140-173`).
The Transformer only accepts image spans during the prefill chunk
(`inference/model.py:1241-1256`, `inference/generate.py:55-63`).

## Weight Naming And Layout Notes For `tnsr`

The released checkpoint has 48 safetensor shards, 96,085 tensors, and
`metadata.total_size=510286023000` in `model.safetensors.index.json`. The
downloaded tree lists shards `model-00001-of-00048.safetensors` through
`model-00048-of-00048.safetensors`; the largest Engram shards are roughly
101.5 GB each. The index maps by already-converted-style names such as
`embed.weight`, `layers.N.attn.*`, `layers.N.ffn.*`, `mtp.N.*`, `head.weight`,
`norm.weight`, `vision.*`, and `layers.{1,14}.engram.*`.

This namespace does not match the existing `tnsr` Qwen3 loader, which expects
Qwen-style names like `model.embed_tokens.weight`,
`model.layers.N.self_attn.q_proj.weight`, `k_proj`, `v_proj`, `o_proj`,
`q_norm`, `k_norm`, and dense `mlp.gate_proj/up_proj/down_proj`
(`ferric_continuum/tnsr/src/qwen3_load.rs:1-35`,
`ferric_continuum/tnsr/src/qwen3_load.rs:256-389`). DeepSeek V4.1 uses
`attn.wq_a`, `attn.wq_b`, `attn.wkv`, `attn.kv_norm`, grouped `wo_a/wo_b`,
`attn_sink`, CSA2 compressor/indexer tensors, MoE `ffn.gate`, routed
`ffn.experts.{i}.w{1,2,3}`, and `ffn.shared_experts.w{1,2,3}`.

Upstream `convert.py` documents the intended tensor-parallel sharding:
`embed`, `wq_b`, `wo_a`, `head`, `attn_sink`, and `weights_proj` are split on
dimension 0; `wo_b` is split on dimension 1; routed experts are assigned by
expert id; Engram embedding rows are sharded with padding; MTP tied
`embed.weight` and `head.weight` copies are skipped (`inference/convert.py:56-64`,
`inference/convert.py:80-146`). `wo_a.weight` is dequantized to bf16 because
the reference uses grouped `einsum` instead of an fp8 grouped GEMM
(`inference/convert.py:154-173`). FP4 expert weights can either stay packed
as `torch.float4_e2m1fn_x2` or be converted losslessly to FP8 depending on
`--expert-dtype` (`inference/convert.py:174-182`).

The linear layout in upstream Python is PyTorch-style `[out, in]`; the local
`tnsr` Qwen3 loader already documents that `tnsr` stores linear parameters as
`[Din, Dout]` and transposes HF `nn.Linear` weights on load
(`ferric_continuum/tnsr/src/qwen3_load.rs:9-13`). That transpose convention
should still apply to DeepSeek V4.1 `Linear` weights unless a loader decides to
store them in an explicit quantized `[out, in]` layout for kernel parity.

Do not reuse the Qwen3 RoPE adapter mechanically. Local `tnsr` Qwen3 had to
permute Q/K output columns and Q/K norm gammas because HF Qwen3 used half-split
RoPE while `tnsr` used interleaved RoPE
(`ferric_continuum/tnsr/src/qwen3_load.rs:15-32`). DeepSeek V4.1's reference
text path already uses adjacent-pair complex RoPE in `inference/model.py`
(`inference/model.py:392-406`), so the Rust implementation should first match
that interleaved convention and only permute if the actual Transformers class
later proves a different convention.

Quantization is a first-class loader concern. Standard linears can have
`.weight` plus `.scale`; FP8 weights use scale tensors shaped by 32x32 blocks
(`inference/model.py:210-235`). FP4 expert weights are logically `[out, in]`
but stored with two values per byte as `[out, in//2]`, with scales
`[out, in/32]` (`inference/model.py:219-224`). The existing `tnsr` loader only
decodes bf16/f32 safetensors for Qwen3 (`ferric_continuum/tnsr/src/qwen3_load.rs:170-201`),
so DeepSeek V4.1 needs either quantized tensor support or a deliberate
dequantizing conversion path before numeric forward parity is realistic.

## Verifier Recommendations

1. Start with source-of-truth metadata tests: parse nested `config.json` and
   flat `inference/config.json`; assert the exact architecture, text, vision,
   CSA2, MoE, Engram, DSpark, and quantization fields above. This should fail
   loudly if `architectures`, `model_type`, `vocab_size`, `rms_norm_eps`, or
   `compress_ratios` drift.

2. Add an index-only loader verifier before loading full weights. Read
   `model.safetensors.index.json`; assert shard count 48, tensor count 96085,
   `total_size=510286023000`, presence of top-level `embed.weight`,
   `head.weight`, `norm.weight`, representative `layers.0.attn.wq_a.weight`,
   `layers.2.attn.compressor.*`, `layers.2.attn.indexer.*`,
   `layers.1.engram.*`, `mtp.0.*`, `vision.*`, and absence of Qwen-style
   `model.layers.0.self_attn.q_proj.weight`.

3. Port upstream small-model tests before real-weight tests. The official
   `python model.py` self-test uses small `ModelArgs` defaults and uninitialized
   weights to exercise shape and kernel plumbing (`inference/README.md:63-71`,
   `inference/model.py:1295-1309`). A Rust equivalent should run without large
   weights and verify block shapes, cache lengths, compress-ratio behavior,
   MoE route shapes, and DSpark prefill/decode API shape.

4. Build focused golden tests from upstream Python functions for tricky math:
   `precompute_freqs_cis` and `apply_rotary_emb`, `get_window_topk_idxs`,
   `Compressor.forward`, `select_candidate_blocks`, `Gate.forward`,
   `build_compressed_token_map`, `EngramLayout.from_args`, and
   `NgramHashState.forward`. These are small enough to compare against JSON
   fixtures without full model execution.

5. Treat tokenizer verification as separate from model math. Reuse upstream
   `encoding/tests/*` as prompt-string fixtures, then compare token ids through
   `AutoTokenizer` or the first-party `deepseek-recipe` Rust library. The
   current `tnsr` Qwen3 path already supports `--token-ids` to isolate tokenizer
   drift from model-math drift (`ferric_continuum/tnsr/src/bin/qwen3_infer.rs:220-242`);
   keep that pattern.

6. For numeric parity, stage the ladder: one-layer tiny random parity against
   the upstream reference first, then converted-per-rank checkpoint parity on a
   short text-only prompt, then full prompt encoding, then image and Engram
   parity. Full released-weight parity will need host-side assets because the
   checkpoint is hundreds of GB and the existing Bazel sandbox has no network
   access. Existing project memory for Qwen3 also says to prefer native Rust
   safetensors loading and logits-based Hugging Face compatibility checks.

7. Make unsupported surfaces explicit in early Rust binaries. A text-only first
   path should reject image placeholders, DSpark speculative mode, and Engram
   unless each is intentionally implemented. Silent fallbacks would produce
   plausible but non-equivalent logits.
