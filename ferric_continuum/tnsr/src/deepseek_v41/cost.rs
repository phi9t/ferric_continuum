//! DeepSeek V4.1 forward cost accounting: parameter counts, forward FLOPs, and
//! weight/activation byte estimates derived directly from
//! [`DeepSeekV41TextConfig`].
//!
//! This is the Wave-4 analytical seam.  It is deliberately *symbolic* — it
//! counts multiply-accumulates as 2 FLOPs and f32 activation bytes, matching the
//! `scaling` module's convention — and does not touch the executed forward path
//! or any device.  It exists so a caller (a CLI, a report, a roofline lookup)
//! can size the model without loading a 510GB checkpoint.
//!
//! # Model shape recap
//!
//! DeepSeek V4.1 uses Multi-head Latent Attention (MLA): the query path is a
//! low-rank `wq_a: [D, q_lora]` then `wq_b: [q_lora, H*head_dim]`; the KV path is
//! a single shared `wkv: [D, head_dim]` (`num_key_value_heads == 1`); the output
//! path is grouped `wo_a` (`[H*head_dim, o_groups*o_lora]`) then
//! `wo_b: [o_groups*o_lora, D]`.  The FFN is a Mixture-of-Experts: a router gate
//! `[D, n_routed_experts]`, `num_experts_per_tok` active routed experts plus
//! `n_shared_experts` always-on experts, each a 3-matrix SwiGLU
//! (`w1,w3: [D, moe_inter]`, `w2: [moe_inter, D]`).
//!
//! Only matmul FLOPs are counted; norms/RoPE/softmax/HyperConnection mixing are
//! elementwise and dominate bandwidth, not FLOPs, so they are folded into the
//! activation-byte estimate rather than the FLOP total.

use super::config::DeepSeekV41TextConfig;

/// Bytes per f32 element (matches `scaling::F32_BYTES`).
const F32_BYTES: u64 = 4;

/// Per-block parameter counts (matmul weights only; elementwise norm/HC scalars
/// are excluded from the "N" used by the `6·N·T` training-FLOPs rule).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockParams {
    /// MLA query path: `wq_a` + `wq_b`.
    pub attn_q: u64,
    /// MLA shared KV projection `wkv` (`num_key_value_heads == 1`).
    pub attn_kv: u64,
    /// MLA grouped output path: `wo_a` + `wo_b`.
    pub attn_o: u64,
    /// Router gate `[D, n_routed_experts]`.
    pub moe_gate: u64,
    /// All routed experts' `w1+w2+w3` (the full bank, not just the active ones).
    pub moe_routed_experts: u64,
    /// Shared experts' `w1+w2+w3`.
    pub moe_shared_experts: u64,
    /// Sum of the above.
    pub total: u64,
}

/// Per-block forward FLOPs for a `[batch, seq]` workload, and the activation
/// bytes the block materialises.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockForward {
    /// Attention projection FLOPs (q/kv/o linear maps).
    pub attn_proj_flops: u64,
    /// Scaled-dot-product attention FLOPs (scores + mix), dense upper bound.
    pub attn_sdpa_flops: u64,
    /// MoE FLOPs: gate + `num_experts_per_tok` routed + `n_shared_experts`
    /// shared experts, per token.
    pub moe_flops: u64,
    /// Sum of the above.
    pub total_flops: u64,
    /// f32 activation bytes materialised by the block (order-of-magnitude).
    pub act_bytes: u64,
}

/// Whole-model forward cost: embedding + `num_hidden_layers` blocks + final norm
/// + LM head, plus the aggregate parameter count.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelForward {
    /// Total matmul parameters across every block plus the LM head + embedding.
    pub params_total: u64,
    /// Per-block matmul parameters (identical for every block).
    pub block_params: BlockParams,
    /// Embedding table `[vocab, D]` parameters.
    pub embed_params: u64,
    /// LM head `[D, vocab]` parameters.
    pub lm_head_params: u64,
    /// Forward FLOPs across every block.
    pub blocks_flops: u64,
    /// LM head forward FLOPs `2·B·T·D·vocab`.
    pub lm_head_flops: u64,
    /// Total forward FLOPs (blocks + LM head).
    pub total_flops: u64,
    /// Per-block forward breakdown (identical for every block).
    pub block_forward: BlockForward,
    /// Tokens processed per forward pass (`batch·seq`).
    pub tokens: u64,
}

/// Count the matmul parameters in one DeepSeek V4.1 transformer block.
///
/// `H*head_dim` is the fanned query/attention width; the KV path is a single
/// shared head (`num_key_value_heads == 1`), so `wkv` is `[D, head_dim]`.
pub fn block_params(cfg: &DeepSeekV41TextConfig) -> BlockParams {
    let d = cfg.hidden_size as u64;
    let h = cfg.num_attention_heads as u64;
    let head_dim = cfg.head_dim as u64;
    let q_lora = cfg.q_lora_rank as u64;
    let o_lora = cfg.o_lora_rank as u64;
    let o_groups = cfg.o_groups as u64;
    let moe_inter = cfg.moe_intermediate_size as u64;
    let n_routed = cfg.n_routed_experts as u64;
    let n_shared = cfg.n_shared_experts as u64;

    // MLA query: wq_a [D, q_lora] + wq_b [q_lora, H*head_dim].
    let attn_q = d * q_lora + q_lora * (h * head_dim);
    // Shared KV projection wkv [D, head_dim].
    let attn_kv = d * head_dim;
    // Grouped output: wo_a [H*head_dim, o_groups*o_lora] + wo_b [o_groups*o_lora, D].
    let attn_o = (h * head_dim) * (o_groups * o_lora) + (o_groups * o_lora) * d;

    // Router gate [D, n_routed_experts].
    let moe_gate = d * n_routed;
    // One SwiGLU expert = w1 + w3 (both [D, moe_inter]) + w2 [moe_inter, D].
    let per_expert = 3 * d * moe_inter;
    let moe_routed_experts = n_routed * per_expert;
    let moe_shared_experts = n_shared * per_expert;

    let total = attn_q + attn_kv + attn_o + moe_gate + moe_routed_experts + moe_shared_experts;
    BlockParams {
        attn_q,
        attn_kv,
        attn_o,
        moe_gate,
        moe_routed_experts,
        moe_shared_experts,
        total,
    }
}

/// Per-block forward FLOPs and activation bytes for a `[batch, seq]` workload.
///
/// A linear `[M, K] · [K, N]` costs `2·M·K·N` forward FLOPs; with `M = B·T`
/// tokens each projection costs `2·B·T·K·N`.  MoE only runs
/// `num_experts_per_tok` routed experts per token, so its cost scales with the
/// *active* experts, not the full bank.
pub fn block_forward(cfg: &DeepSeekV41TextConfig, batch: usize, seq: usize) -> BlockForward {
    let bt = (batch as u64) * (seq as u64);
    let d = cfg.hidden_size as u64;
    let h = cfg.num_attention_heads as u64;
    let head_dim = cfg.head_dim as u64;
    let q_lora = cfg.q_lora_rank as u64;
    let o_lora = cfg.o_lora_rank as u64;
    let o_groups = cfg.o_groups as u64;
    let moe_inter = cfg.moe_intermediate_size as u64;
    let n_active = cfg.num_experts_per_tok as u64;
    let n_shared = cfg.n_shared_experts as u64;
    let n_routed = cfg.n_routed_experts as u64;
    let t = seq as u64;
    let b = batch as u64;

    let lin = |k: u64, n: u64| 2 * bt * k * n;

    // MLA projection FLOPs.
    let attn_proj_flops = lin(d, q_lora)                 // wq_a
        + lin(q_lora, h * head_dim)                      // wq_b
        + lin(d, head_dim)                               // wkv (shared head)
        + lin(h * head_dim, o_groups * o_lora)           // wo_a
        + lin(o_groups * o_lora, d); // wo_b

    // Dense SDPA upper bound: scores Q·Kᵀ (2·B·H·T²·head_dim) + mix P·V (same).
    // Real DeepSeek attention is sparse/windowed, so this is an upper bound.
    let attn_sdpa_flops = 2 * (2 * b * h * t * t * head_dim);

    // MoE: gate over all tokens, then n_active routed + n_shared experts per
    // token, each a 3-matmul SwiGLU (w1,w3: [D,moe_inter]; w2: [moe_inter,D]).
    let gate_flops = lin(d, n_routed);
    let per_expert_flops = 2 * bt * (2 * d * moe_inter + moe_inter * d);
    let moe_flops = gate_flops + (n_active + n_shared) * per_expert_flops;

    let total_flops = attn_proj_flops + attn_sdpa_flops + moe_flops;

    // Activation bytes: the block streams a handful of [B,T,D]-sized buffers
    // (residual, attn out, ffn out) plus the [B,T,moe_inter] expert hidden.
    let act_bytes = (3 * bt * d + bt * moe_inter) * F32_BYTES;

    BlockForward {
        attn_proj_flops,
        attn_sdpa_flops,
        moe_flops,
        total_flops,
        act_bytes,
    }
}

/// Whole-model forward cost for a `[batch, seq]` text-only workload.
pub fn model_forward(cfg: &DeepSeekV41TextConfig, batch: usize, seq: usize) -> ModelForward {
    let bp = block_params(cfg);
    let bf = block_forward(cfg, batch, seq);
    let n_layers = cfg.num_hidden_layers as u64;
    let d = cfg.hidden_size as u64;
    let vocab = cfg.vocab_size as u64;
    let bt = (batch as u64) * (seq as u64);

    let embed_params = vocab * d;
    let lm_head_params = d * vocab;
    let params_total = n_layers * bp.total + embed_params + lm_head_params;

    let blocks_flops = n_layers * bf.total_flops;
    let lm_head_flops = 2 * bt * d * vocab;
    let total_flops = blocks_flops + lm_head_flops;

    ModelForward {
        params_total,
        block_params: bp,
        embed_params,
        lm_head_params,
        blocks_flops,
        lm_head_flops,
        total_flops,
        block_forward: bf,
        tokens: bt,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deepseek_v41::config::{
        DeepSeekV41DsparkConfig, DeepSeekV41TextConfig, DeepSeekV41VisionConfig,
    };

    /// A tiny hand-checkable config: D=4, H=2, head_dim=2, q_lora=3, o_lora=2,
    /// o_groups=2, moe_inter=3, n_routed=2, n_shared=1, n_active=1, vocab=5.
    fn tiny_cfg() -> DeepSeekV41TextConfig {
        DeepSeekV41TextConfig {
            vocab_size: 5,
            hidden_size: 4,
            moe_intermediate_size: 3,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 2,
            qk_rope_head_dim: 2,
            q_lora_rank: 3,
            o_lora_rank: 2,
            o_groups: 2,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            rope_factor: 1.0,
            original_seq_len: 8,
            beta_fast: 32.0,
            beta_slow: 1.0,
            sliding_window: 4,
            compress_ratios: vec![0, 0],
            compress_rope_theta: 10000.0,
            kv_source_layer_ids: vec![],
            index_source_layer_ids: vec![],
            index_n_heads: 1,
            index_head_dim: 2,
            index_topk: 1,
            candidate_source_layer_id: 0,
            candidate_topk_blocks: 1,
            candidate_block_size: 1,
            hc_mult: 2,
            hc_sinkhorn_iters: 1,
            hc_eps: 1e-6,
            n_routed_experts: 2,
            n_shared_experts: 1,
            num_experts_per_tok: 1,
            scoring_func: "sigmoid".to_string(),
            norm_topk_prob: true,
            routed_scaling_factor: 1.0,
            swiglu_limit: 7.0,
            engram_layer_ids: vec![],
            engram_num_embeddings: vec![],
            engram_max_ngram_size: 0,
            engram_vocab_size: 0,
            engram_n_heads: 0,
            engram_head_dim: 0,
            engram_pad_token_id: 0,
            engram_compressed_vocab_size: 0,
            image_token_id: 0,
            dtype: "bf16".to_string(),
            expert_dtype: "bf16".to_string(),
            dspark: DeepSeekV41DsparkConfig {
                n_mtp_layers: 0,
                dspark_block_size: 0,
                dspark_noise_token_id: 0,
                dspark_target_layer_ids: vec![],
                dspark_markov_rank: 0,
                dspark_n_routed_experts: 0,
                dspark_num_experts_per_tok: 0,
            },
            vision: DeepSeekV41VisionConfig {
                num_hidden_layers: 0,
                hidden_size: 0,
                num_attention_heads: 0,
                intermediate_size: 0,
                patch_size: 0,
                rope_theta: 0.0,
                downsample_ratio: 0,
                max_image_tokens: 0,
                min_pixels: 0,
                max_wh_ratio: None,
            },
        }
    }

    #[test]
    fn block_params_match_hand_computed() {
        let bp = block_params(&tiny_cfg());
        // attn_q = D*q_lora + q_lora*(H*head_dim) = 4*3 + 3*(2*2) = 12 + 12 = 24
        assert_eq!(bp.attn_q, 24);
        // attn_kv = D*head_dim = 4*2 = 8
        assert_eq!(bp.attn_kv, 8);
        // attn_o = (H*head_dim)*(o_groups*o_lora) + (o_groups*o_lora)*D
        //        = 4*(2*2) + (2*2)*4 = 16 + 16 = 32
        assert_eq!(bp.attn_o, 32);
        // moe_gate = D*n_routed = 4*2 = 8
        assert_eq!(bp.moe_gate, 8);
        // per_expert = 3*D*moe_inter = 3*4*3 = 36
        // routed = 2*36 = 72 ; shared = 1*36 = 36
        assert_eq!(bp.moe_routed_experts, 72);
        assert_eq!(bp.moe_shared_experts, 36);
        assert_eq!(bp.total, 24 + 8 + 32 + 8 + 72 + 36);
    }

    #[test]
    fn block_forward_flops_match_hand_computed() {
        let cfg = tiny_cfg();
        let bf = block_forward(&cfg, 1, 2);
        // attn_proj = 2*bt*(D*q_lora + q_lora*H*head_dim + D*head_dim
        //   + H*head_dim*o_groups*o_lora + o_groups*o_lora*D)
        //   = 2*2*(12 + 12 + 8 + 16 + 16) = 4*64 = 256
        assert_eq!(bf.attn_proj_flops, 256);
        // sdpa = 2*(2*B*H*T²*head_dim) = 2*(2*1*2*4*2) = 2*32 = 64
        assert_eq!(bf.attn_sdpa_flops, 64);
        // moe = gate(2*bt*D*n_routed) + (n_active+n_shared)*per_expert
        //   gate = 2*2*4*2 = 32
        //   per_expert = 2*bt*(2*D*moe_inter + moe_inter*D) = 2*2*(24+12) = 144
        //   (1+1)*144 = 288 ; moe = 32 + 288 = 320
        assert_eq!(bf.moe_flops, 320);
        assert_eq!(bf.total_flops, 256 + 64 + 320);
    }

    #[test]
    fn model_forward_aggregates_layers_and_head() {
        let cfg = tiny_cfg();
        let mf = model_forward(&cfg, 1, 2);
        let bp = block_params(&cfg);
        let bf = block_forward(&cfg, 1, 2);
        // 2 layers.
        assert_eq!(mf.blocks_flops, 2 * bf.total_flops);
        // embed + lm_head = 2 * (vocab*D) = 2*20 = 40
        assert_eq!(mf.embed_params, 20);
        assert_eq!(mf.lm_head_params, 20);
        assert_eq!(mf.params_total, 2 * bp.total + 40);
        // lm_head_flops = 2*bt*D*vocab = 2*2*4*5 = 80
        assert_eq!(mf.lm_head_flops, 80);
        assert_eq!(mf.total_flops, mf.blocks_flops + 80);
        assert_eq!(mf.tokens, 2);
    }

    #[test]
    fn release_config_is_within_expected_magnitude() {
        // Sanity guard on the real release shape: ~1T total params is the
        // published order of magnitude for DeepSeek V4.1's full MoE bank.
        let cfg = tiny_cfg();
        let big = DeepSeekV41TextConfig {
            vocab_size: 129280,
            hidden_size: 5120,
            moe_intermediate_size: 2304,
            num_hidden_layers: 40,
            num_attention_heads: 64,
            head_dim: 512,
            q_lora_rank: 1280,
            o_lora_rank: 1024,
            o_groups: 8,
            n_routed_experts: 384,
            n_shared_experts: 1,
            num_experts_per_tok: 6,
            ..cfg
        };
        let mf = model_forward(&big, 1, 1);
        // Full parameter bank should land in the hundreds-of-billions range.
        assert!(
            mf.params_total > 100_000_000_000,
            "params_total too small: {}",
            mf.params_total
        );
        // Active FLOPs per token must be far below the full-bank param count
        // (only 6 of 384 routed experts run), proving the sparsity is modeled.
        let full_bank_flops = 2 * mf.tokens * mf.params_total;
        assert!(
            mf.total_flops < full_bank_flops / 10,
            "MoE sparsity not reflected: total_flops={} full_bank_flops={}",
            mf.total_flops,
            full_bank_flops
        );
    }
}
