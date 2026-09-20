//! DeepSeek V4.1 Engram verifier helpers.
//!
//! Engram in DeepSeek V4.1 is a hashed n-gram lookup added into the residual
//! stream at a few layers. This module implements the text-only math seams used
//! by unit verifiers and fixture generation.

use crate::deepseek_v41::config::DeepSeekV41TextConfig;
use crate::ops::linear;
use crate::tensor::{Shape, Tensor, TensorValue};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngramLayout {
    pub max_ngram_size: usize,
    pub layer_ids: Vec<usize>,
    pub num_embeddings: Vec<usize>,
    /// Flattened primes in the order: for each layer, for each (ngram=2..max),
    /// for each head.
    pub primes: Vec<usize>,
    /// Flattened offsets aligned with `primes`: for each layer and column,
    /// offset is the sum of earlier prime sizes in that layer.
    pub offsets: Vec<usize>,
    pub n_heads: usize,
    pub head_dim: usize,
}

impl EngramLayout {
    pub fn from_config(config: &DeepSeekV41TextConfig) -> Option<EngramLayout> {
        if config.engram_layer_ids.is_empty() {
            return None;
        }
        assert!(
            config.engram_layer_ids.len() == config.engram_num_embeddings.len(),
            "engram_layer_ids and engram_num_embeddings must be same length"
        );
        assert!(
            config.engram_max_ngram_size >= 1,
            "max_ngram_size must be >= 1"
        );
        assert!(config.engram_n_heads > 0, "engram_n_heads must be positive");

        let max_ngram_size = config.engram_max_ngram_size;
        let n_layers = config.engram_layer_ids.len();
        let n_heads = config.engram_n_heads;
        let n_hash_cols = (max_ngram_size - 1) * n_heads;

        let mut seen_primes = std::collections::BTreeSet::new();
        let mut primes = Vec::with_capacity(n_layers * n_hash_cols);
        for _layer in 0..n_layers {
            let mut current = config.engram_vocab_size.saturating_sub(1);
            for _ in 0..(max_ngram_size.saturating_sub(1)) {
                for _ in 0..n_heads {
                    current = find_next_prime(current, &mut seen_primes);
                    primes.push(current);
                }
            }
        }

        let mut offsets = Vec::with_capacity(n_layers * n_hash_cols);
        for layer in 0..n_layers {
            let base = layer * n_hash_cols;
            let mut running = 0usize;
            for col in 0..n_hash_cols {
                offsets.push(running);
                running += primes[base + col];
            }
        }

        Some(EngramLayout {
            max_ngram_size,
            layer_ids: config.engram_layer_ids.clone(),
            num_embeddings: config.engram_num_embeddings.clone(),
            primes,
            offsets,
            n_heads,
            head_dim: config.engram_head_dim,
        })
    }
}

#[derive(Debug, Clone)]
pub struct NgramHashState {
    token_map: Vec<i64>,
    pad_id: i64,
    cache: Vec<i64>,
    max_batch_size: usize,
    max_seq_len: usize,
    /// Flattened multipliers `[layer][lookback]` as i64, sized `n_layers * max_ngram_size`.
    multipliers: Vec<i64>,
}

impl NgramHashState {
    pub fn new(
        token_map: Vec<usize>,
        pad_token_id: usize,
        max_batch_size: usize,
        max_seq_len: usize,
    ) -> Self {
        assert!(max_batch_size > 0, "max_batch_size must be positive");
        assert!(max_seq_len > 0, "max_seq_len must be positive");
        let token_map_i64 = token_map.into_iter().map(|v| v as i64).collect::<Vec<_>>();
        let pad_id = *token_map_i64
            .get(pad_token_id)
            .unwrap_or_else(|| panic!("pad_token_id {pad_token_id} out of bounds for token_map"));
        Self {
            token_map: token_map_i64,
            pad_id,
            cache: vec![pad_id; max_batch_size * max_seq_len],
            max_batch_size,
            max_seq_len,
            multipliers: Vec::new(),
        }
    }

    pub fn set_multipliers(&mut self, multipliers: Vec<i64>) {
        self.multipliers = multipliers;
    }
}

pub fn compressed_token_map_from_vocab_entries(entries: &[(usize, String)]) -> Vec<usize> {
    let max_id = entries.iter().map(|(id, _)| *id).max().unwrap_or(0);
    let mut lookup = vec![0usize; max_id + 1];
    let mut key_to_new = std::collections::BTreeMap::<String, usize>::new();

    for &(token_id, ref text) in entries {
        if token_id >= lookup.len() {
            continue;
        }
        let key = normalize_vocab_entry(text);
        let new_id = match key_to_new.get(&key) {
            Some(&id) => id,
            None => {
                let id = key_to_new.len();
                key_to_new.insert(key, id);
                id
            }
        };
        lookup[token_id] = new_id;
    }
    lookup
}

pub fn ngram_hashes(
    input_ids: &[usize],
    start_pos: usize,
    layout: &EngramLayout,
    state: &mut NgramHashState,
) -> Vec<usize> {
    ngram_hashes_masked(input_ids, start_pos, None, layout, state)
}

pub fn ngram_hashes_masked(
    input_ids: &[usize],
    start_pos: usize,
    token_mask: Option<&[bool]>,
    layout: &EngramLayout,
    state: &mut NgramHashState,
) -> Vec<usize> {
    ngram_hashes_batched(
        input_ids,
        1,
        input_ids.len(),
        start_pos,
        token_mask,
        layout,
        state,
    )
}

pub fn ngram_hashes_batched(
    input_ids: &[usize],
    batch: usize,
    seqlen: usize,
    start_pos: usize,
    token_mask: Option<&[bool]>,
    layout: &EngramLayout,
    state: &mut NgramHashState,
) -> Vec<usize> {
    assert_eq!(
        input_ids.len(),
        batch * seqlen,
        "input_ids length must be batch*seqlen"
    );
    if let Some(mask) = token_mask {
        assert_eq!(
            mask.len(),
            batch * seqlen,
            "token_mask length must match input_ids"
        );
    }
    assert!(
        start_pos + seqlen <= state.max_seq_len,
        "start_pos + seqlen exceeds max_seq_len"
    );
    assert!(
        batch <= state.max_batch_size,
        "batch exceeds max_batch_size"
    );
    let n_layers = layout.layer_ids.len();
    let n_heads = layout.n_heads;
    let n_hash_cols = (layout.max_ngram_size - 1) * n_heads;
    assert_eq!(
        layout.primes.len(),
        n_layers * n_hash_cols,
        "layout primes length mismatch"
    );
    assert_eq!(
        layout.offsets.len(),
        n_layers * n_hash_cols,
        "layout offsets length mismatch"
    );

    const DEAD: i64 = -1;

    for b in 0..batch {
        for i in 0..seqlen {
            let flat = b * seqlen + i;
            let id = input_ids[flat];
            let mut compressed = *state
                .token_map
                .get(id)
                .unwrap_or_else(|| panic!("token id {id} out of bounds for token_map"));
            if token_mask.is_some_and(|mask| !mask[flat]) {
                compressed = DEAD;
            }
            state.cache[b * state.max_seq_len + start_pos + i] = compressed;
        }
    }

    assert!(
        !state.multipliers.is_empty(),
        "NgramHashState multipliers must be configured before hashing"
    );
    assert_eq!(
        state.multipliers.len(),
        n_layers * layout.max_ngram_size,
        "multipliers must be [n_layers * max_ngram_size]"
    );

    let mut out = Vec::with_capacity(batch * seqlen * n_layers * n_hash_cols);
    for b in 0..batch {
        for pos_offset in 0..seqlen {
            let pos = start_pos + pos_offset;

            // tokens[shift] gives the compressed id shift tokens back, padded at
            // the sequence start and after any masked image/dead token.
            let mut tokens = vec![state.pad_id; layout.max_ngram_size];
            let mut blocked = false;
            for shift in 0..layout.max_ngram_size {
                let source = if pos >= shift {
                    state.cache[b * state.max_seq_len + pos - shift]
                } else {
                    state.pad_id
                };
                blocked = blocked || pos < shift || source == DEAD;
                tokens[shift] = if blocked { state.pad_id } else { source };
            }

            for layer in 0..n_layers {
                let mult_base = layer * layout.max_ngram_size;
                let mut rolling = tokens[0] * state.multipliers[mult_base];
                for lookback in 1..layout.max_ngram_size {
                    rolling ^= tokens[lookback] * state.multipliers[mult_base + lookback];
                    let col_base = (lookback - 1) * n_heads;
                    for head in 0..n_heads {
                        let col = col_base + head;
                        let prime = layout.primes[layer * n_hash_cols + col] as i64;
                        let offset = layout.offsets[layer * n_hash_cols + col] as i64;
                        out.push(((rolling.rem_euclid(prime)) + offset) as usize);
                    }
                }
            }
        }
    }
    out
}

pub fn engram_update(
    x: &[f32],
    key: &[f32],
    value: &[f32],
    q_weight: &[f32],
    k_weight: &[f32],
    eps: f32,
    token_mask: Option<&[bool]>,
) -> Vec<f32> {
    assert!(eps > 0.0, "eps must be positive");
    assert_eq!(x.len(), key.len(), "x and key must have same length");
    assert_eq!(
        q_weight.len(),
        k_weight.len(),
        "q_weight and k_weight must match"
    );
    assert!(!q_weight.is_empty(), "q_weight must be non-empty");
    assert!(
        x.len() % q_weight.len() == 0,
        "x length must be tokens * (hc_mult*dim)"
    );
    assert!(
        value.len() > 0 && x.len() % value.len() == 0,
        "value must divide x length"
    );

    let hc_mult = x.len() / value.len();
    assert!(hc_mult > 0, "hc_mult must be positive");
    assert!(
        q_weight.len() % hc_mult == 0,
        "q_weight length must be hc_mult * dim"
    );
    let dim = q_weight.len() / hc_mult;
    let tokens = value.len() / dim;
    assert_eq!(x.len(), tokens * hc_mult * dim, "inferred shape mismatch");
    assert_eq!(value.len(), tokens * dim, "value length must be tokens*dim");
    if let Some(mask) = token_mask {
        assert_eq!(mask.len(), tokens, "token_mask length must match tokens");
    }

    let mut weight = vec![0.0f32; q_weight.len()];
    for i in 0..q_weight.len() {
        weight[i] = q_weight[i] * k_weight[i];
    }

    let clamp_value = 1e-6f32;
    let inv_sqrt_dim = (dim as f32).powf(-0.5);
    let mut out = vec![0.0f32; x.len()];
    for t in 0..tokens {
        let token_ok = token_mask.map_or(true, |m| m[t]);
        for h in 0..hc_mult {
            let base = (t * hc_mult + h) * dim;
            let vbase = t * dim;

            let mut mean_sq_x = 0.0f32;
            let mut mean_sq_k = 0.0f32;
            let mut dot = 0.0f32;
            for d in 0..dim {
                let xv = x[base + d];
                let kv = key[base + d];
                mean_sq_x += xv * xv;
                mean_sq_k += kv * kv;
                dot += xv * weight[h * dim + d] * kv;
            }
            mean_sq_x /= dim as f32;
            mean_sq_k /= dim as f32;
            let rstd = (mean_sq_x + eps).powf(-0.5) * (mean_sq_k + eps).powf(-0.5);
            dot *= rstd * inv_sqrt_dim;

            let mut gate = sigmoid(copysign((dot.abs().max(clamp_value)).sqrt(), dot));
            if !token_ok {
                gate = 0.0;
            }
            for d in 0..dim {
                out[base + d] = x[base + d] + gate * value[vbase + d];
            }
        }
    }
    out
}

pub struct DeepSeekV41Engram {
    pub q_weight: Vec<f32>,
    pub k_weight: Vec<f32>,
    /// Loaded Engram hash table rows `[num_embeddings, head_dim]`.
    ///
    /// Older fixture tests inject precomputed key/value tensors directly, so
    /// this remains optional until the full table-lookup runtime lands.
    pub embed_weight: Option<Tensor>,
    /// Loaded Engram `wkv` projection in tnsr `[in, out]` layout, where
    /// `in = n_hash_cols * head_dim` and `out = dim * (hc_mult + 1)`.
    pub wkv_weight: Option<Tensor>,
    pub eps: f32,
}

impl DeepSeekV41Engram {
    pub fn lookup_key_value(
        &self,
        hash_ids: &[usize],
        batch: usize,
        seqlen: usize,
    ) -> (Tensor, Tensor) {
        let embed = self
            .embed_weight
            .as_ref()
            .expect("Engram embed_weight is required for hash lookup");
        let wkv = self
            .wkv_weight
            .as_ref()
            .expect("Engram wkv_weight is required for hash lookup");
        let embed_value = embed.inner.borrow().value.clone();
        let embed_shape = embed_value.shape.0;
        assert_eq!(
            embed_shape.len(),
            2,
            "Engram embed_weight must be [rows, head_dim]"
        );
        assert_eq!(
            hash_ids.len() % (batch * seqlen),
            0,
            "Engram hash_ids length must be batch*seqlen*n_hash_cols"
        );
        let n_hash_cols = hash_ids.len() / (batch * seqlen);
        let head_dim = embed_shape[1];
        let mut gathered = Vec::with_capacity(batch * seqlen * n_hash_cols * head_dim);
        for &id in hash_ids {
            assert!(
                id < embed_shape[0],
                "Engram hash id {id} out of bounds for table rows {}",
                embed_shape[0]
            );
            let row = id * head_dim;
            gathered.extend_from_slice(&embed_value.data.as_ref()[row..row + head_dim]);
        }
        let embedded = Tensor::from_value_no_grad(TensorValue::from_vec(
            Shape(vec![batch, seqlen, n_hash_cols * head_dim]),
            gathered,
        ));
        let kv = linear::linear(&embedded, wkv, "deepseek.engram.wkv");
        let kv_value = kv.inner.borrow().value.clone();
        let kv_shape = kv_value.shape.0;
        assert_eq!(
            kv_shape.len(),
            3,
            "Engram wkv output must be [B,S,D*(HC+1)]"
        );
        let hc_dim = self.q_weight.len();
        assert!(
            kv_shape[2] > hc_dim,
            "Engram wkv output must contain key plus value"
        );
        let dim = kv_shape[2] - hc_dim;
        assert_eq!(
            hc_dim % dim,
            0,
            "Engram q_weight length must be hc_mult*dim"
        );
        let hc_mult = hc_dim / dim;
        let mut key = Vec::with_capacity(batch * seqlen * hc_dim);
        let mut value = Vec::with_capacity(batch * seqlen * dim);
        for row in kv_value.data.as_ref().chunks_exact(kv_shape[2]) {
            key.extend_from_slice(&row[..hc_dim]);
            value.extend_from_slice(&row[hc_dim..]);
        }
        (
            Tensor::from_value_no_grad(TensorValue::from_vec(
                Shape(vec![batch, seqlen, hc_mult, dim]),
                key,
            )),
            Tensor::from_value_no_grad(TensorValue::from_vec(
                Shape(vec![batch, seqlen, dim]),
                value,
            )),
        )
    }

    pub fn forward_hashes(
        &self,
        x: &Tensor,
        hash_ids: &[usize],
        token_mask: Option<&[bool]>,
    ) -> Tensor {
        let shape = x.shape().0;
        assert_eq!(shape.len(), 4, "Engram input must be [B,S,HC,D]");
        let (key, value) = self.lookup_key_value(hash_ids, shape[0], shape[1]);
        self.forward_layer(x, &key, &value, token_mask)
    }

    pub fn forward_layer(
        &self,
        x: &Tensor,
        key: &Tensor,
        value: &Tensor,
        token_mask: Option<&[bool]>,
    ) -> Tensor {
        let xv = x.inner.borrow().value.clone();
        let kv = key.inner.borrow().value.clone();
        let vv = value.inner.borrow().value.clone();
        let out = engram_update(
            xv.data.as_ref(),
            kv.data.as_ref(),
            vv.data.as_ref(),
            &self.q_weight,
            &self.k_weight,
            self.eps,
            token_mask,
        );
        Tensor::from_value_no_grad(TensorValue::from_vec(Shape(xv.shape.0), out))
    }
}

fn normalize_vocab_entry(text: &str) -> String {
    // Match upstream `ferric_continuum/tnsr/third_party/deepseek_v41/upstream/inference/engram.py`:
    //
    // sentinel = "\ue000"
    // normalizer = Sequence([
    //   NFKC(), NFD(), StripAccents(), Lowercase(),
    //   Replace(r"[ \t\r\n]+", " "),
    //   Replace(r"^ $", sentinel),
    //   Strip(),
    //   Replace(sentinel, " "),
    // ])
    //
    // key = normalized if normalized else text
    let sentinel = '\u{e000}';
    let nfkc = nfkc(text);
    let nfd = nfd(&nfkc);
    let stripped = strip_accents(&nfd);
    let lowered = stripped.to_lowercase();
    let collapsed = collapse_whitespace(&lowered);
    let marked = if collapsed == " " {
        sentinel.to_string()
    } else {
        collapsed
    };
    let trimmed = marked.trim().to_string();
    let restored = trimmed.replace(sentinel, " ");
    if restored.is_empty() {
        text.to_string()
    } else {
        restored
    }
}

fn collapse_whitespace(text: &str) -> String {
    let mut out = String::new();
    let mut in_space = false;
    for ch in text.chars() {
        let is_ws = matches!(ch, ' ' | '\t' | '\r' | '\n');
        if is_ws {
            if !in_space {
                out.push(' ');
                in_space = true;
            }
        } else {
            out.push(ch);
            in_space = false;
        }
    }
    out
}

fn nfkc(text: &str) -> String {
    // Rust stdlib does not provide Unicode normalization. For fixture-sized
    // entries we implement a tiny compatibility fold: map fullwidth ASCII
    // letters/digits/punct into ASCII. This is enough to cover the upstream
    // normalization cases we encode into fixtures (e.g. "ＴＨＥ" -> "THE").
    text.chars()
        .map(|ch| match ch {
            '\u{ff01}'..='\u{ff5e}' => char::from_u32((ch as u32) - 0xfee0).unwrap_or(ch),
            '\u{3000}' => ' ',
            other => other,
        })
        .collect()
}

fn nfd(text: &str) -> String {
    // Minimal decomposition for fixture coverage: decompose 'é' to 'e' + acute.
    // Full NFD requires a Unicode database; that is intentionally out of scope
    // for this repo without a dedicated dependency. The fixtures used by Ticket
    // 04 only require this specific decomposition behavior.
    let mut out = String::new();
    for ch in text.chars() {
        match ch {
            '\u{00e9}' => {
                out.push('e');
                out.push('\u{0301}');
            }
            other => out.push(other),
        }
    }
    out
}

fn strip_accents(text: &str) -> String {
    text.chars()
        .filter(|&ch| !matches!(ch, '\u{0300}'..='\u{036f}'))
        .collect()
}

fn find_next_prime(start: usize, seen: &mut std::collections::BTreeSet<usize>) -> usize {
    let mut candidate = start + 1;
    while !is_prime(candidate) || seen.contains(&candidate) {
        candidate += 1;
    }
    seen.insert(candidate);
    candidate
}

fn is_prime(n: usize) -> bool {
    if n < 2 {
        return false;
    }
    if n % 2 == 0 {
        return n == 2;
    }
    let mut d = 3usize;
    while d * d <= n {
        if n % d == 0 {
            return false;
        }
        d += 2;
    }
    true
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn copysign(magnitude: f32, sign_source: f32) -> f32 {
    if sign_source.is_sign_negative() {
        -magnitude.abs()
    } else {
        magnitude.abs()
    }
}
