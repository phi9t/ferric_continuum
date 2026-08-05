//! Minimal byte-level BPE tokenizer (GPT-2 / Qwen2 family).
//!
//! Loads `vocab.json` (token string → id) and `merges.txt` (ranked BPE merge
//! rules) and implements just enough of the Hugging Face `GPT2Tokenizer`
//! byte-level BPE to encode/decode plain text for a coherent-text inference
//! smoke test.
//!
//! Pipeline (encode):
//! 1. **Pre-tokenize** the input into GPT-2-style word chunks (a hand-rolled
//!    approximation of the GPT-2 regex: runs of letters, digits, and other
//!    characters, with leading spaces attached to the following word).
//! 2. **Byte→unicode map** each chunk's UTF-8 bytes into the printable-unicode
//!    alphabet GPT-2 uses (so no raw control bytes appear as vocab keys).
//! 3. Greedily apply merges by ascending rank until no adjacent pair has a rule.
//! 4. Map each resulting token string to its id via `vocab.json`.
//!
//! Decode reverses steps 4→2: id → token string → bytes → UTF-8.
//!
//! This is intentionally **not** byte-exact to HF's fast tokenizer (the regex
//! pre-tokenization is approximate and special/chat tokens are not injected),
//! but it is faithful enough for greedy completion of ordinary English prompts.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Byte-level BPE tokenizer.
pub struct Tokenizer {
    /// token string → id
    encoder: HashMap<String, usize>,
    /// id → token string
    decoder: HashMap<usize, String>,
    /// (left, right) merged token → rank (lower merges first)
    merge_ranks: HashMap<(String, String), usize>,
    /// byte value → unicode char (GPT-2 table)
    byte_to_unicode: [char; 256],
    /// unicode char → byte value (inverse table)
    unicode_to_byte: HashMap<char, u8>,
}

/// GPT-2 reversible byte↔unicode table: maps every byte to a printable unicode
/// codepoint so BPE never operates on raw control bytes.
fn build_byte_unicode() -> ([char; 256], HashMap<char, u8>) {
    let mut bs: Vec<u32> = Vec::new();
    // Printable ASCII/Latin ranges kept as themselves.
    bs.extend(b'!' as u32..=b'~' as u32);
    bs.extend(0xA1u32..=0xAC);
    bs.extend(0xAEu32..=0xFF);

    let mut cs: Vec<u32> = bs.clone();
    let mut n = 0u32;
    for b in 0u32..256 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }

    let mut b2u = ['\0'; 256];
    let mut u2b = HashMap::new();
    for (b, c) in bs.iter().zip(cs.iter()) {
        let ch = char::from_u32(*c).unwrap();
        b2u[*b as usize] = ch;
        u2b.insert(ch, *b as u8);
    }
    (b2u, u2b)
}

/// GPT-2-approximate pre-tokenizer: split into runs, attaching a leading space
/// to the following run (so words carry their preceding space, matching the
/// `Ġword` convention).
fn pretokenize(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut out: Vec<String> = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        // Absorb a single leading space into the next chunk.
        let mut chunk = String::new();
        if chars[i] == ' ' {
            chunk.push(' ');
            i += 1;
            if i >= chars.len() {
                out.push(chunk);
                break;
            }
        }
        let c = chars[i];
        if c.is_alphabetic() {
            while i < chars.len() && chars[i].is_alphabetic() {
                chunk.push(chars[i]);
                i += 1;
            }
        } else if c.is_numeric() {
            while i < chars.len() && chars[i].is_numeric() {
                chunk.push(chars[i]);
                i += 1;
            }
        } else if c == ' ' {
            // Multiple consecutive spaces: emit remaining spaces individually.
            while i < chars.len() && chars[i] == ' ' {
                chunk.push(chars[i]);
                i += 1;
            }
        } else {
            // Punctuation / symbols: one char per chunk.
            chunk.push(c);
            i += 1;
        }
        out.push(chunk);
    }
    out
}

impl Tokenizer {
    /// Build a tokenizer from `vocab.json` + `merges.txt` in `model_dir`.
    pub fn from_files(model_dir: &Path) -> Result<Tokenizer, String> {
        let vocab_path = model_dir.join("vocab.json");
        let vtext = fs::read_to_string(&vocab_path)
            .map_err(|e| format!("read {}: {e}", vocab_path.display()))?;
        let vjson: serde_json::Value =
            serde_json::from_str(&vtext).map_err(|e| format!("parse vocab.json: {e}"))?;
        let obj = vjson
            .as_object()
            .ok_or_else(|| "vocab.json is not an object".to_string())?;

        let mut encoder = HashMap::with_capacity(obj.len());
        let mut decoder = HashMap::with_capacity(obj.len());
        for (tok, id) in obj {
            let id = id
                .as_u64()
                .ok_or_else(|| format!("vocab id for `{tok}` not an int"))?
                as usize;
            encoder.insert(tok.clone(), id);
            decoder.insert(id, tok.clone());
        }

        let merges_path = model_dir.join("merges.txt");
        let mtext = fs::read_to_string(&merges_path)
            .map_err(|e| format!("read {}: {e}", merges_path.display()))?;
        let mut merge_ranks = HashMap::new();
        let mut rank = 0usize;
        for line in mtext.lines() {
            if line.starts_with("#version") || line.trim().is_empty() {
                continue;
            }
            let mut parts = line.split(' ');
            if let (Some(a), Some(b)) = (parts.next(), parts.next()) {
                merge_ranks.insert((a.to_string(), b.to_string()), rank);
                rank += 1;
            }
        }

        let (byte_to_unicode, unicode_to_byte) = build_byte_unicode();

        Ok(Tokenizer {
            encoder,
            decoder,
            merge_ranks,
            byte_to_unicode,
            unicode_to_byte,
        })
    }

    /// Apply BPE merges to a sequence of single-char token strings.
    fn bpe(&self, mut symbols: Vec<String>) -> Vec<String> {
        if symbols.len() < 2 {
            return symbols;
        }
        loop {
            // Find the lowest-rank adjacent pair.
            let mut best: Option<(usize, usize)> = None; // (rank, index)
            for i in 0..symbols.len() - 1 {
                if let Some(&r) = self
                    .merge_ranks
                    .get(&(symbols[i].clone(), symbols[i + 1].clone()))
                {
                    if best.map(|(br, _)| r < br).unwrap_or(true) {
                        best = Some((r, i));
                    }
                }
            }
            let Some((_, idx)) = best else { break };
            let merged = format!("{}{}", symbols[idx], symbols[idx + 1]);
            symbols.splice(idx..idx + 2, std::iter::once(merged));
        }
        symbols
    }

    /// Encode `text` into token ids.
    pub fn encode(&self, text: &str) -> Vec<usize> {
        let mut ids = Vec::new();
        for chunk in pretokenize(text) {
            // Map the chunk's UTF-8 bytes into the GPT-2 unicode alphabet.
            let symbols: Vec<String> = chunk
                .as_bytes()
                .iter()
                .map(|&b| self.byte_to_unicode[b as usize].to_string())
                .collect();
            for tok in self.bpe(symbols) {
                match self.encoder.get(&tok) {
                    Some(&id) => ids.push(id),
                    None => {
                        // Fall back to per-char (single-byte-unicode) ids; every
                        // byte-mapped char is guaranteed present in the vocab.
                        for ch in tok.chars() {
                            let s = ch.to_string();
                            if let Some(&id) = self.encoder.get(&s) {
                                ids.push(id);
                            }
                        }
                    }
                }
            }
        }
        ids
    }

    /// Decode token ids back into a UTF-8 string.
    pub fn decode(&self, ids: &[usize]) -> String {
        let mut bytes: Vec<u8> = Vec::new();
        for &id in ids {
            if let Some(tok) = self.decoder.get(&id) {
                for ch in tok.chars() {
                    if let Some(&b) = self.unicode_to_byte.get(&ch) {
                        bytes.push(b);
                    }
                }
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_unicode_is_bijective() {
        let (b2u, u2b) = build_byte_unicode();
        for b in 0u16..256 {
            let ch = b2u[b as usize];
            assert_eq!(u2b.get(&ch), Some(&(b as u8)));
        }
    }

    #[test]
    fn pretokenize_attaches_leading_space() {
        assert_eq!(pretokenize("the cat"), vec!["the", " cat"]);
        assert_eq!(pretokenize("a1 b"), vec!["a", "1", " b"]);
    }
}
