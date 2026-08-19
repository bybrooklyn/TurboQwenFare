//! TQKV: paged, mixed-precision KV representation for the ten full-attention
//! layers (spec Part VIII sections 58-61; page mechanics sections 155-162).
//!
//! Phase 27 REFERENCE BASELINE: 256-token sealed pages with a high-precision
//! (f32) mutable tail, symmetric per-page Q8 (section 158) and Q4
//! (section 159) Key/Value quantization, and a two-pass streaming attention
//! consumer that dequantizes one token fragment at a time rather than
//! materializing a whole page as BF16/FP32 (section 161: "a reference
//! two-pass page implementation is acceptable before fusion"). Q3/Q2/
//! rotation/outlier/pre-RoPE variants remain RESEARCH CANDIDATES (Phase 28,
//! section 160) and are not implemented here.
//!
//! The BF16 `Bf16KvCache` in `model::qwen36::attention` remains the
//! correctness oracle; this module is validated against it differentially
//! (see `tests::` below and `docs/research/qualification/phase-27-*`).

use half::f16;

use crate::error::{ModelError, Result};
use crate::ids::{Bytes, LayerId};
use crate::memory::{MemoryBroker, MemoryClass, MemoryLease, MemoryOwner};
use crate::model::qwen36::geometry::Qwen36Geometry;

pub mod candidates;
pub mod scaling_bench;

const KV_HEADS: usize = Qwen36Geometry::FULL_KV_HEADS;
const HEAD_DIM: usize = Qwen36Geometry::FULL_HEAD_DIM;
const KV_WIDTH: usize = KV_HEADS * HEAD_DIM;
const VALUE_GROUP: usize = 64;
const VALUE_GROUPS: usize = HEAD_DIM / VALUE_GROUP;

/// Spec section 155 reference page geometry: 256 tokens per sealed page.
pub const PAGE_TOKENS: usize = 256;

/// Spec section 157: 128-byte page header.
pub const PAGE_HEADER_BYTES: usize = 128;

/// Which quantized encoding a sealed page uses (spec sections 158-159).
/// `KeyEncoding`/`ValueEncoding` share these IDs since Phase 27 always pairs
/// Key and Value at the same nominal precision class.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TqkvPrecision {
    Q8,
    Q4,
}

impl TqkvPrecision {
    pub(crate) fn encoding_id(self) -> u16 {
        match self {
            TqkvPrecision::Q8 => 1,
            TqkvPrecision::Q4 => 2,
        }
    }

    pub(crate) fn from_encoding_id(id: u16) -> Result<Self> {
        match id {
            1 => Ok(TqkvPrecision::Q8),
            2 => Ok(TqkvPrecision::Q4),
            other => Err(ModelError::Shape {
                tensor: "TQKV page encoding id",
                expected: 1,
                actual: other as usize,
            }
            .into()),
        }
    }

    /// Signed integer clamp range: Q8 uses [-127,127] (section 158), Q4 uses
    /// [-7,7] (section 159) so both keep a symmetric zero-centered code space.
    fn clamp_abs(self) -> f32 {
        match self {
            TqkvPrecision::Q8 => 127.0,
            TqkvPrecision::Q4 => 7.0,
        }
    }

    /// Bytes to store `count` quantized values (Q8: 1 byte/value; Q4: packed
    /// two values/byte, section 159 "packed signed 4-bit").
    fn packed_bytes(self, count: usize) -> usize {
        match self {
            TqkvPrecision::Q8 => count,
            TqkvPrecision::Q4 => count.div_ceil(2),
        }
    }
}

/// Spec section 157 page header, stored little-endian (crate invariant #2).
/// Search fields stay zero in Phase 27 (TQAttn's self-index summary is
/// Phase 32, section 164) and `backing_generation` stays zero (no SSD
/// promote/demote lifecycle yet, section 156 — that is Phase 28+).
#[derive(Clone, Debug, PartialEq)]
pub struct TqkvPageHeader {
    pub page_id: u64,
    pub token_start: u32,
    pub token_count: u16,
    pub layer_id: u8,
    pub kv_head_count: u8,
    pub head_dim: u16,
    pub key_encoding: u16,
    pub value_encoding: u16,
    pub search_encoding: u16,
    pub flags: u16,
    pub key_payload_bytes: u32,
    pub value_payload_bytes: u32,
    pub quant_meta_bytes: u32,
    pub outlier_bytes: u32,
    pub search_bytes: u32,
    pub backing_generation: u32,
    pub key_payload_offset: u64,
    pub value_payload_offset: u64,
    pub quant_meta_offset: u64,
    pub outlier_offset: u64,
    pub search_offset: u64,
    pub content_hash: [u8; 32],
}

impl TqkvPageHeader {
    pub fn to_le_bytes(&self) -> [u8; PAGE_HEADER_BYTES] {
        let mut buf = [0u8; PAGE_HEADER_BYTES];
        buf[0..8].copy_from_slice(&self.page_id.to_le_bytes());
        buf[8..12].copy_from_slice(&self.token_start.to_le_bytes());
        buf[12..14].copy_from_slice(&self.token_count.to_le_bytes());
        buf[14] = self.layer_id;
        buf[15] = self.kv_head_count;
        buf[16..18].copy_from_slice(&self.head_dim.to_le_bytes());
        buf[18..20].copy_from_slice(&self.key_encoding.to_le_bytes());
        buf[20..22].copy_from_slice(&self.value_encoding.to_le_bytes());
        buf[22..24].copy_from_slice(&self.search_encoding.to_le_bytes());
        buf[24..26].copy_from_slice(&self.flags.to_le_bytes());
        buf[26..30].copy_from_slice(&self.key_payload_bytes.to_le_bytes());
        buf[30..34].copy_from_slice(&self.value_payload_bytes.to_le_bytes());
        buf[34..38].copy_from_slice(&self.quant_meta_bytes.to_le_bytes());
        buf[38..42].copy_from_slice(&self.outlier_bytes.to_le_bytes());
        buf[42..46].copy_from_slice(&self.search_bytes.to_le_bytes());
        buf[46..50].copy_from_slice(&self.backing_generation.to_le_bytes());
        buf[50..58].copy_from_slice(&self.key_payload_offset.to_le_bytes());
        buf[58..66].copy_from_slice(&self.value_payload_offset.to_le_bytes());
        buf[66..74].copy_from_slice(&self.quant_meta_offset.to_le_bytes());
        buf[74..82].copy_from_slice(&self.outlier_offset.to_le_bytes());
        buf[82..90].copy_from_slice(&self.search_offset.to_le_bytes());
        buf[90..122].copy_from_slice(&self.content_hash);
        buf
    }

    pub fn from_le_bytes(buf: &[u8; PAGE_HEADER_BYTES]) -> Self {
        let mut content_hash = [0u8; 32];
        content_hash.copy_from_slice(&buf[90..122]);
        Self {
            page_id: u64::from_le_bytes(buf[0..8].try_into().unwrap()),
            token_start: u32::from_le_bytes(buf[8..12].try_into().unwrap()),
            token_count: u16::from_le_bytes(buf[12..14].try_into().unwrap()),
            layer_id: buf[14],
            kv_head_count: buf[15],
            head_dim: u16::from_le_bytes(buf[16..18].try_into().unwrap()),
            key_encoding: u16::from_le_bytes(buf[18..20].try_into().unwrap()),
            value_encoding: u16::from_le_bytes(buf[20..22].try_into().unwrap()),
            search_encoding: u16::from_le_bytes(buf[22..24].try_into().unwrap()),
            flags: u16::from_le_bytes(buf[24..26].try_into().unwrap()),
            key_payload_bytes: u32::from_le_bytes(buf[26..30].try_into().unwrap()),
            value_payload_bytes: u32::from_le_bytes(buf[30..34].try_into().unwrap()),
            quant_meta_bytes: u32::from_le_bytes(buf[34..38].try_into().unwrap()),
            outlier_bytes: u32::from_le_bytes(buf[38..42].try_into().unwrap()),
            search_bytes: u32::from_le_bytes(buf[42..46].try_into().unwrap()),
            backing_generation: u32::from_le_bytes(buf[46..50].try_into().unwrap()),
            key_payload_offset: u64::from_le_bytes(buf[50..58].try_into().unwrap()),
            value_payload_offset: u64::from_le_bytes(buf[58..66].try_into().unwrap()),
            quant_meta_offset: u64::from_le_bytes(buf[66..74].try_into().unwrap()),
            outlier_offset: u64::from_le_bytes(buf[74..82].try_into().unwrap()),
            search_offset: u64::from_le_bytes(buf[82..90].try_into().unwrap()),
            content_hash,
        }
    }
}

/// One sealed, immutable (spec section 156) TQKV page for a single layer.
/// Key scales are per `(kv_head, dim)` over the whole page (section 158);
/// Value scales are per `(token, kv_head, group)`.
#[derive(Clone)]
pub(crate) struct SealedPage {
    header: TqkvPageHeader,
    key_bytes: Vec<u8>,
    key_scales: Vec<f16>,
    value_bytes: Vec<u8>,
    value_scales: Vec<f16>,
    precision: TqkvPrecision,
}

impl SealedPage {
    /// `keys`/`values` are `[token][kv_head][dim]` f32, post-RoPE, exactly
    /// `token_count * KV_WIDTH` elements each (matching `Bf16KvCache`'s
    /// layout so the two caches are drop-in interchangeable).
    fn seal(
        page_id: u64,
        layer: LayerId,
        token_start: u32,
        token_count: usize,
        keys: &[f32],
        values: &[f32],
        precision: TqkvPrecision,
    ) -> Self {
        debug_assert_eq!(keys.len(), token_count * KV_WIDTH);
        debug_assert_eq!(values.len(), token_count * KV_WIDTH);

        // Key scale: max |k| over all tokens in the page, per (kv_head, dim).
        let mut key_scales = vec![0f32; KV_HEADS * HEAD_DIM];
        for token in 0..token_count {
            for head in 0..KV_HEADS {
                let base = (token * KV_HEADS + head) * HEAD_DIM;
                for dim in 0..HEAD_DIM {
                    let v = keys[base + dim].abs();
                    let slot = head * HEAD_DIM + dim;
                    if v > key_scales[slot] {
                        key_scales[slot] = v;
                    }
                }
            }
        }
        let clamp = precision.clamp_abs();
        for scale in key_scales.iter_mut() {
            *scale = if *scale > 0.0 { *scale / clamp } else { 1.0 };
        }

        let mut key_codes = vec![0i8; token_count * KV_WIDTH];
        for token in 0..token_count {
            for head in 0..KV_HEADS {
                let base = (token * KV_HEADS + head) * HEAD_DIM;
                for dim in 0..HEAD_DIM {
                    let scale = key_scales[head * HEAD_DIM + dim];
                    let q = (keys[base + dim] / scale).round().clamp(-clamp, clamp);
                    key_codes[base + dim] = q as i8;
                }
            }
        }

        // Value scale: max |v| per (token, kv_head, group of 64 dims).
        let mut value_scales = vec![0f32; token_count * KV_HEADS * VALUE_GROUPS];
        let mut value_codes = vec![0i8; token_count * KV_WIDTH];
        for token in 0..token_count {
            for head in 0..KV_HEADS {
                let base = (token * KV_HEADS + head) * HEAD_DIM;
                for group in 0..VALUE_GROUPS {
                    let g0 = group * VALUE_GROUP;
                    let max_abs = values[base + g0..base + g0 + VALUE_GROUP]
                        .iter()
                        .fold(0f32, |acc, v| acc.max(v.abs()));
                    let scale = if max_abs > 0.0 { max_abs / clamp } else { 1.0 };
                    value_scales[(token * KV_HEADS + head) * VALUE_GROUPS + group] = scale;
                    for dim in g0..g0 + VALUE_GROUP {
                        let q = (values[base + dim] / scale).round().clamp(-clamp, clamp);
                        value_codes[base + dim] = q as i8;
                    }
                }
            }
        }

        let key_bytes = pack_codes(&key_codes, precision);
        let value_bytes = pack_codes(&value_codes, precision);
        let key_scales: Vec<f16> = key_scales.into_iter().map(f16::from_f32).collect();
        let value_scales: Vec<f16> = value_scales.into_iter().map(f16::from_f32).collect();

        let quant_meta_bytes =
            ((key_scales.len() + value_scales.len()) * std::mem::size_of::<f16>()) as u32;
        let mut hasher = blake3::Hasher::new();
        hasher.update(&key_bytes);
        hasher.update(&value_bytes);
        for s in &key_scales {
            hasher.update(&s.to_le_bytes());
        }
        for s in &value_scales {
            hasher.update(&s.to_le_bytes());
        }
        let content_hash: [u8; 32] = *hasher.finalize().as_bytes();

        let header = TqkvPageHeader {
            page_id,
            token_start,
            token_count: token_count as u16,
            layer_id: layer.0,
            kv_head_count: KV_HEADS as u8,
            head_dim: HEAD_DIM as u16,
            key_encoding: precision.encoding_id(),
            value_encoding: precision.encoding_id(),
            search_encoding: 0,
            flags: 0,
            key_payload_bytes: key_bytes.len() as u32,
            value_payload_bytes: value_bytes.len() as u32,
            quant_meta_bytes,
            outlier_bytes: 0,
            search_bytes: 0,
            backing_generation: 0,
            key_payload_offset: 0,
            value_payload_offset: key_bytes.len() as u64,
            quant_meta_offset: (key_bytes.len() + value_bytes.len()) as u64,
            outlier_offset: 0,
            search_offset: 0,
            content_hash,
        };

        Self {
            header,
            key_bytes,
            key_scales,
            value_bytes,
            value_scales,
            precision,
        }
    }

    fn token_count(&self) -> usize {
        self.header.token_count as usize
    }

    /// Content-addressed identity (spec §66's "immutable page IDs"): the
    /// same BLAKE3 hash already computed at seal time and checked by
    /// `verify`, reused directly rather than hashed twice.
    pub(crate) fn content_id(&self) -> [u8; 32] {
        self.header.content_hash
    }

    /// Verifies the sealed payload against its header's content hash —
    /// the section-156 "canonical page bytes are immutable" guarantee made
    /// checkable, mirroring `format::tqf`'s per-tile BLAKE3 checksums.
    fn verify(&self) -> Result<()> {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&self.key_bytes);
        hasher.update(&self.value_bytes);
        for s in &self.key_scales {
            hasher.update(&s.to_le_bytes());
        }
        for s in &self.value_scales {
            hasher.update(&s.to_le_bytes());
        }
        let actual: [u8; 32] = *hasher.finalize().as_bytes();
        if actual != self.header.content_hash {
            return Err(ModelError::Shape {
                tensor: "TQKV page content hash",
                expected: 0,
                actual: 1,
            }
            .into());
        }
        Ok(())
    }

    fn decode_key(&self, local_token: usize, kv_head: usize) -> [f32; HEAD_DIM] {
        let mut out = [0f32; HEAD_DIM];
        let codes = unpack_codes(
            &self.key_bytes,
            (local_token * KV_HEADS + kv_head) * HEAD_DIM,
            HEAD_DIM,
            self.precision,
        );
        for dim in 0..HEAD_DIM {
            let scale = self.key_scales[kv_head * HEAD_DIM + dim].to_f32();
            out[dim] = codes[dim] as f32 * scale;
        }
        out
    }

    fn decode_value(&self, local_token: usize, kv_head: usize) -> [f32; HEAD_DIM] {
        let mut out = [0f32; HEAD_DIM];
        let codes = unpack_codes(
            &self.value_bytes,
            (local_token * KV_HEADS + kv_head) * HEAD_DIM,
            HEAD_DIM,
            self.precision,
        );
        for group in 0..VALUE_GROUPS {
            let scale =
                self.value_scales[(local_token * KV_HEADS + kv_head) * VALUE_GROUPS + group]
                    .to_f32();
            for dim in group * VALUE_GROUP..(group + 1) * VALUE_GROUP {
                out[dim] = codes[dim] as f32 * scale;
            }
        }
        out
    }

    /// Serializes to the exact layout the header's own offsets describe:
    /// `[header][key_payload][value_payload][key_scales][value_scales]`.
    /// Used by `context::prefix` to persist sealed pages as content-addressed
    /// blobs (spec §66: "Snapshots reference immutable page IDs").
    pub(crate) fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.resident_bytes());
        out.extend_from_slice(&self.header.to_le_bytes());
        out.extend_from_slice(&self.key_bytes);
        out.extend_from_slice(&self.value_bytes);
        for s in &self.key_scales {
            out.extend_from_slice(&s.to_le_bytes());
        }
        for s in &self.value_scales {
            out.extend_from_slice(&s.to_le_bytes());
        }
        out
    }

    pub(crate) fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < PAGE_HEADER_BYTES {
            return Err(ModelError::Shape {
                tensor: "TQKV persisted page",
                expected: PAGE_HEADER_BYTES,
                actual: bytes.len(),
            }
            .into());
        }
        let mut header_bytes = [0u8; PAGE_HEADER_BYTES];
        header_bytes.copy_from_slice(&bytes[..PAGE_HEADER_BYTES]);
        let header = TqkvPageHeader::from_le_bytes(&header_bytes);
        let precision = TqkvPrecision::from_encoding_id(header.key_encoding)?;
        let body = &bytes[PAGE_HEADER_BYTES..];
        let key_len = header.key_payload_bytes as usize;
        let value_len = header.value_payload_bytes as usize;
        let key_bytes = body
            .get(..key_len)
            .ok_or(ModelError::Shape {
                tensor: "TQKV persisted key payload",
                expected: key_len,
                actual: body.len(),
            })?
            .to_vec();
        let value_bytes = body
            .get(key_len..key_len + value_len)
            .ok_or(ModelError::Shape {
                tensor: "TQKV persisted value payload",
                expected: value_len,
                actual: body.len().saturating_sub(key_len),
            })?
            .to_vec();
        let scales_region = &body[key_len + value_len..];
        let key_scale_count = KV_HEADS * HEAD_DIM;
        let value_scale_count = header.token_count as usize * KV_HEADS * VALUE_GROUPS;
        let expected_scale_bytes = (key_scale_count + value_scale_count) * std::mem::size_of::<f16>();
        if scales_region.len() < expected_scale_bytes {
            return Err(ModelError::Shape {
                tensor: "TQKV persisted scale metadata",
                expected: expected_scale_bytes,
                actual: scales_region.len(),
            }
            .into());
        }
        let key_scales = scales_region[..key_scale_count * 2]
            .chunks_exact(2)
            .map(|c| f16::from_le_bytes([c[0], c[1]]))
            .collect();
        let value_scales_region = &scales_region[key_scale_count * 2..];
        let value_scales = value_scales_region[..value_scale_count * 2]
            .chunks_exact(2)
            .map(|c| f16::from_le_bytes([c[0], c[1]]))
            .collect();
        Ok(Self {
            header,
            key_bytes,
            key_scales,
            value_bytes,
            value_scales,
            precision,
        })
    }

    /// Total resident bytes: header + payloads + scale metadata (spec
    /// section 157's on-disk/in-memory backing accounting).
    fn resident_bytes(&self) -> usize {
        PAGE_HEADER_BYTES
            + self.key_bytes.len()
            + self.value_bytes.len()
            + self.key_scales.len() * std::mem::size_of::<f16>()
            + self.value_scales.len() * std::mem::size_of::<f16>()
    }
}

/// Packs `count` per-(token,head) codes starting at `offset` within a flat
/// `[token][kv_head][dim]` code array into the page's on-disk representation.
fn pack_codes(codes: &[i8], precision: TqkvPrecision) -> Vec<u8> {
    match precision {
        TqkvPrecision::Q8 => codes.iter().map(|&c| c as u8).collect(),
        TqkvPrecision::Q4 => {
            let mut out = Vec::with_capacity(codes.len().div_ceil(2));
            for pair in codes.chunks(2) {
                let lo = (pair[0] & 0x0f) as u8;
                let hi = if pair.len() == 2 { (pair[1] & 0x0f) as u8 } else { 0 };
                out.push(lo | (hi << 4));
            }
            out
        }
    }
}

fn unpack_codes(
    bytes: &[u8],
    offset: usize,
    count: usize,
    precision: TqkvPrecision,
) -> Vec<i8> {
    match precision {
        TqkvPrecision::Q8 => bytes[offset..offset + count].iter().map(|&b| b as i8).collect(),
        TqkvPrecision::Q4 => {
            let mut out = Vec::with_capacity(count);
            for i in 0..count {
                let index = offset + i;
                let byte = bytes[index / 2];
                let nibble = if index.is_multiple_of(2) { byte & 0x0f } else { byte >> 4 };
                // Sign-extend the low 4 bits.
                out.push(((nibble << 4) as i8) >> 4);
            }
            out
        }
    }
}

fn checked_len(name: &'static str, actual: usize, expected: usize) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(ModelError::Shape {
            tensor: name,
            expected,
            actual,
        }
        .into())
    }
}

/// Key-only Q4 encode/decode over a whole page, exposed for the Phase 28
/// candidate-matrix comparison harness (`candidates::tests`) so every
/// RESEARCH CANDIDATE is measured against the same Phase 27 baseline
/// (bytes and error) rather than an approximation of it.
#[cfg(test)]
pub(crate) fn q4_key_baseline(keys: &[f32], token_count: usize) -> (usize, f32) {
    let mut scales = vec![0f32; KV_HEADS * HEAD_DIM];
    for token in 0..token_count {
        for head in 0..KV_HEADS {
            let base = (token * KV_HEADS + head) * HEAD_DIM;
            for dim in 0..HEAD_DIM {
                let v = keys[base + dim].abs();
                let slot = head * HEAD_DIM + dim;
                if v > scales[slot] {
                    scales[slot] = v;
                }
            }
        }
    }
    let clamp = TqkvPrecision::Q4.clamp_abs();
    for scale in scales.iter_mut() {
        *scale = if *scale > 0.0 { *scale / clamp } else { 1.0 };
    }
    let mut codes = vec![0i8; token_count * KV_WIDTH];
    for token in 0..token_count {
        for head in 0..KV_HEADS {
            let base = (token * KV_HEADS + head) * HEAD_DIM;
            for dim in 0..HEAD_DIM {
                let scale = scales[head * HEAD_DIM + dim];
                let q = (keys[base + dim] / scale).round().clamp(-clamp, clamp);
                codes[base + dim] = q as i8;
            }
        }
    }
    let packed = pack_codes(&codes, TqkvPrecision::Q4);
    let f16_scales: Vec<f16> = scales.iter().map(|&s| f16::from_f32(s)).collect();
    let bytes = packed.len() + f16_scales.len() * std::mem::size_of::<f16>();

    let mut max_err = 0f32;
    for token in 0..token_count {
        for head in 0..KV_HEADS {
            let local_codes = unpack_codes(&packed, (token * KV_HEADS + head) * HEAD_DIM, HEAD_DIM, TqkvPrecision::Q4);
            let base = (token * KV_HEADS + head) * HEAD_DIM;
            for dim in 0..HEAD_DIM {
                let decoded = local_codes[dim] as f32 * f16_scales[head * HEAD_DIM + dim].to_f32();
                max_err = max_err.max((decoded - keys[base + dim]).abs());
            }
        }
    }
    (bytes, max_err)
}

/// Paged, mixed-precision replacement for `Bf16KvCache` (spec sections
/// 155-159). Same external contract (`push`/`key`/`value`/`len`) so
/// `FullAttentionLayer` can select this backend transparently.
pub struct TqkvPagedCache {
    layer: LayerId,
    max_tokens: usize,
    precision: TqkvPrecision,
    sealed: Vec<SealedPage>,
    tail_keys: Vec<f32>,
    tail_values: Vec<f32>,
    next_page_id: u64,
    _lease: MemoryLease,
}

impl TqkvPagedCache {
    /// Upper-bound resident bytes for `max_tokens` at `precision`: every
    /// full page sealed at the target precision, plus one high-precision
    /// (f32) mutable tail page — this is what gets reserved from the broker
    /// before any physical allocation (crate invariant #4).
    pub fn bytes_for_tokens(max_tokens: usize, precision: TqkvPrecision) -> Result<Bytes> {
        let sealed_pages = max_tokens.div_ceil(PAGE_TOKENS).max(1);
        let key_payload = precision.packed_bytes(PAGE_TOKENS * KV_WIDTH);
        let value_payload = precision.packed_bytes(PAGE_TOKENS * KV_WIDTH);
        let key_scale_bytes = KV_HEADS * HEAD_DIM * std::mem::size_of::<f16>();
        let value_scale_bytes = PAGE_TOKENS * KV_HEADS * VALUE_GROUPS * std::mem::size_of::<f16>();
        let per_page = PAGE_HEADER_BYTES
            + key_payload
            + value_payload
            + key_scale_bytes
            + value_scale_bytes;
        let sealed_total = per_page
            .checked_mul(sealed_pages)
            .ok_or(ModelError::Shape {
                tensor: "TQKV sealed capacity",
                expected: usize::MAX,
                actual: sealed_pages,
            })?;
        let tail_total = PAGE_TOKENS
            .checked_mul(KV_WIDTH)
            .and_then(|n| n.checked_mul(2)) // key + value
            .and_then(|n| n.checked_mul(std::mem::size_of::<f32>()))
            .ok_or(ModelError::Shape {
                tensor: "TQKV tail capacity",
                expected: usize::MAX,
                actual: PAGE_TOKENS,
            })?;
        Ok(Bytes((sealed_total + tail_total) as u64))
    }

    pub fn new(
        broker: &MemoryBroker,
        layer: LayerId,
        max_tokens: usize,
        precision: TqkvPrecision,
    ) -> Result<Self> {
        if max_tokens == 0 {
            return Err(ModelError::Shape {
                tensor: "TQKV capacity",
                expected: 1,
                actual: 0,
            }
            .into());
        }
        let lease = broker.reserve(
            MemoryOwner::ContextCold,
            MemoryClass::Elastic,
            Self::bytes_for_tokens(max_tokens, precision)?,
            64,
        )?;
        Ok(Self {
            layer,
            max_tokens,
            precision,
            sealed: Vec::new(),
            tail_keys: Vec::new(),
            tail_values: Vec::new(),
            next_page_id: 0,
            _lease: lease,
        })
    }

    pub fn layer(&self) -> LayerId {
        self.layer
    }

    pub fn len(&self) -> usize {
        self.sealed.iter().map(SealedPage::token_count).sum::<usize>() + self.tail_len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn tail_len(&self) -> usize {
        self.tail_keys.len() / KV_WIDTH
    }

    pub fn precision(&self) -> TqkvPrecision {
        self.precision
    }

    /// Verifies every sealed page's content hash — a live analogue of
    /// `format::tqf`'s tile-checksum verified reads (spec section 156's
    /// immutability guarantee).
    pub fn verify_sealed_pages(&self) -> Result<()> {
        for page in &self.sealed {
            page.verify()?;
        }
        Ok(())
    }

    /// Bytes of quantized page payload actually resident right now (sealed
    /// pages only — the tail is accounted separately since it is still
    /// high-precision and not yet quantized).
    pub fn sealed_resident_bytes(&self) -> Bytes {
        Bytes(self.sealed.iter().map(SealedPage::resident_bytes).sum::<usize>() as u64)
    }

    /// Clears logical content while keeping the existing broker lease
    /// (same reserved capacity, matching `Bf16KvCache`'s reset behavior).
    pub fn reset(&mut self) {
        self.sealed.clear();
        self.tail_keys.clear();
        self.tail_values.clear();
        self.next_page_id = 0;
    }

    /// Sealed pages in order, for `context::prefix` snapshot export — each
    /// one's `content_id()`/`to_bytes()` is what gets persisted as a
    /// content-addressed blob.
    pub(crate) fn sealed_pages(&self) -> &[SealedPage] {
        &self.sealed
    }

    /// The mutable tail's raw post-RoPE Key/Value history — not
    /// content-addressed (it is not yet immutable, spec §156), so a
    /// snapshot stores it inline rather than by content ID.
    pub(crate) fn tail_raw(&self) -> (&[f32], &[f32]) {
        (&self.tail_keys, &self.tail_values)
    }

    /// Rebuilds cache content from a restored snapshot (sealed pages plus
    /// tail), keeping the existing broker lease/capacity — the
    /// `context::prefix` restart-reuse path.
    pub(crate) fn restore_from_snapshot(
        &mut self,
        sealed: Vec<SealedPage>,
        tail_keys: Vec<f32>,
        tail_values: Vec<f32>,
    ) {
        self.next_page_id = sealed.len() as u64;
        self.sealed = sealed;
        self.tail_keys = tail_keys;
        self.tail_values = tail_values;
    }

    pub(crate) fn push(&mut self, key: &[f32], value: &[f32]) -> Result<()> {
        checked_len("TQKV key", key.len(), KV_WIDTH)?;
        checked_len("TQKV value", value.len(), KV_WIDTH)?;
        if self.len() == self.max_tokens {
            return Err(ModelError::ContextCapacity {
                layer: self.layer.0,
                capacity: self.max_tokens,
            }
            .into());
        }
        self.tail_keys.extend_from_slice(key);
        self.tail_values.extend_from_slice(value);
        if self.tail_len() == PAGE_TOKENS {
            self.seal_tail();
        }
        Ok(())
    }

    fn seal_tail(&mut self) {
        let token_start = (self.len() - self.tail_len()) as u32;
        let page = SealedPage::seal(
            self.next_page_id,
            self.layer,
            token_start,
            self.tail_len(),
            &self.tail_keys,
            &self.tail_values,
            self.precision,
        );
        self.next_page_id += 1;
        self.sealed.push(page);
        self.tail_keys.clear();
        self.tail_values.clear();
    }

    pub(crate) fn key(&self, token: usize, kv_head: usize) -> [f32; HEAD_DIM] {
        let sealed_tokens = self.len() - self.tail_len();
        if token < sealed_tokens {
            let page_index = token / PAGE_TOKENS;
            let local = token % PAGE_TOKENS;
            self.sealed[page_index].decode_key(local, kv_head)
        } else {
            let local = token - sealed_tokens;
            let base = (local * KV_HEADS + kv_head) * HEAD_DIM;
            let mut out = [0f32; HEAD_DIM];
            out.copy_from_slice(&self.tail_keys[base..base + HEAD_DIM]);
            out
        }
    }

    pub(crate) fn value(&self, token: usize, kv_head: usize) -> [f32; HEAD_DIM] {
        let sealed_tokens = self.len() - self.tail_len();
        if token < sealed_tokens {
            let page_index = token / PAGE_TOKENS;
            let local = token % PAGE_TOKENS;
            self.sealed[page_index].decode_value(local, kv_head)
        } else {
            let local = token - sealed_tokens;
            let base = (local * KV_HEADS + kv_head) * HEAD_DIM;
            let mut out = [0f32; HEAD_DIM];
            out.copy_from_slice(&self.tail_values[base..base + HEAD_DIM]);
            out
        }
    }
}

/// Reads `TQF_TQKV_ENABLED` once (spec invariant #10: every optimization is
/// disableable for A/B, off by default like Phase 23's prefetch). The BF16
/// reference cache remains the default production backend.
pub fn tqkv_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("TQF_TQKV_ENABLED").ok().as_deref(),
            Some("1") | Some("true") | Some("on")
        )
    })
}

/// Reads `TQF_TQKV_PRECISION` once; defaults to Q8, the section-158
/// "first compressed oracle beneath BF16".
pub fn tqkv_precision() -> TqkvPrecision {
    static PRECISION: std::sync::OnceLock<TqkvPrecision> = std::sync::OnceLock::new();
    *PRECISION.get_or_init(|| match std::env::var("TQF_TQKV_PRECISION").ok().as_deref() {
        Some("q4") | Some("Q4") => TqkvPrecision::Q4,
        _ => TqkvPrecision::Q8,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn broker() -> MemoryBroker {
        MemoryBroker::new(Bytes(64 * 1024 * 1024))
    }

    fn synthetic_kv(tokens: usize, seed: u64) -> (Vec<f32>, Vec<f32>) {
        // Deterministic pseudo-random generator (no external dependency):
        // xorshift64, scaled into a realistic post-RoPE activation range.
        let mut state = seed | 1;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state as f64 / u64::MAX as f64) * 2.0 - 1.0) as f32 * 4.0
        };
        let keys: Vec<f32> = (0..tokens * KV_WIDTH).map(|_| next()).collect();
        let values: Vec<f32> = (0..tokens * KV_WIDTH).map(|_| next()).collect();
        (keys, values)
    }

    #[test]
    fn header_round_trips_through_le_bytes() {
        let header = TqkvPageHeader {
            page_id: 7,
            token_start: 256,
            token_count: 200,
            layer_id: 3,
            kv_head_count: 2,
            head_dim: 256,
            key_encoding: 1,
            value_encoding: 1,
            search_encoding: 0,
            flags: 0,
            key_payload_bytes: 1000,
            value_payload_bytes: 2000,
            quant_meta_bytes: 300,
            outlier_bytes: 0,
            search_bytes: 0,
            backing_generation: 0,
            key_payload_offset: 0,
            value_payload_offset: 1000,
            quant_meta_offset: 3000,
            outlier_offset: 0,
            search_offset: 0,
            content_hash: [9u8; 32],
        };
        let bytes = header.to_le_bytes();
        assert_eq!(bytes.len(), PAGE_HEADER_BYTES);
        assert_eq!(TqkvPageHeader::from_le_bytes(&bytes), header);
    }

    #[test]
    fn precision_encoding_ids_round_trip() {
        assert_eq!(
            TqkvPrecision::from_encoding_id(TqkvPrecision::Q8.encoding_id()).unwrap(),
            TqkvPrecision::Q8
        );
        assert_eq!(
            TqkvPrecision::from_encoding_id(TqkvPrecision::Q4.encoding_id()).unwrap(),
            TqkvPrecision::Q4
        );
        assert!(TqkvPrecision::from_encoding_id(99).is_err());
    }

    #[test]
    fn cache_reserves_bytes_before_allocating_and_releases_on_drop() {
        let broker = broker();
        let cache =
            TqkvPagedCache::new(&broker, LayerId(3), 512, TqkvPrecision::Q8).unwrap();
        assert_eq!(
            broker.snapshot().reserved,
            TqkvPagedCache::bytes_for_tokens(512, TqkvPrecision::Q8).unwrap()
        );
        drop(cache);
        assert_eq!(broker.snapshot().reserved, Bytes(0));
    }

    #[test]
    fn q8_is_smaller_than_bf16_and_q4_is_smaller_than_q8_at_128k() {
        let tokens = 131_072;
        let bf16 = crate::model::qwen36::attention::Bf16KvCache::bytes_for_tokens(tokens).unwrap();
        let q8 = TqkvPagedCache::bytes_for_tokens(tokens, TqkvPrecision::Q8).unwrap();
        let q4 = TqkvPagedCache::bytes_for_tokens(tokens, TqkvPrecision::Q4).unwrap();
        println!(
            "phase27_capacity tokens={tokens} bf16_bytes_per_layer={} q8_bytes_per_layer={} q4_bytes_per_layer={} bf16_gib_10layer={:.4} q8_gib_10layer={:.4} q4_gib_10layer={:.4}",
            bf16.0, q8.0, q4.0,
            (bf16.0 as f64 * 10.0) / (1024.0*1024.0*1024.0),
            (q8.0 as f64 * 10.0) / (1024.0*1024.0*1024.0),
            (q4.0 as f64 * 10.0) / (1024.0*1024.0*1024.0),
        );
        assert!(q8.0 < bf16.0, "Q8 {} should be < BF16 {}", q8.0, bf16.0);
        assert!(q4.0 < q8.0, "Q4 {} should be < Q8 {}", q4.0, q8.0);
    }

    #[test]
    fn push_seals_a_page_exactly_at_the_boundary_and_verifies() {
        let broker = broker();
        let mut cache =
            TqkvPagedCache::new(&broker, LayerId(1), 1024, TqkvPrecision::Q8).unwrap();
        let (keys, values) = synthetic_kv(PAGE_TOKENS + 5, 42);
        for token in 0..PAGE_TOKENS + 5 {
            let base = token * KV_WIDTH;
            cache
                .push(&keys[base..base + KV_WIDTH], &values[base..base + KV_WIDTH])
                .unwrap();
        }
        assert_eq!(cache.len(), PAGE_TOKENS + 5);
        assert_eq!(cache.sealed.len(), 1);
        assert_eq!(cache.tail_len(), 5);
        cache.verify_sealed_pages().unwrap();
    }

    #[test]
    fn capacity_error_matches_bf16_cache_semantics() {
        let broker = broker();
        let mut cache = TqkvPagedCache::new(&broker, LayerId(1), 2, TqkvPrecision::Q8).unwrap();
        let (keys, values) = synthetic_kv(3, 7);
        for token in 0..2 {
            let base = token * KV_WIDTH;
            cache
                .push(&keys[base..base + KV_WIDTH], &values[base..base + KV_WIDTH])
                .unwrap();
        }
        let base = 2 * KV_WIDTH;
        assert!(cache
            .push(&keys[base..base + KV_WIDTH], &values[base..base + KV_WIDTH])
            .is_err());
    }

    /// Section 158's stated Q8 error model: symmetric per-channel/per-group
    /// quantization from a real (non-adversarial) activation distribution
    /// should recover values to a small fraction of the dynamic range.
    #[test]
    fn q8_round_trip_error_is_small_relative_to_dynamic_range() {
        let broker = broker();
        let mut cache =
            TqkvPagedCache::new(&broker, LayerId(0), PAGE_TOKENS, TqkvPrecision::Q8).unwrap();
        let (keys, values) = synthetic_kv(PAGE_TOKENS, 1234);
        for token in 0..PAGE_TOKENS {
            let base = token * KV_WIDTH;
            cache
                .push(&keys[base..base + KV_WIDTH], &values[base..base + KV_WIDTH])
                .unwrap();
        }
        let mut max_err = 0f32;
        for token in 0..PAGE_TOKENS {
            for head in 0..KV_HEADS {
                let decoded = cache.key(token, head);
                let base = (token * KV_HEADS + head) * HEAD_DIM;
                for dim in 0..HEAD_DIM {
                    max_err = max_err.max((decoded[dim] - keys[base + dim]).abs());
                }
            }
        }
        // Values are U(-4,4); Q8 step is ~4/127 ~= 0.0315, half-step error bound ~0.016.
        assert!(max_err < 0.05, "Q8 key max abs error too large: {max_err}");
    }

    #[test]
    fn q4_round_trip_error_is_larger_than_q8_but_bounded() {
        let broker = broker();
        let mut cache =
            TqkvPagedCache::new(&broker, LayerId(0), PAGE_TOKENS, TqkvPrecision::Q4).unwrap();
        let (keys, values) = synthetic_kv(PAGE_TOKENS, 999);
        for token in 0..PAGE_TOKENS {
            let base = token * KV_WIDTH;
            cache
                .push(&keys[base..base + KV_WIDTH], &values[base..base + KV_WIDTH])
                .unwrap();
        }
        let mut max_err = 0f32;
        for token in 0..PAGE_TOKENS {
            for head in 0..KV_HEADS {
                let decoded = cache.value(token, head);
                let base = (token * KV_HEADS + head) * HEAD_DIM;
                for dim in 0..HEAD_DIM {
                    max_err = max_err.max((decoded[dim] - values[base + dim]).abs());
                }
            }
        }
        // Step is ~4/7 ~= 0.57, half-step bound ~0.29; loose ceiling here.
        assert!(max_err < 0.5, "Q4 value max abs error too large: {max_err}");
    }

    #[test]
    fn tail_tokens_decode_exactly_at_full_precision() {
        let broker = broker();
        let mut cache =
            TqkvPagedCache::new(&broker, LayerId(0), 16, TqkvPrecision::Q8).unwrap();
        let (keys, values) = synthetic_kv(3, 55);
        for token in 0..3 {
            let base = token * KV_WIDTH;
            cache
                .push(&keys[base..base + KV_WIDTH], &values[base..base + KV_WIDTH])
                .unwrap();
        }
        for token in 0..3 {
            for head in 0..KV_HEADS {
                let k = cache.key(token, head);
                let base = (token * KV_HEADS + head) * HEAD_DIM;
                assert_eq!(&k[..], &keys[base..base + HEAD_DIM]);
            }
        }
    }

    #[test]
    fn corrupted_page_bytes_fail_verification() {
        let broker = broker();
        let mut cache =
            TqkvPagedCache::new(&broker, LayerId(0), PAGE_TOKENS, TqkvPrecision::Q8).unwrap();
        let (keys, values) = synthetic_kv(PAGE_TOKENS, 3);
        for token in 0..PAGE_TOKENS {
            let base = token * KV_WIDTH;
            cache
                .push(&keys[base..base + KV_WIDTH], &values[base..base + KV_WIDTH])
                .unwrap();
        }
        cache.sealed[0].key_bytes[0] ^= 0xff;
        assert!(cache.verify_sealed_pages().is_err());
    }
}
