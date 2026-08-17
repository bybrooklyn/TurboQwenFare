//! Fixed-graph weight manifest for the one supported Qwen3.6 checkpoint.
//! This is deliberately not a generic model loader: it validates that a
//! `.tqf` container has every tensor role required by Qwen3.6's exact
//! 30-GDN/10-full-attention MoE graph before the runtime is allowed to bind
//! any weights. It owns only validated TQF metadata/file handles; individual
//! payloads remain checksum-validated when a kernel reads them.

use std::path::Path;

use crate::dev::inventory::TensorRole;
use crate::error::{ModelError, Result};
use crate::format::quant::repack::{ggml_type_for_quant_layout, TQF_QUANT_PASSTHROUGH_Q4_K};
use crate::format::tqf::{canonical_header, ExpertMatrix, TqfReader};
use crate::format::{quant::dequant::dequantize_block, quant::GgmlType};
use crate::ids::{Bytes, LayerId, LayerKind};
use crate::memory::{MemoryBroker, MemoryClass, MemoryLease, MemoryOwner};

use super::geometry::Qwen36Geometry;

fn expected_elements(role: TensorRole) -> Option<usize> {
    let hidden = Qwen36Geometry::HIDDEN_SIZE;
    Some(match role {
        TensorRole::TokenEmbedding | TensorRole::LmHead => Qwen36Geometry::VOCAB_SIZE * hidden,
        TensorRole::FinalNorm | TensorRole::AttnNorm | TensorRole::FfnNorm => hidden,
        TensorRole::AttnQProj => {
            hidden * Qwen36Geometry::FULL_ATTENTION_HEADS * Qwen36Geometry::FULL_HEAD_DIM * 2
        }
        TensorRole::AttnKProj | TensorRole::AttnVProj => {
            hidden * Qwen36Geometry::FULL_KV_HEADS * Qwen36Geometry::FULL_HEAD_DIM
        }
        TensorRole::AttnOProj => {
            hidden * Qwen36Geometry::FULL_ATTENTION_HEADS * Qwen36Geometry::FULL_HEAD_DIM
        }
        TensorRole::AttnQNorm | TensorRole::AttnKNorm => Qwen36Geometry::FULL_HEAD_DIM,
        TensorRole::GdnInProjQkv => hidden * Qwen36Geometry::GDN_CONV_CHANNELS,
        TensorRole::GdnInProjZ => hidden * Qwen36Geometry::GDN_VALUE_DIM,
        TensorRole::GdnInProjA | TensorRole::GdnInProjB => {
            Qwen36Geometry::GDN_KEY_DIM * Qwen36Geometry::GDN_VALUE_HEADS
        }
        TensorRole::GdnConv1d => Qwen36Geometry::GDN_CONV_CHANNELS * Qwen36Geometry::GDN_CONV_WIDTH,
        TensorRole::GdnALog | TensorRole::GdnDtBias => Qwen36Geometry::GDN_VALUE_HEADS,
        TensorRole::GdnGatedNorm => Qwen36Geometry::GDN_VALUE_HEAD_DIM,
        TensorRole::GdnOutProj => Qwen36Geometry::GDN_VALUE_DIM * hidden,
        TensorRole::RouterGate => hidden * Qwen36Geometry::NUM_EXPERTS,
        TensorRole::SharedExpertInputGate => hidden,
        TensorRole::SharedExpertGate
        | TensorRole::SharedExpertUp
        | TensorRole::SharedExpertDown => hidden * Qwen36Geometry::SHARED_EXPERT_WIDTH,
        TensorRole::RoutedExpertGate
        | TensorRole::RoutedExpertUp
        | TensorRole::RoutedExpertDown => {
            hidden * Qwen36Geometry::ROUTED_EXPERT_WIDTH * Qwen36Geometry::NUM_EXPERTS
        }
        TensorRole::VisionProjector => return None,
    })
}

/// Canonical GGUF dimension order. GGUF stores matrix dimensions as
/// `{input_columns, output_rows}`; preserving that order here catches a
/// transposed payload before it reaches a GEMV kernel.
fn expected_shape(role: TensorRole) -> Option<&'static [u64]> {
    Some(match role {
        TensorRole::TokenEmbedding | TensorRole::LmHead => &[2048, 248_320],
        TensorRole::FinalNorm | TensorRole::AttnNorm | TensorRole::FfnNorm => &[2048],
        TensorRole::AttnQProj | TensorRole::GdnInProjQkv => &[2048, 8192],
        TensorRole::AttnKProj | TensorRole::AttnVProj => &[2048, 512],
        TensorRole::AttnOProj | TensorRole::GdnOutProj => &[4096, 2048],
        TensorRole::AttnQNorm | TensorRole::AttnKNorm => &[256],
        TensorRole::GdnInProjZ => &[2048, 4096],
        TensorRole::GdnInProjA | TensorRole::GdnInProjB => &[2048, 32],
        TensorRole::GdnConv1d => &[4, 8192],
        TensorRole::GdnALog | TensorRole::GdnDtBias => &[32],
        TensorRole::GdnGatedNorm => &[128],
        TensorRole::RouterGate => &[2048, 256],
        // llama.cpp canonicalizes a one-row matrix to a rank-1 GGUF tensor.
        TensorRole::SharedExpertInputGate => &[2048],
        TensorRole::SharedExpertGate | TensorRole::SharedExpertUp => &[2048, 512],
        TensorRole::SharedExpertDown => &[512, 2048],
        TensorRole::RoutedExpertGate | TensorRole::RoutedExpertUp => &[2048, 512, 256],
        TensorRole::RoutedExpertDown => &[512, 2048, 256],
        TensorRole::VisionProjector => return None,
    })
}

pub struct Qwen36WeightManifest {
    reader: TqfReader,
}

/// A broker-accounted raw TQF extent. Quantization-specific code consumes
/// `bytes`; the lease stays alive until those bytes are dropped, satisfying
/// the reserve-before-allocation rule for reference and future GPU paths.
pub struct LoadedQwen36Tensor {
    pub role: TensorRole,
    pub layer: Option<LayerId>,
    /// GGML wire type preserved by the lossless TQF conversion.
    pub dtype: GgmlType,
    /// Canonical GGUF dimension order, with matrix shape `{columns, rows}`.
    pub dims: Vec<u64>,
    pub bytes: Vec<u8>,
    _lease: MemoryLease,
}

/// A single routed SwiGLU expert, loaded from one checksummed TQF
/// superextent. Its backing lease is deliberately private: evicting the
/// cache entry is the only way to release its broker reservation.
pub struct LoadedQwen36Expert {
    pub layer: LayerId,
    pub expert: crate::ids::ExpertId,
    bytes: Vec<u8>,
    _lease: MemoryLease,
}

/// Broker-accounted activation produced by the Qwen reference matvec path.
/// Keeping the lease next to its values prevents a seemingly harmless CPU
/// fallback from bypassing the hard `--memory` budget.
pub struct Qwen36Activation {
    pub values: Vec<f32>,
    _lease: MemoryLease,
}

impl Qwen36Activation {
    /// Allocates an f32 activation only after reserving it with the broker.
    pub fn zeros(broker: &MemoryBroker, elements: usize) -> Result<Self> {
        let bytes = elements
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or(ModelError::Shape {
                tensor: "Qwen activation bytes",
                expected: usize::MAX,
                actual: elements,
            })?;
        let lease = broker.reserve(
            MemoryOwner::Scratch,
            MemoryClass::Transient,
            Bytes(bytes as u64),
            64,
        )?;
        Ok(Self {
            values: vec![0.0; elements],
            _lease: lease,
        })
    }

    /// Copies a bounded intermediate slice into a separately leased
    /// activation.  Fixed-graph operators use this at real tensor boundaries
    /// (for example QKV splitting); a borrowed slice is never allowed to
    /// outlive the larger activation that owns it.
    pub fn from_slice(broker: &MemoryBroker, values: &[f32]) -> Result<Self> {
        let mut output = Self::zeros(broker, values.len())?;
        output.values.copy_from_slice(values);
        Ok(output)
    }

    pub fn residual_add(
        broker: &MemoryBroker,
        left: &Qwen36Activation,
        right: &Qwen36Activation,
    ) -> Result<Self> {
        if left.values.len() != right.values.len() {
            return Err(ModelError::Shape {
                tensor: "Qwen residual add",
                expected: left.values.len(),
                actual: right.values.len(),
            }
            .into());
        }
        let mut output = Self::zeros(broker, left.values.len())?;
        for ((output, left), right) in output
            .values
            .iter_mut()
            .zip(&left.values)
            .zip(&right.values)
        {
            *output = left + right;
        }
        Ok(output)
    }

    /// Standard RMSNorm over canonical GGUF bytes. The upstream GGUF
    /// converter has already folded Qwen's zero-centered source parameter
    /// into the stored scale (`stored = 1 + source_weight`), so applying the
    /// residual one again here would corrupt every layer.
    pub fn qwen_rmsnorm(
        broker: &MemoryBroker,
        input: &Qwen36Activation,
        weight: &[f32],
    ) -> Result<Self> {
        if input.values.len() != weight.len() {
            return Err(ModelError::Shape {
                tensor: "Qwen RMSNorm weight",
                expected: input.values.len(),
                actual: weight.len(),
            }
            .into());
        }
        let inv_rms = 1.0
            / (input.values.iter().map(|value| value * value).sum::<f32>()
                / input.values.len() as f32
                + 1e-6)
                .sqrt();
        let mut output = Self::zeros(broker, input.values.len())?;
        for ((output, input), weight) in output.values.iter_mut().zip(&input.values).zip(weight) {
            *output = input * inv_rms * weight;
        }
        Ok(output)
    }

    pub fn silu_mul(
        broker: &MemoryBroker,
        gate: &Qwen36Activation,
        up: &Qwen36Activation,
    ) -> Result<Self> {
        if gate.values.len() != up.values.len() {
            return Err(ModelError::Shape {
                tensor: "Qwen SwiGLU inputs",
                expected: gate.values.len(),
                actual: up.values.len(),
            }
            .into());
        }
        let mut output = Self::zeros(broker, gate.values.len())?;
        for ((output, gate), up) in output.values.iter_mut().zip(&gate.values).zip(&up.values) {
            *output = (gate / (1.0 + (-gate).exp())) * up;
        }
        Ok(output)
    }

    pub fn scale_in_place(&mut self, scale: f32) {
        for value in &mut self.values {
            *value *= scale;
        }
    }

    pub fn add_scaled_in_place(&mut self, source: &Qwen36Activation, scale: f32) -> Result<()> {
        if self.values.len() != source.values.len() {
            return Err(ModelError::Shape {
                tensor: "Qwen scaled add",
                expected: self.values.len(),
                actual: source.values.len(),
            }
            .into());
        }
        for (target, source) in self.values.iter_mut().zip(&source.values) {
            *target += source * scale;
        }
        Ok(())
    }
}

impl LoadedQwen36Tensor {
    /// Decodes a rank-one Qwen parameter tensor (norm scale, router scalar,
    /// or GDN time parameter) into a broker-owned activation.
    pub fn vector(&self, broker: &MemoryBroker) -> Result<Qwen36Activation> {
        if self.dims.len() != 1 {
            return Err(ModelError::Unsupported(format!(
                "{} is rank {}, not a Qwen vector",
                self.role as u32,
                self.dims.len()
            ))
            .into());
        }
        let elements: usize = self.dims[0].try_into().map_err(|_| ModelError::Shape {
            tensor: "Qwen vector elements",
            expected: usize::MAX,
            actual: usize::MAX,
        })?;
        self.decode_values(broker, &self.bytes, elements)
    }

    /// Computes the scalar product of a stored rank-one parameter with one
    /// activation. This is the canonical representation of Qwen's shared
    /// expert input gate; it is not a synthetic one-row matrix.
    pub fn dot(&self, broker: &MemoryBroker, input: &[f32]) -> Result<f32> {
        if self.dims.len() != 1 {
            return Err(ModelError::Unsupported(format!(
                "{} is rank {}, not a Qwen dot-product vector",
                self.role as u32,
                self.dims.len()
            ))
            .into());
        }
        let cols: usize = self.dims[0].try_into().map_err(|_| ModelError::Shape {
            tensor: "Qwen dot-product vector elements",
            expected: usize::MAX,
            actual: usize::MAX,
        })?;
        let output = self.matvec_payload(broker, &self.bytes, 1, cols, input)?;
        Ok(output.values[0])
    }

    /// Decodes the canonical depthwise GDN convolution weights into their
    /// channel-major flat layout. GGUF stores this logical
    /// `[channels, width]` tensor as `{width, channels}`, so it is rank two
    /// even though the streaming convolution consumes one contiguous slice.
    pub fn gdn_conv1d_weights(&self, broker: &MemoryBroker) -> Result<Qwen36Activation> {
        let expected = expected_shape(TensorRole::GdnConv1d)
            .expect("the fixed Qwen graph always defines GDN convolution dimensions");
        if self.role != TensorRole::GdnConv1d || self.dims.as_slice() != expected {
            return Err(ModelError::Unsupported(format!(
                "role {:?} with dimensions {:?} is not canonical Qwen GDN convolution storage {:?}",
                self.role, self.dims, expected
            ))
            .into());
        }
        let elements = expected_elements(TensorRole::GdnConv1d)
            .expect("the fixed Qwen graph always defines GDN convolution elements");
        self.decode_values(broker, &self.bytes, elements)
    }

    /// Reads one logical row from a rank-two Qwen tensor. This is used for
    /// token embedding lookup and deliberately does not materialize the
    /// whole embedding matrix.
    pub fn row(&self, broker: &MemoryBroker, row: usize) -> Result<Qwen36Activation> {
        let (rows, cols) = self.matrix_shape()?;
        if row >= rows {
            return Err(ModelError::Shape {
                tensor: "Qwen matrix row index",
                expected: rows,
                actual: row,
            }
            .into());
        }
        let row_bytes: usize =
            self.dtype
                .byte_size(cols as u64)?
                .try_into()
                .map_err(|_| ModelError::Shape {
                    tensor: "Qwen matrix row bytes",
                    expected: usize::MAX,
                    actual: usize::MAX,
                })?;
        let offset = row.checked_mul(row_bytes).ok_or(ModelError::Shape {
            tensor: "Qwen matrix row offset",
            expected: usize::MAX,
            actual: row,
        })?;
        let payload = self
            .bytes
            .get(offset..offset + row_bytes)
            .ok_or(ModelError::Shape {
                tensor: "Qwen matrix row payload",
                expected: row_bytes,
                actual: self.bytes.len().saturating_sub(offset),
            })?;
        self.decode_values(broker, payload, cols)
    }

    fn matrix_shape(&self) -> Result<(usize, usize)> {
        if self.dims.len() != 2 {
            return Err(ModelError::Unsupported(format!(
                "{} is rank {}, not a Qwen matrix",
                self.role as u32,
                self.dims.len()
            ))
            .into());
        }
        let cols = self.dims[0].try_into().map_err(|_| ModelError::Shape {
            tensor: "Qwen matrix columns",
            expected: usize::MAX,
            actual: usize::MAX,
        })?;
        let rows = self.dims[1].try_into().map_err(|_| ModelError::Shape {
            tensor: "Qwen matrix rows",
            expected: usize::MAX,
            actual: usize::MAX,
        })?;
        Ok((rows, cols))
    }

    /// Reference matrix-vector multiply over an already loaded Qwen extent.
    /// TQF keeps GGUF's `{columns, rows}` dimension order; the stored payload
    /// is consequently traversed as `rows` contiguous logical rows. Only the
    /// canonical Qwen dtypes are accepted, never reinterpreted by guesswork.
    pub fn matvec(&self, broker: &MemoryBroker, input: &[f32]) -> Result<Qwen36Activation> {
        let (rows, cols) = self.matrix_shape()?;
        self.matvec_payload(broker, &self.bytes, rows, cols, input)
    }

    /// Matrix-vector multiply for one expert plane in Qwen's canonical
    /// `{columns, rows, expert}` routed tensor. The source extent remains a
    /// single broker-owned resident allocation; this method only allocates
    /// one selected expert's output activation.
    pub fn matvec_expert(
        &self,
        broker: &MemoryBroker,
        expert: usize,
        input: &[f32],
    ) -> Result<Qwen36Activation> {
        if self.dims.len() != 3 {
            return Err(ModelError::Unsupported(format!(
                "{} is rank {}, not a routed-expert tensor",
                self.role as u32,
                self.dims.len()
            ))
            .into());
        }
        let cols: usize = self.dims[0].try_into().map_err(|_| ModelError::Shape {
            tensor: "Qwen expert matrix columns",
            expected: usize::MAX,
            actual: usize::MAX,
        })?;
        let rows: usize = self.dims[1].try_into().map_err(|_| ModelError::Shape {
            tensor: "Qwen expert matrix rows",
            expected: usize::MAX,
            actual: usize::MAX,
        })?;
        let expert_count: usize = self.dims[2].try_into().map_err(|_| ModelError::Shape {
            tensor: "Qwen expert count",
            expected: usize::MAX,
            actual: usize::MAX,
        })?;
        if expert >= expert_count {
            return Err(ModelError::Shape {
                tensor: "Qwen routed expert index",
                expected: expert_count,
                actual: expert,
            }
            .into());
        }
        let matrix_elements = rows.checked_mul(cols).ok_or(ModelError::Shape {
            tensor: "Qwen expert matrix elements",
            expected: usize::MAX,
            actual: rows,
        })?;
        let matrix_bytes: usize = self
            .dtype
            .byte_size(matrix_elements as u64)?
            .try_into()
            .map_err(|_| ModelError::Shape {
                tensor: "Qwen expert matrix bytes",
                expected: usize::MAX,
                actual: usize::MAX,
            })?;
        let offset = expert.checked_mul(matrix_bytes).ok_or(ModelError::Shape {
            tensor: "Qwen expert matrix offset",
            expected: usize::MAX,
            actual: expert,
        })?;
        let payload = self
            .bytes
            .get(offset..offset + matrix_bytes)
            .ok_or(ModelError::Shape {
                tensor: "Qwen expert matrix payload",
                expected: matrix_bytes,
                actual: self.bytes.len().saturating_sub(offset),
            })?;
        self.matvec_payload(broker, payload, rows, cols, input)
    }

    fn matvec_payload(
        &self,
        broker: &MemoryBroker,
        payload: &[u8],
        rows: usize,
        cols: usize,
        input: &[f32],
    ) -> Result<Qwen36Activation> {
        matvec_payload(broker, self.dtype, payload, rows, cols, input)
    }

    fn decode_values(
        &self,
        broker: &MemoryBroker,
        payload: &[u8],
        elements: usize,
    ) -> Result<Qwen36Activation> {
        let expected_bytes: usize =
            self.dtype
                .byte_size(elements as u64)?
                .try_into()
                .map_err(|_| ModelError::Shape {
                    tensor: "Qwen decoded tensor bytes",
                    expected: usize::MAX,
                    actual: usize::MAX,
                })?;
        if payload.len() != expected_bytes {
            return Err(ModelError::Shape {
                tensor: "Qwen decoded tensor bytes",
                expected: expected_bytes,
                actual: payload.len(),
            }
            .into());
        }
        let output_bytes =
            elements
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or(ModelError::Shape {
                    tensor: "Qwen decoded activation bytes",
                    expected: usize::MAX,
                    actual: elements,
                })?;
        let lease = broker.reserve(
            MemoryOwner::Scratch,
            MemoryClass::Transient,
            Bytes(output_bytes as u64),
            64,
        )?;
        // The broker reservation deliberately precedes this Vec allocation.
        let values = decode_values(self.dtype, payload, elements)?;
        Ok(Qwen36Activation {
            values,
            _lease: lease,
        })
    }
}

impl LoadedQwen36Expert {
    const GATE_BYTES: usize = 589_824;
    const UP_BYTES: usize = 589_824;
    const DOWN_BYTES: usize = 589_824;

    /// Executes this Q4_K whole-expert payload in the canonical SwiGLU
    /// order. The cache retains the bytes; each activation remains a separate
    /// broker-accounted transient allocation.
    pub fn forward(&self, broker: &MemoryBroker, input: &[f32]) -> Result<Qwen36Activation> {
        let gate_end = Self::GATE_BYTES;
        let up_end = gate_end + Self::UP_BYTES;
        let down_end = up_end + Self::DOWN_BYTES;
        let gate = self.bytes.get(..gate_end).ok_or(ModelError::Shape {
            tensor: "Qwen expert gate payload",
            expected: gate_end,
            actual: self.bytes.len(),
        })?;
        let up = self.bytes.get(gate_end..up_end).ok_or(ModelError::Shape {
            tensor: "Qwen expert up payload",
            expected: up_end,
            actual: self.bytes.len(),
        })?;
        let down = self.bytes.get(up_end..down_end).ok_or(ModelError::Shape {
            tensor: "Qwen expert down payload",
            expected: down_end,
            actual: self.bytes.len(),
        })?;
        let gate = matvec_payload(
            broker,
            GgmlType::Q4K,
            gate,
            Qwen36Geometry::ROUTED_EXPERT_WIDTH,
            Qwen36Geometry::HIDDEN_SIZE,
            input,
        )?;
        let up = matvec_payload(
            broker,
            GgmlType::Q4K,
            up,
            Qwen36Geometry::ROUTED_EXPERT_WIDTH,
            Qwen36Geometry::HIDDEN_SIZE,
            input,
        )?;
        let hidden = Qwen36Activation::silu_mul(broker, &gate, &up)?;
        matvec_payload(
            broker,
            GgmlType::Q4K,
            down,
            Qwen36Geometry::HIDDEN_SIZE,
            Qwen36Geometry::ROUTED_EXPERT_WIDTH,
            &hidden.values,
        )
    }

    pub fn stored_bytes(&self) -> Bytes {
        Bytes(self.bytes.len() as u64)
    }
}

fn matvec_payload(
    broker: &MemoryBroker,
    dtype: GgmlType,
    payload: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Result<Qwen36Activation> {
    if input.len() != cols {
        return Err(ModelError::Shape {
            tensor: "Qwen matvec input",
            expected: cols,
            actual: input.len(),
        }
        .into());
    }
    let logical_elements = rows.checked_mul(cols).ok_or(ModelError::Shape {
        tensor: "Qwen matvec elements",
        expected: usize::MAX,
        actual: rows,
    })?;
    let expected_bytes: usize = dtype
        .byte_size(logical_elements as u64)?
        .try_into()
        .map_err(|_| ModelError::Shape {
            tensor: "Qwen matvec stored bytes",
            expected: usize::MAX,
            actual: usize::MAX,
        })?;
    if payload.len() != expected_bytes {
        return Err(ModelError::Shape {
            tensor: "Qwen matvec stored bytes",
            expected: expected_bytes,
            actual: payload.len(),
        }
        .into());
    }

    let output_bytes = rows
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or(ModelError::Shape {
            tensor: "Qwen matvec output bytes",
            expected: usize::MAX,
            actual: rows,
        })?;
    let lease = broker.reserve(
        MemoryOwner::Scratch,
        MemoryClass::Transient,
        Bytes(output_bytes as u64),
        64,
    )?;
    // Reserve above succeeds before this physical activation allocation.
    let mut values = vec![0.0; rows];
    let block_elements = dtype.block_size() as usize;
    let block_bytes = dtype.block_bytes() as usize;
    if cols % block_elements != 0 {
        return Err(ModelError::Shape {
            tensor: "Qwen matvec columns per quant block",
            expected: block_elements,
            actual: cols,
        }
        .into());
    }
    let blocks_per_row = cols / block_elements;
    for (row, output) in values.iter_mut().enumerate() {
        let row_start = row * blocks_per_row * block_bytes;
        let row_bytes = &payload[row_start..row_start + blocks_per_row * block_bytes];
        *output = matvec_row(dtype, row_bytes, input)?;
    }
    Ok(Qwen36Activation {
        values,
        _lease: lease,
    })
}

fn decode_values(dtype: GgmlType, payload: &[u8], elements: usize) -> Result<Vec<f32>> {
    match dtype {
        GgmlType::F32 => Ok(payload
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte chunk")))
            .collect()),
        GgmlType::F16 => Ok(payload
            .chunks_exact(2)
            .map(|bytes| {
                crate::format::quant::dequant::f16_to_f32(u16::from_le_bytes(
                    bytes.try_into().expect("two-byte chunk"),
                ))
            })
            .collect()),
        GgmlType::Bf16 => Ok(payload
            .chunks_exact(2)
            .map(|bytes| {
                f32::from_bits(
                    (u16::from_le_bytes(bytes.try_into().expect("two-byte chunk")) as u32) << 16,
                )
            })
            .collect()),
        GgmlType::Q4_0 | GgmlType::Q4K | GgmlType::Q6K | GgmlType::Q8_0 => {
            let block_bytes = dtype.block_bytes() as usize;
            let mut values = Vec::with_capacity(elements);
            for block in payload.chunks_exact(block_bytes) {
                values.extend(
                    dequantize_block(dtype, block).expect("canonical quant decoder exists"),
                );
            }
            if values.len() != elements {
                return Err(ModelError::Shape {
                    tensor: "Qwen quantized decoded elements",
                    expected: elements,
                    actual: values.len(),
                }
                .into());
            }
            Ok(values)
        }
        other => Err(ModelError::Unsupported(format!(
            "Qwen reference decoder does not support GGML type {}",
            other.ggml_id()
        ))
        .into()),
    }
}

fn matvec_row(dtype: GgmlType, row: &[u8], input: &[f32]) -> Result<f32> {
    match dtype {
        GgmlType::F32 => Ok(row
            .chunks_exact(4)
            .zip(input)
            .map(|(bytes, input)| {
                f32::from_le_bytes(bytes.try_into().expect("four-byte chunk")) * input
            })
            .sum()),
        GgmlType::F16 => Ok(row
            .chunks_exact(2)
            .zip(input)
            .map(|(bytes, input)| {
                crate::format::quant::dequant::f16_to_f32(u16::from_le_bytes(
                    bytes.try_into().expect("two-byte chunk"),
                )) * input
            })
            .sum()),
        GgmlType::Bf16 => Ok(row
            .chunks_exact(2)
            .zip(input)
            .map(|(bytes, input)| {
                f32::from_bits(
                    (u16::from_le_bytes(bytes.try_into().expect("two-byte chunk")) as u32) << 16,
                ) * input
            })
            .sum()),
        GgmlType::Q4_0 | GgmlType::Q4K | GgmlType::Q6K | GgmlType::Q8_0 => {
            let block_bytes = dtype.block_bytes() as usize;
            let block_elements = dtype.block_size() as usize;
            let mut output = 0.0;
            for (block_index, block) in row.chunks_exact(block_bytes).enumerate() {
                let decoded =
                    dequantize_block(dtype, block).expect("canonical quant decoder exists");
                let vector =
                    &input[block_index * block_elements..(block_index + 1) * block_elements];
                output += decoded
                    .iter()
                    .zip(vector)
                    .map(|(weight, input)| weight * input)
                    .sum::<f32>();
            }
            Ok(output)
        }
        other => Err(ModelError::Unsupported(format!(
            "Qwen reference matvec does not support GGML type {}",
            other.ggml_id()
        ))
        .into()),
    }
}

/// Qwen-only lazy weight binder. It deliberately loads exactly one extent at
/// a time; Phase 14's high-memory resident MoE profile can retain the returned
/// handles, while the Phase-18 bounded path keeps the same lease contract and
/// changes only cache/operation ownership.
pub struct Qwen36WeightLoader {
    manifest: Qwen36WeightManifest,
    broker: MemoryBroker,
}

impl Qwen36WeightManifest {
    /// Opens and proves the TQF has the complete fixed Qwen3.6 tensor-role
    /// topology. This is intentionally before any model allocation or GPU
    /// resource construction, as required by the container validation order.
    pub fn open_with_broker(path: &Path, broker: &MemoryBroker) -> Result<Self> {
        let reader = TqfReader::open_validated_with_broker(path, broker)?;
        let expected = canonical_header(&"00".repeat(32))?;
        if reader.superblock.model_family_id != expected.model_family_id {
            return Err(ModelError::Unsupported(
                "TQF model family is not qwen3.6-35b-a3b".to_string(),
            )
            .into());
        }
        let manifest = Self { reader };
        manifest.validate_topology()?;
        Ok(manifest)
    }

    #[cfg(test)]
    pub fn open(path: &Path) -> Result<Self> {
        let broker = MemoryBroker::new(Bytes(256 * 1024 * 1024));
        Self::open_with_broker(path, &broker)
    }

    pub fn reader(&self) -> &TqfReader {
        &self.reader
    }

    fn require(&self, role: TensorRole, layer: Option<LayerId>) -> Result<()> {
        self.reader.tensor(role as u32, layer).map(|_| ())
    }

    fn require_elements(&self, role: TensorRole, layer: Option<LayerId>) -> Result<()> {
        let Some(expected) = expected_elements(role) else {
            return self.require(role, layer);
        };
        let extent = self.reader.tensor(role as u32, layer)?;
        let actual: usize = extent
            .logical_elements
            .try_into()
            .map_err(|_| ModelError::Shape {
                tensor: "TQF tensor logical elements",
                expected,
                actual: usize::MAX,
            })?;
        if actual != expected {
            return Err(ModelError::Shape {
                tensor: "Qwen3.6 tensor logical elements",
                expected,
                actual,
            }
            .into());
        }
        Ok(())
    }

    fn require_shape(&self, role: TensorRole, layer: Option<LayerId>) -> Result<()> {
        let Some(expected) = expected_shape(role) else {
            return Ok(());
        };
        let extent = self.reader.tensor(role as u32, layer)?;
        let actual_rank: usize = extent.rank.try_into().map_err(|_| ModelError::Shape {
            tensor: "TQF tensor rank",
            expected: expected.len(),
            actual: usize::MAX,
        })?;
        let actual = extent.dims.get(..actual_rank).ok_or(ModelError::Shape {
            tensor: "TQF tensor rank",
            expected: expected.len(),
            actual: actual_rank,
        })?;
        if actual != expected {
            return Err(ModelError::Unsupported(format!(
                "Qwen3.6 tensor {role:?} at layer {layer:?} has dimensions {actual:?}; expected {expected:?}"
            ))
            .into());
        }
        Ok(())
    }

    fn require_each(&self, layer: LayerId, roles: &[TensorRole]) -> Result<()> {
        for &role in roles {
            self.require_elements(role, Some(layer))?;
            self.require_shape(role, Some(layer))?;
        }
        Ok(())
    }

    /// Ensures every layer has its shared MoE/norm roles and exactly the
    /// operator family prescribed by the compile-time layer-kind table.
    /// TQF tensor lookups return typed `TensorNotFound` errors, so malformed
    /// or partial conversions fail before installation instead of generating
    /// from a silently incomplete graph.
    pub fn validate_topology(&self) -> Result<()> {
        for role in [
            TensorRole::TokenEmbedding,
            TensorRole::FinalNorm,
            TensorRole::LmHead,
        ] {
            self.require_elements(role, None)?;
            self.require_shape(role, None)?;
        }

        const COMMON: &[TensorRole] = &[
            TensorRole::AttnNorm,
            TensorRole::FfnNorm,
            TensorRole::RouterGate,
            TensorRole::SharedExpertInputGate,
            TensorRole::SharedExpertGate,
            TensorRole::SharedExpertUp,
            TensorRole::SharedExpertDown,
        ];
        const FULL_ATTENTION: &[TensorRole] = &[
            TensorRole::AttnQProj,
            TensorRole::AttnKProj,
            TensorRole::AttnVProj,
            TensorRole::AttnOProj,
            TensorRole::AttnQNorm,
            TensorRole::AttnKNorm,
        ];
        const GDN: &[TensorRole] = &[
            TensorRole::GdnInProjQkv,
            TensorRole::GdnInProjZ,
            TensorRole::GdnInProjA,
            TensorRole::GdnInProjB,
            TensorRole::GdnConv1d,
            TensorRole::GdnALog,
            TensorRole::GdnDtBias,
            TensorRole::GdnGatedNorm,
            TensorRole::GdnOutProj,
        ];

        for index in 0..Qwen36Geometry::NUM_LAYERS {
            let layer = LayerId(index as u8);
            self.require_each(layer, COMMON)?;
            self.require_routed_experts(layer)?;
            match Qwen36Geometry::layer_kind(layer) {
                LayerKind::FullAttention => self.require_each(layer, FULL_ATTENTION)?,
                LayerKind::GatedDeltaNet => self.require_each(layer, GDN)?,
            }
        }
        Ok(())
    }

    /// Accepts the original resident tensor triplet only as a development
    /// oracle. Canonical v2 conversions instead require every Q4_K
    /// whole-expert superextent so normal startup can bind the streaming
    /// cache without retaining 256 planes per layer.
    fn require_routed_experts(&self, layer: LayerId) -> Result<()> {
        let roles = [
            TensorRole::RoutedExpertGate,
            TensorRole::RoutedExpertUp,
            TensorRole::RoutedExpertDown,
        ];
        if roles
            .iter()
            .all(|&role| self.reader.tensor(role as u32, Some(layer)).is_ok())
        {
            return self.require_each(layer, &roles);
        }

        let gate_bytes: u32 = GgmlType::Q4K
            .byte_size((Qwen36Geometry::HIDDEN_SIZE * Qwen36Geometry::ROUTED_EXPERT_WIDTH) as u64)?
            .try_into()
            .map_err(|_| ModelError::Shape {
                tensor: "Qwen routed expert gate bytes",
                expected: usize::MAX,
                actual: usize::MAX,
            })?;
        let expected_total = gate_bytes.checked_mul(3).ok_or(ModelError::Shape {
            tensor: "Qwen routed expert superextent bytes",
            expected: usize::MAX,
            actual: usize::MAX,
        })?;
        let q4k_layout = TQF_QUANT_PASSTHROUGH_Q4_K as u16;
        for expert in 0..Qwen36Geometry::NUM_EXPERTS {
            let (index, tiles) = self
                .reader
                .expert(layer, crate::ids::ExpertId(expert as u16))?;
            if index.layout_id != q4k_layout
                || index.stored_bytes != expected_total
                || tiles.len() != 2
                || tiles[0].matrix != ExpertMatrix::GateUp
                || tiles[0].relative_offset != 0
                || tiles[0].stored_bytes != gate_bytes * 2
                || tiles[0].quant_layout_id != q4k_layout
                || tiles[1].matrix != ExpertMatrix::Down
                || tiles[1].relative_offset != gate_bytes * 2
                || tiles[1].stored_bytes != gate_bytes
                || tiles[1].quant_layout_id != q4k_layout
            {
                return Err(ModelError::Unsupported(format!(
                    "layer {} routed-expert superextent {} is not canonical Q4_K",
                    layer.0, expert
                ))
                .into());
            }
        }
        Ok(())
    }

    pub fn uses_expert_superextents(&self) -> bool {
        self.reader
            .tensor(TensorRole::RoutedExpertGate as u32, Some(LayerId(0)))
            .is_err()
            && self
                .reader
                .expert(LayerId(0), crate::ids::ExpertId(0))
                .is_ok()
    }
}

impl Qwen36WeightLoader {
    pub fn open(path: &Path, broker: MemoryBroker) -> Result<Self> {
        Ok(Self {
            manifest: Qwen36WeightManifest::open_with_broker(path, &broker)?,
            broker,
        })
    }

    pub fn manifest(&self) -> &Qwen36WeightManifest {
        &self.manifest
    }

    /// Returns the exact stored size without allocating or reading payload.
    /// The bounded runtime uses this to leave enough broker headroom for its
    /// largest single core extent before choosing an expert-cache capacity.
    pub fn stored_bytes(&self, role: TensorRole, layer: Option<LayerId>) -> Result<Bytes> {
        Ok(Bytes(
            self.manifest
                .reader
                .tensor(role as u32, layer)?
                .stored_bytes,
        ))
    }

    /// Loads one verified extent after reserving its exact stored byte count.
    /// Routed-expert matrices use the expert owner/class; everything else is
    /// resident core. The caller decides retention by holding or dropping the
    /// returned handle.
    pub fn load(&self, role: TensorRole, layer: Option<LayerId>) -> Result<LoadedQwen36Tensor> {
        let extent = self.manifest.reader.tensor(role as u32, layer)?;
        let dtype = GgmlType::from_ggml_id(extent.dtype_id)?;
        let (owner, class) = match role {
            TensorRole::RoutedExpertGate
            | TensorRole::RoutedExpertUp
            | TensorRole::RoutedExpertDown => (MemoryOwner::ExpertPinned, MemoryClass::Elastic),
            _ => (MemoryOwner::Core, MemoryClass::Fixed),
        };
        let lease = self
            .broker
            .reserve(owner, class, Bytes(extent.stored_bytes), 64)?;
        let length: usize = extent
            .stored_bytes
            .try_into()
            .map_err(|_| ModelError::Shape {
                tensor: "TQF stored tensor bytes",
                expected: usize::MAX,
                actual: usize::MAX,
            })?;
        // The reservation above succeeds before this allocation.
        let mut bytes = vec![0u8; length];
        if let Err(error) = self.manifest.reader.read_extent_into(extent, &mut bytes) {
            drop(bytes);
            drop(lease);
            return Err(error);
        }
        Ok(LoadedQwen36Tensor {
            role,
            layer,
            dtype,
            dims: extent.dims[..extent.rank as usize].to_vec(),
            bytes,
            _lease: lease,
        })
    }

    /// Returns the exact cache reservation required for a routed expert
    /// without reading its payload. Cache policy uses this before admitting a
    /// miss, so eviction happens before any physical allocation.
    pub fn expert_stored_bytes(
        &self,
        layer: LayerId,
        expert: crate::ids::ExpertId,
    ) -> Result<Bytes> {
        let (index, _) = self.manifest.reader.expert(layer, expert)?;
        Ok(Bytes(index.stored_bytes as u64))
    }

    /// Loads one canonical routed expert from its two-tile superextent. This
    /// is intentionally narrower than the old rank-three resident tensors:
    /// selected gate, up, and down matrices become one broker-accounted cache
    /// entry, with no route-dependent reinterpretation of their bytes.
    pub fn load_expert(
        &self,
        layer: LayerId,
        expert: crate::ids::ExpertId,
    ) -> Result<LoadedQwen36Expert> {
        let (index, tiles) = self.manifest.reader.expert(layer, expert)?;
        let q4k_layout = TQF_QUANT_PASSTHROUGH_Q4_K as u16;
        if index.layout_id != q4k_layout
            || ggml_type_for_quant_layout(index.layout_id as u32) != Some(GgmlType::Q4K)
            || tiles.len() != 2
            || tiles[0].matrix != ExpertMatrix::GateUp
            || tiles[0].relative_offset != 0
            || tiles[0].quant_layout_id != q4k_layout
            || tiles[0].stored_bytes
                != (LoadedQwen36Expert::GATE_BYTES + LoadedQwen36Expert::UP_BYTES) as u32
            || tiles[1].matrix != ExpertMatrix::Down
            || tiles[1].relative_offset != tiles[0].stored_bytes
            || tiles[1].quant_layout_id != q4k_layout
            || tiles[1].stored_bytes != LoadedQwen36Expert::DOWN_BYTES as u32
            || index.stored_bytes
                != (LoadedQwen36Expert::GATE_BYTES
                    + LoadedQwen36Expert::UP_BYTES
                    + LoadedQwen36Expert::DOWN_BYTES) as u32
        {
            return Err(ModelError::Unsupported(
                "TQF routed expert tiles are not canonical Qwen3.6 Q4_K matrices".to_string(),
            )
            .into());
        }
        let lease = self.broker.reserve(
            MemoryOwner::ExpertPinned,
            MemoryClass::Elastic,
            Bytes(index.stored_bytes as u64),
            64,
        )?;
        let mut bytes = vec![0; index.stored_bytes as usize];
        if let Err(error) = self.manifest.reader.read_expert_into(index, &mut bytes) {
            drop(bytes);
            drop(lease);
            return Err(error);
        }
        Ok(LoadedQwen36Expert {
            layer,
            expert,
            bytes,
            _lease: lease,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experts::{RouterResult, WholeExpertLfuCache};
    use crate::format::tqf::{canonical_header, TqfSectionKind, TqfWriter};

    fn fixture_path(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "tqf-qwen36-manifest-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(name)
    }

    fn loaded_tensor(
        broker: &MemoryBroker,
        dtype: GgmlType,
        dims: Vec<u64>,
        bytes: Vec<u8>,
    ) -> LoadedQwen36Tensor {
        let lease = broker
            .reserve(
                MemoryOwner::Core,
                MemoryClass::Fixed,
                Bytes(bytes.len() as u64),
                64,
            )
            .unwrap();
        LoadedQwen36Tensor {
            role: TensorRole::AttnQProj,
            layer: Some(LayerId(3)),
            dtype,
            dims,
            bytes,
            _lease: lease,
        }
    }

    #[test]
    fn canonical_gdn_conv_storage_decodes_as_channel_major_weights() {
        let broker = MemoryBroker::new(Bytes(1024 * 1024));
        let elements = Qwen36Geometry::GDN_CONV_CHANNELS * Qwen36Geometry::GDN_CONV_WIDTH;
        let bytes = (0..elements)
            .flat_map(|index| (index as f32).to_le_bytes())
            .collect();
        let mut tensor = loaded_tensor(
            &broker,
            GgmlType::F32,
            vec![
                Qwen36Geometry::GDN_CONV_WIDTH as u64,
                Qwen36Geometry::GDN_CONV_CHANNELS as u64,
            ],
            bytes,
        );
        tensor.role = TensorRole::GdnConv1d;

        let weights = tensor.gdn_conv1d_weights(&broker).unwrap();
        assert_eq!(weights.values.len(), elements);
        assert_eq!(weights.values[0], 0.0);
        assert_eq!(weights.values[Qwen36Geometry::GDN_CONV_WIDTH], 4.0);
        assert_eq!(weights.values[elements - 1], (elements - 1) as f32);
    }

    #[test]
    fn rank_one_parameter_computes_a_scalar_dot_product() {
        let broker = MemoryBroker::new(Bytes(1024 * 1024));
        let bytes = [2.0_f32, -3.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        let tensor = loaded_tensor(&broker, GgmlType::F32, vec![2], bytes);
        assert_eq!(tensor.dot(&broker, &[4.0, 5.0]).unwrap(), -7.0);
    }

    fn write_role(writer: &mut TqfWriter, role: TensorRole, layer: Option<LayerId>) {
        write_role_with_shape(writer, role, layer, expected_shape(role).unwrap());
    }

    fn write_role_with_shape(
        writer: &mut TqfWriter,
        role: TensorRole,
        layer: Option<LayerId>,
        shape: &[u64],
    ) {
        let suffix = layer.map(|value| value.0).unwrap_or(255);
        writer
            .write_extent(
                role as u32,
                &format!("{role:?}-{suffix}"),
                layer,
                TqfSectionKind::ResidentCore,
                shape,
                0,
                0,
                1,
                &[1],
            )
            .unwrap();
    }

    fn build_complete_fixture(
        path: &Path,
        omit: Option<(TensorRole, LayerId)>,
        bad_shape: Option<(TensorRole, LayerId)>,
        expert_count: usize,
    ) {
        let mut writer =
            TqfWriter::create_partial(path, canonical_header(&"ab".repeat(32)).unwrap()).unwrap();
        for role in [
            TensorRole::TokenEmbedding,
            TensorRole::FinalNorm,
            TensorRole::LmHead,
        ] {
            write_role(&mut writer, role, None);
        }
        let common = [
            TensorRole::AttnNorm,
            TensorRole::FfnNorm,
            TensorRole::RouterGate,
            TensorRole::SharedExpertInputGate,
            TensorRole::SharedExpertGate,
            TensorRole::SharedExpertUp,
            TensorRole::SharedExpertDown,
            TensorRole::RoutedExpertGate,
            TensorRole::RoutedExpertUp,
            TensorRole::RoutedExpertDown,
        ];
        let full = [
            TensorRole::AttnQProj,
            TensorRole::AttnKProj,
            TensorRole::AttnVProj,
            TensorRole::AttnOProj,
            TensorRole::AttnQNorm,
            TensorRole::AttnKNorm,
        ];
        let gdn = [
            TensorRole::GdnInProjQkv,
            TensorRole::GdnInProjZ,
            TensorRole::GdnInProjA,
            TensorRole::GdnInProjB,
            TensorRole::GdnConv1d,
            TensorRole::GdnALog,
            TensorRole::GdnDtBias,
            TensorRole::GdnGatedNorm,
            TensorRole::GdnOutProj,
        ];
        for index in 0..Qwen36Geometry::NUM_LAYERS {
            let layer = LayerId(index as u8);
            let layer_roles: &[TensorRole] = match Qwen36Geometry::layer_kind(layer) {
                LayerKind::FullAttention => &full,
                LayerKind::GatedDeltaNet => &gdn,
            };
            for role in common.iter().chain(layer_roles).copied() {
                if omit == Some((role, layer)) {
                    continue;
                }
                if bad_shape == Some((role, layer)) {
                    // Same element count but not the canonical GGUF rank or
                    // orientation: it must be rejected before kernel binding.
                    write_role_with_shape(
                        &mut writer,
                        role,
                        Some(layer),
                        &[expected_elements(role).unwrap() as u64],
                    );
                } else {
                    write_role(&mut writer, role, Some(layer));
                }
            }
        }
        let expert_bytes = vec![0u8; LoadedQwen36Expert::GATE_BYTES];
        for expert in 0..expert_count {
            writer
                .write_expert_parts(
                    LayerId(0),
                    crate::ids::ExpertId(expert as u16),
                    TQF_QUANT_PASSTHROUGH_Q4_K as u16,
                    &expert_bytes,
                    &expert_bytes,
                    &expert_bytes,
                )
                .unwrap();
        }
        writer.commit().unwrap();
    }

    #[test]
    fn complete_fixed_graph_is_accepted() {
        let path = fixture_path("complete.tqf");
        build_complete_fixture(&path, None, None, 0);
        Qwen36WeightManifest::open(&path).unwrap();
        std::fs::remove_file(path).unwrap();
    }

    /// Real-container qualification seam. The canonical TQF is intentionally
    /// not a repository fixture, so normal unit runs leave this ignored.
    #[test]
    #[ignore = "requires the converted canonical TQF"]
    fn canonical_tqf_topology_is_accepted() {
        let path = std::env::var_os("TQF_CANONICAL_TQF")
            .map(std::path::PathBuf::from)
            .expect("set TQF_CANONICAL_TQF to the converted canonical container");
        let broker = MemoryBroker::new(Bytes(4 * 1024 * 1024 * 1024));
        Qwen36WeightManifest::open_with_broker(&path, &broker).unwrap();
    }

    #[test]
    fn a_missing_full_attention_role_rejects_installation() {
        let path = fixture_path("missing.tqf");
        let layer = LayerId(3);
        build_complete_fixture(&path, Some((TensorRole::AttnQProj, layer)), None, 0);
        assert!(Qwen36WeightManifest::open(&path).is_err());
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn a_transposed_or_flattened_tensor_rejects_installation() {
        let path = fixture_path("bad-shape.tqf");
        let layer = LayerId(3);
        build_complete_fixture(&path, None, Some((TensorRole::AttnQProj, layer)), 0);
        assert!(Qwen36WeightManifest::open(&path).is_err());
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn loader_reserves_before_reading_and_releases_with_tensor() {
        let path = fixture_path("loader.tqf");
        build_complete_fixture(&path, None, None, 0);
        let broker = MemoryBroker::new(Bytes(1024 * 1024));
        let loader = Qwen36WeightLoader::open(&path, broker.clone()).unwrap();
        let metadata_bytes = broker.snapshot().reserved.0;
        let tensor = loader.load(TensorRole::TokenEmbedding, None).unwrap();
        assert_eq!(tensor.bytes, vec![1]);
        assert_eq!(broker.snapshot().reserved, Bytes(metadata_bytes + 1));
        drop(tensor);
        assert_eq!(broker.snapshot().reserved, Bytes(metadata_bytes));
        drop(loader);
        assert_eq!(broker.snapshot().reserved, Bytes(0));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn routed_expert_load_is_broker_accounted_and_validates_two_tiles() {
        let path = fixture_path("expert-loader.tqf");
        build_complete_fixture(&path, None, None, 1);
        let expected = (LoadedQwen36Expert::GATE_BYTES * 3) as u64;
        let broker = MemoryBroker::new(Bytes(expected + 1024 * 1024));
        let loader = Qwen36WeightLoader::open(&path, broker.clone()).unwrap();
        let metadata_bytes = broker.snapshot().reserved.0;
        assert_eq!(
            loader
                .expert_stored_bytes(LayerId(0), crate::ids::ExpertId(0))
                .unwrap(),
            Bytes(expected)
        );
        let expert = loader
            .load_expert(LayerId(0), crate::ids::ExpertId(0))
            .unwrap();
        assert_eq!(expert.stored_bytes(), Bytes(expected));
        assert_eq!(broker.snapshot().reserved, Bytes(metadata_bytes + expected));
        drop(expert);
        assert_eq!(broker.snapshot().reserved, Bytes(metadata_bytes));
        drop(loader);
        assert_eq!(broker.snapshot().reserved, Bytes(0));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn expert_cache_evicts_the_least_frequent_whole_superextent_before_reading() {
        let path = fixture_path("expert-cache.tqf");
        build_complete_fixture(&path, None, None, 3);
        let expert_bytes = (LoadedQwen36Expert::GATE_BYTES * 3) as u64;
        let broker = MemoryBroker::new(Bytes(expert_bytes * 2 + 1024 * 1024));
        let loader = Qwen36WeightLoader::open(&path, broker.clone()).unwrap();
        let metadata_bytes = broker.snapshot().reserved.0;
        let mut cache = WholeExpertLfuCache::new(Bytes(expert_bytes * 2));
        let layer = LayerId(0);

        assert_eq!(
            cache
                .get_or_load(&loader, layer, crate::ids::ExpertId(0))
                .unwrap()
                .expert,
            crate::ids::ExpertId(0)
        );
        // Make expert zero hotter than expert one before capacity forces an
        // eviction. The second lookup must not allocate another payload.
        cache
            .get_or_load(&loader, layer, crate::ids::ExpertId(0))
            .unwrap();
        cache
            .get_or_load(&loader, layer, crate::ids::ExpertId(1))
            .unwrap();
        cache
            .get_or_load(&loader, layer, crate::ids::ExpertId(2))
            .unwrap();
        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 3);
        assert_eq!(stats.evictions, 1);
        assert_eq!(stats.resident_bytes, Bytes(expert_bytes * 2));
        assert_eq!(stats.raw_miss_bytes, Bytes(expert_bytes * 3));
        assert_eq!(
            broker.snapshot().reserved,
            Bytes(metadata_bytes + expert_bytes * 2)
        );
        // Expert one was the LFU victim; revisiting it is a new miss and
        // proves eviction happens at whole-expert granularity.
        cache
            .get_or_load(&loader, layer, crate::ids::ExpertId(1))
            .unwrap();
        assert_eq!(cache.stats().misses, 4);
        assert_eq!(cache.stats().evictions, 2);
        assert_eq!(cache.stats().raw_miss_bytes, Bytes(expert_bytes * 4));
        drop(cache);
        assert_eq!(broker.snapshot().reserved, Bytes(metadata_bytes));
        drop(loader);
        assert_eq!(broker.snapshot().reserved, Bytes(0));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn exact_route_transaction_pins_all_selected_experts_until_finish() {
        let path = fixture_path("expert-plan.tqf");
        build_complete_fixture(&path, None, None, 9);
        let expert_bytes = (LoadedQwen36Expert::GATE_BYTES * 3) as u64;
        let broker = MemoryBroker::new(Bytes(expert_bytes * 8 + 1024 * 1024));
        let loader = Qwen36WeightLoader::open(&path, broker.clone()).unwrap();
        let metadata_bytes = broker.snapshot().reserved.0;
        let mut cache = WholeExpertLfuCache::new(Bytes(expert_bytes * 8));
        let layer = LayerId(0);
        let route = RouterResult {
            ids: std::array::from_fn(|index| crate::ids::ExpertId(index as u16)),
            weights: [0.125; 8],
        };

        let plan = cache.prepare_exact_route(&loader, layer, &route).unwrap();
        assert_eq!(plan.route, route);
        assert_eq!(plan.hits, [false; 8]);
        assert_eq!(plan.miss_bytes, Bytes(expert_bytes * 8));
        for expert in route.ids {
            assert_eq!(cache.planned_expert(&plan, expert).unwrap().expert, expert);
        }
        let mut tampered = plan.clone();
        tampered.route.ids[0] = crate::ids::ExpertId(8);
        assert!(cache.finish_exact_route(&tampered).is_err());
        assert_eq!(
            cache
                .planned_expert(&plan, crate::ids::ExpertId(0))
                .unwrap()
                .expert,
            crate::ids::ExpertId(0)
        );
        assert!(cache
            .get_or_load(&loader, layer, crate::ids::ExpertId(8))
            .is_err());
        assert_eq!(cache.stats().evictions, 0);
        assert_eq!(cache.stats().misses, 8);
        assert_eq!(cache.stats().raw_miss_bytes, Bytes(expert_bytes * 8));

        cache.finish_exact_route(&plan).unwrap();
        cache
            .get_or_load(&loader, layer, crate::ids::ExpertId(8))
            .unwrap();
        assert_eq!(cache.stats().evictions, 1);
        assert_eq!(cache.stats().misses, 9);
        assert_eq!(cache.stats().raw_miss_bytes, Bytes(expert_bytes * 9));
        drop(cache);
        assert_eq!(broker.snapshot().reserved, Bytes(metadata_bytes));
        drop(loader);
        assert_eq!(broker.snapshot().reserved, Bytes(0));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn matvec_preserves_gguf_column_row_order_and_broker_accounting() {
        let broker = MemoryBroker::new(Bytes(128));
        let mut bytes = Vec::new();
        for value in [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        // GGUF `{3, 2}` means two contiguous rows of three inputs.
        let tensor = loaded_tensor(&broker, GgmlType::F32, vec![3, 2], bytes);
        let output = tensor.matvec(&broker, &[1.0, 2.0, 3.0]).unwrap();
        assert_eq!(output.values, vec![14.0, 32.0]);
        assert_eq!(broker.snapshot().reserved, Bytes(32));
        drop(output);
        assert_eq!(broker.snapshot().reserved, Bytes(24));
        drop(tensor);
        assert_eq!(broker.snapshot().reserved, Bytes(0));
    }

    #[test]
    fn vector_and_row_decode_without_materializing_whole_tensor() {
        let broker = MemoryBroker::new(Bytes(512));
        let vector = loaded_tensor(
            &broker,
            GgmlType::F32,
            vec![3],
            [1.0f32, -2.0, 3.5]
                .into_iter()
                .flat_map(|value| value.to_le_bytes())
                .collect(),
        );
        assert_eq!(vector.vector(&broker).unwrap().values, vec![1.0, -2.0, 3.5]);

        let mut first = vec![0u8; GgmlType::Q4_0.block_bytes() as usize];
        first[..2].copy_from_slice(&0x3c00u16.to_le_bytes());
        first[2..].fill(0xff);
        let second = vec![0u8; GgmlType::Q4_0.block_bytes() as usize];
        let mut rows = first;
        rows.extend_from_slice(&second);
        let matrix = loaded_tensor(&broker, GgmlType::Q4_0, vec![32, 2], rows);
        assert_eq!(matrix.row(&broker, 0).unwrap().values, vec![7.0; 32]);
        assert_eq!(matrix.row(&broker, 1).unwrap().values, vec![0.0; 32]);
    }

    #[test]
    fn activation_algebra_is_broker_accounted_and_uses_qwen_norm_scale() {
        let broker = MemoryBroker::new(Bytes(256));
        let mut left = Qwen36Activation::zeros(&broker, 2).unwrap();
        left.values.copy_from_slice(&[3.0, 4.0]);
        let mut right = Qwen36Activation::zeros(&broker, 2).unwrap();
        right.values.copy_from_slice(&[1.0, 2.0]);
        let sum = Qwen36Activation::residual_add(&broker, &left, &right).unwrap();
        assert_eq!(sum.values, vec![4.0, 6.0]);
        let norm = Qwen36Activation::qwen_rmsnorm(&broker, &left, &[1.0, 1.0]).unwrap();
        let rms = (norm.values.iter().map(|value| value * value).sum::<f32>() / 2.0).sqrt();
        assert!((rms - 1.0).abs() < 1e-5);
        let swiglu = Qwen36Activation::silu_mul(&broker, &left, &right).unwrap();
        assert!(swiglu.values[0] > 0.0);
        assert_eq!(broker.snapshot().reserved, Bytes(40));
    }

    #[test]
    fn matvec_decodes_canonical_q4_zero_blocks() {
        let broker = MemoryBroker::new(Bytes(256));
        let mut block = vec![0u8; GgmlType::Q4_0.block_bytes() as usize];
        block[..2].copy_from_slice(&0x3c00u16.to_le_bytes()); // f16 1.0 scale
        block[2..].fill(0xff); // each quantized value is 15 - 8 = 7
        let tensor = loaded_tensor(&broker, GgmlType::Q4_0, vec![32, 1], block);
        let output = tensor.matvec(&broker, &vec![1.0; 32]).unwrap();
        assert_eq!(output.values, vec![224.0]);
    }

    #[test]
    fn routed_expert_matvec_selects_one_canonical_expert_plane() {
        let broker = MemoryBroker::new(Bytes(256));
        let mut expert_zero = vec![0u8; GgmlType::Q4_0.block_bytes() as usize];
        expert_zero[..2].copy_from_slice(&0x3c00u16.to_le_bytes());
        expert_zero[2..].fill(0xff); // 32 values of 7
        let expert_one = vec![0u8; GgmlType::Q4_0.block_bytes() as usize];
        let mut bytes = expert_zero;
        bytes.extend_from_slice(&expert_one);
        let tensor = loaded_tensor(&broker, GgmlType::Q4_0, vec![32, 1, 2], bytes);
        assert_eq!(
            tensor
                .matvec_expert(&broker, 0, &vec![1.0; 32])
                .unwrap()
                .values,
            vec![224.0]
        );
        assert_eq!(
            tensor
                .matvec_expert(&broker, 1, &vec![1.0; 32])
                .unwrap()
                .values,
            vec![0.0]
        );
        assert!(tensor.matvec_expert(&broker, 2, &vec![1.0; 32]).is_err());
    }

    #[test]
    fn matvec_dispatches_canonical_q6_k_lm_head_blocks() {
        let broker = MemoryBroker::new(Bytes(512));
        let mut block = vec![0u8; GgmlType::Q6K.block_bytes() as usize];
        block[192..208].fill(1); // signed scale 1 for every 16-value group
        block[208..210].copy_from_slice(&0x3c00u16.to_le_bytes()); // f16 1.0
        let tensor = loaded_tensor(&broker, GgmlType::Q6K, vec![256, 1], block);
        assert_eq!(
            tensor.matvec(&broker, &vec![1.0; 256]).unwrap().values,
            vec![-8192.0]
        );
    }

    #[test]
    fn unsupported_quant_type_is_rejected_without_leaking_output_budget() {
        let broker = MemoryBroker::new(Bytes(512));
        let bytes = vec![0u8; GgmlType::Q5K.block_bytes() as usize];
        let tensor = loaded_tensor(&broker, GgmlType::Q5K, vec![256, 1], bytes);
        assert!(tensor.matvec(&broker, &vec![1.0; 256]).is_err());
        assert_eq!(
            broker.snapshot().reserved,
            Bytes(GgmlType::Q5K.block_bytes())
        );
    }
}
