//! Pipeline-state cache (spec §50: "Pipeline-state objects are compiled/
//! cached once per specialization."). Keyed by function name plus a
//! caller-supplied specialization key — the baseline harness only ever
//! uses the empty specialization, but the key exists now so per-M4
//! function-constant variants (spec §51's table of kernel-family
//! specializations) slot in without a cache-shape change later.

use std::collections::HashMap;

use metal_sys::{ComputePipelineState, Library};

use crate::error::{BackendError, Result};

#[derive(Default)]
pub struct PipelineCache {
    entries: HashMap<(String, String), ComputePipelineState>,
}

impl PipelineCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the cached pipeline for `(function_name, specialization)`,
    /// compiling and inserting it on first use.
    pub fn get_or_compile(
        &mut self,
        device: &metal_sys::Device,
        library: &Library,
        function_name: &str,
        specialization: &str,
    ) -> Result<&ComputePipelineState> {
        self.get_or_compile_with_constants(device, library, function_name, specialization, &[])
    }

    /// Same as `get_or_compile`, but compiles the function with the given
    /// MSL function-constant values (`[[function_constant(index)]]` →
    /// u32 value). Phase 20's shape specialization uses this so a kernel
    /// family can compile one pipeline per Qwen shape (e.g. blocks-per-row
    /// 2 vs 8) and the Metal compiler unrolls on the constant. The
    /// `specialization` string is the cache key and must encode the
    /// constants uniquely.
    pub fn get_or_compile_with_constants(
        &mut self,
        device: &metal_sys::Device,
        library: &Library,
        function_name: &str,
        specialization: &str,
        constants: &[(u32, u32)],
    ) -> Result<&ComputePipelineState> {
        let key = (function_name.to_string(), specialization.to_string());
        if !self.entries.contains_key(&key) {
            let function = if constants.is_empty() {
                library.get_function(function_name, None).map_err(|e| {
                    BackendError::Gpu(format!("function {function_name:?} not found: {e}"))
                })?
            } else {
                let values = metal_sys::FunctionConstantValues::new();
                for (index, value) in constants {
                    values.set_constant_value_at_index(
                        (value as *const u32).cast(),
                        metal_sys::MTLDataType::UInt,
                        *index as u64,
                    );
                }
                library
                    .get_function(function_name, Some(values))
                    .map_err(|e| {
                        BackendError::Gpu(format!("function {function_name:?} not found: {e}"))
                    })?
            };
            let pipeline = device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(|e| {
                    BackendError::Gpu(format!(
                        "failed to build pipeline for {function_name:?}: {e}"
                    ))
                })?;
            self.entries.insert(key.clone(), pipeline);
        }
        Ok(self.entries.get(&key).expect("just inserted"))
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::metal::context::MetalContext;
    use crate::backend::metal::shaderlib::{self, BANDWIDTH_COPY_FUNCTION};

    #[test]
    fn compiles_once_and_reuses_cached_pipeline() {
        let Ok(ctx) = MetalContext::init() else {
            eprintln!("skipping Metal test: no device available in this environment");
            return;
        };
        let library = shaderlib::load_baseline_library(&ctx).unwrap();
        let mut cache = PipelineCache::new();

        cache
            .get_or_compile(ctx.device(), &library, BANDWIDTH_COPY_FUNCTION, "")
            .unwrap();
        assert_eq!(cache.len(), 1);

        // Second lookup for the same key must not grow the cache.
        cache
            .get_or_compile(ctx.device(), &library, BANDWIDTH_COPY_FUNCTION, "")
            .unwrap();
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn unknown_function_is_a_typed_error() {
        let Ok(ctx) = MetalContext::init() else {
            eprintln!("skipping Metal test: no device available in this environment");
            return;
        };
        let library = shaderlib::load_baseline_library(&ctx).unwrap();
        let mut cache = PipelineCache::new();
        assert!(cache
            .get_or_compile(ctx.device(), &library, "does_not_exist", "")
            .is_err());
    }
}
