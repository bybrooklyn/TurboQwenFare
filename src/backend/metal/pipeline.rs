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
        let key = (function_name.to_string(), specialization.to_string());
        if !self.entries.contains_key(&key) {
            let function = library.get_function(function_name, None).map_err(|e| {
                BackendError::Gpu(format!("function {function_name:?} not found: {e}"))
            })?;
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
