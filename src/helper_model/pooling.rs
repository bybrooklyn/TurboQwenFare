//! Mean-token pooling (spec §86: `pplx-embed-v1-0.6b`'s own `1_Pooling`
//! config selects `pooling_mode_mean_tokens`). One request is one
//! unpadded sequence, so every position is valid and the mask is
//! implicitly all-ones.

pub fn mean_pool(hidden: &[Vec<f32>]) -> Vec<f32> {
    let hidden_dim = hidden.first().map(|h| h.len()).unwrap_or(0);
    let mut out = vec![0.0f32; hidden_dim];
    for h in hidden {
        for (o, v) in out.iter_mut().zip(h) {
            *o += v;
        }
    }
    let n = hidden.len().max(1) as f32;
    for v in out.iter_mut() {
        *v /= n;
    }
    out
}
