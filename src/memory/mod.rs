//! The memory broker: the single source of truth for every large allocation
//! (spec Part VI sections 37-40; Part XIV sections 128-132). `--memory` is a
//! hard contract, not an advisory cache size.
