//! Prefix snapshot store (spec §66-67, §156): "TQF improves storage by
//! deduplicating TQKV pages. Snapshots reference immutable page IDs plus a
//! GDN state checkpoint rather than serializing complete K/V history every
//! time." v1 is exact-match only (spec §67): "v1 uses the longest exact
//! token-prefix match only."
//!
//! Layout on disk, all writes going through `atomic_write_bytes`/
//! `atomic_write_toml` (temp file + `fsync` + rename, crate invariant #9):
//!
//! ```text
//! <root>/blobs/<hash[0..2]>/<hash-hex>.bin   content-addressed TQKV page
//!                                            and GDN state blobs (shared
//!                                            across every snapshot that
//!                                            references them)
//! <root>/blob-refcounts.toml                 hash -> reference count
//! <root>/snapshots/<prefix-hash-hex>.toml    one manifest per stored prefix
//! <root>/index.toml                          prefix-hash -> bytes/last-used,
//!                                            for LRU eviction without
//!                                            opening every manifest
//! ```
//!
//! A page/GDN-state blob is deleted only when its refcount reaches zero —
//! two snapshots that share a system-prompt prefix share the same TQKV
//! page bytes on disk, and evicting one snapshot never corrupts the other.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::config::persisted::atomic_write_toml;
use crate::context::tqkv::{SealedPage, TqkvPrecision};
use crate::error::{ContextError, Result};
use crate::ids::LayerId;
use crate::memory::MemoryBroker;
use crate::model::qwen36::gdn::GdnState;

const SCHEMA_VERSION: u32 = 1;

/// One full-attention layer's TQKV state, ready to persist. Built by
/// `FullAttentionLayer::capture_tqkv_for_snapshot`.
pub struct FullAttentionCapture {
    pub layer: LayerId,
    pub precision: TqkvPrecision,
    pub position: u64,
    /// `(content_id, serialized page bytes)` for every sealed page, in
    /// order. Bytes are only actually written to disk for content IDs not
    /// already present (`PrefixSnapshotStore::store` checks first).
    pub pages: Vec<([u8; 32], Vec<u8>)>,
    pub tail_keys: Vec<f32>,
    pub tail_values: Vec<f32>,
}

/// One GDN layer's recurrent state, ready to persist.
pub struct GdnCapture {
    pub layer: LayerId,
    pub bytes: Vec<u8>,
}

/// What `PrefixSnapshotStore::load` hands back — restore-ready per-layer
/// data, sealed pages already reconstructed from disk (`SealedPage::from_bytes`,
/// which itself re-verifies the header, but not the content hash — callers
/// wanting the full immutability check should call `verify_sealed_pages`
/// after `TqkvPagedCache::restore_from_snapshot`).
pub struct LoadedSnapshot {
    pub token_count: usize,
    pub full_attention: Vec<LoadedFullAttentionLayer>,
    pub gdn: Vec<LoadedGdnLayer>,
}

pub struct LoadedFullAttentionLayer {
    pub layer: LayerId,
    pub position: u64,
    pub sealed: Vec<SealedPage>,
    pub tail_keys: Vec<f32>,
    pub tail_values: Vec<f32>,
}

pub struct LoadedGdnLayer {
    pub layer: LayerId,
    pub state: GdnState,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PersistedFullAttentionLayer {
    layer: u8,
    precision: u16,
    position: u64,
    page_content_ids: Vec<String>,
    tail_key_hex: String,
    tail_value_hex: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PersistedGdnLayer {
    layer: u8,
    content_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SnapshotManifest {
    schema_version: u32,
    token_count: usize,
    total_bytes: u64,
    full_attention: Vec<PersistedFullAttentionLayer>,
    gdn: Vec<PersistedGdnLayer>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct IndexEntry {
    prefix_hash: String,
    total_bytes: u64,
    created_unix: u64,
    last_used_unix: u64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct StoreIndex {
    schema_version: u32,
    entries: Vec<IndexEntry>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct RefcountTable {
    schema_version: u32,
    counts: HashMap<String, u64>,
}

pub struct PrefixSnapshotStore {
    root: PathBuf,
    quota_bytes: u64,
}

fn hex_id(id: &[u8; 32]) -> String {
    blake3::Hash::from_bytes(*id).to_hex().to_string()
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn hex_decode(text: &str) -> Result<Vec<u8>> {
    if !text.len().is_multiple_of(2) {
        return Err(ContextError::Invalid(format!("prefix store: odd-length hex {text}")).into());
    }
    (0..text.len())
        .step_by(2)
        .map(|i| {
            u8::from_str_radix(&text[i..i + 2], 16).map_err(|_| {
                ContextError::Invalid(format!("prefix store: bad hex byte in {text}")).into()
            })
        })
        .collect()
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Generic atomic byte write (temp file + `fsync` + rename + parent-dir
/// `fsync`), the same crash-safety pattern as `atomic_write_toml` (crate
/// invariant #9) but for binary blobs rather than serialized structs.
fn atomic_write_bytes(path: &Path, data: &[u8]) -> Result<()> {
    use std::io::Write;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp_path = path.with_extension("bin.tmp");
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&tmp_path)?;
    file.write_all(data)?;
    file.sync_all()?;
    drop(file);
    std::fs::rename(&tmp_path, path)?;
    if let Some(parent) = path.parent() {
        std::fs::File::open(parent)?.sync_all()?;
    }
    Ok(())
}

impl PrefixSnapshotStore {
    pub fn open(root: impl Into<PathBuf>, quota_bytes: u64) -> Result<Self> {
        let root = root.into();
        std::fs::create_dir_all(root.join("blobs"))?;
        std::fs::create_dir_all(root.join("snapshots"))?;
        Ok(Self { root, quota_bytes })
    }

    fn blob_path(&self, hex: &str) -> PathBuf {
        self.root
            .join("blobs")
            .join(&hex[0..2])
            .join(format!("{hex}.bin"))
    }

    fn index_path(&self) -> PathBuf {
        self.root.join("index.toml")
    }

    fn refcounts_path(&self) -> PathBuf {
        self.root.join("blob-refcounts.toml")
    }

    fn manifest_path(&self, prefix_hash_hex: &str) -> PathBuf {
        self.root
            .join("snapshots")
            .join(format!("{prefix_hash_hex}.toml"))
    }

    fn load_index(&self) -> Result<StoreIndex> {
        match std::fs::read_to_string(self.index_path()) {
            Ok(text) => Ok(toml::from_str(&text)
                .map_err(|e| ContextError::Invalid(format!("prefix index corrupt: {e}")))?),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(StoreIndex {
                schema_version: SCHEMA_VERSION,
                entries: Vec::new(),
            }),
            Err(e) => Err(e.into()),
        }
    }

    fn load_refcounts(&self) -> Result<RefcountTable> {
        match std::fs::read_to_string(self.refcounts_path()) {
            Ok(text) => Ok(toml::from_str(&text)
                .map_err(|e| ContextError::Invalid(format!("prefix refcounts corrupt: {e}")))?),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(RefcountTable {
                schema_version: SCHEMA_VERSION,
                counts: HashMap::new(),
            }),
            Err(e) => Err(e.into()),
        }
    }

    /// Exact token-prefix hash (spec §67): BLAKE3 over the little-endian
    /// token ID sequence.
    pub fn token_prefix_hash(tokens: &[u32]) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        for token in tokens {
            hasher.update(&token.to_le_bytes());
        }
        *hasher.finalize().as_bytes()
    }

    fn write_blob_if_absent(
        &self,
        refcounts: &mut RefcountTable,
        id_hex: &str,
        bytes: &[u8],
    ) -> Result<()> {
        let entry = refcounts.counts.entry(id_hex.to_string()).or_insert(0);
        if *entry == 0 {
            atomic_write_bytes(&self.blob_path(id_hex), bytes)?;
        }
        *entry += 1;
        Ok(())
    }

    /// Persists one prefix snapshot: writes any not-yet-present TQKV page
    /// and GDN state blobs (content-addressed, deduplicated against every
    /// existing snapshot), writes the snapshot's manifest, updates the
    /// index, and evicts least-recently-used snapshots if the store is
    /// over `quota_bytes` afterward. Returns the prefix hash.
    pub fn store(
        &self,
        tokens: &[u32],
        full_attention: &[FullAttentionCapture],
        gdn: &[GdnCapture],
    ) -> Result<[u8; 32]> {
        let mut refcounts = self.load_refcounts()?;
        let mut total_bytes = 0u64;

        let mut persisted_full_attention = Vec::with_capacity(full_attention.len());
        for capture in full_attention {
            let mut page_content_ids = Vec::with_capacity(capture.pages.len());
            for (id, bytes) in &capture.pages {
                let id_hex = hex_id(id);
                self.write_blob_if_absent(&mut refcounts, &id_hex, bytes)?;
                total_bytes += bytes.len() as u64;
                page_content_ids.push(id_hex);
            }
            let tail_key_hex = hex_encode(f32_slice_to_le_bytes(&capture.tail_keys).as_slice());
            let tail_value_hex = hex_encode(f32_slice_to_le_bytes(&capture.tail_values).as_slice());
            total_bytes += (tail_key_hex.len() + tail_value_hex.len()) as u64 / 2;
            persisted_full_attention.push(PersistedFullAttentionLayer {
                layer: capture.layer.0,
                precision: capture.precision.encoding_id(),
                position: capture.position,
                page_content_ids,
                tail_key_hex,
                tail_value_hex,
            });
        }

        let mut persisted_gdn = Vec::with_capacity(gdn.len());
        for capture in gdn {
            let id: [u8; 32] = *blake3::hash(&capture.bytes).as_bytes();
            let id_hex = hex_id(&id);
            self.write_blob_if_absent(&mut refcounts, &id_hex, &capture.bytes)?;
            total_bytes += capture.bytes.len() as u64;
            persisted_gdn.push(PersistedGdnLayer {
                layer: capture.layer.0,
                content_id: id_hex,
            });
        }

        let prefix_hash = Self::token_prefix_hash(tokens);
        let prefix_hash_hex = hex_id(&prefix_hash);

        // If this exact prefix was already stored, release its previous
        // blob references before writing the replacement manifest, so
        // re-storing the same prefix doesn't leak refcounts.
        self.remove_snapshot_internal(&mut refcounts, &prefix_hash_hex)?;

        let manifest = SnapshotManifest {
            schema_version: SCHEMA_VERSION,
            token_count: tokens.len(),
            total_bytes,
            full_attention: persisted_full_attention,
            gdn: persisted_gdn,
        };
        atomic_write_toml(&self.manifest_path(&prefix_hash_hex), &manifest)?;
        atomic_write_toml(&self.refcounts_path(), &refcounts)?;

        let mut index = self.load_index()?;
        index.entries.retain(|e| e.prefix_hash != prefix_hash_hex);
        let now = now_unix();
        index.entries.push(IndexEntry {
            prefix_hash: prefix_hash_hex,
            total_bytes,
            created_unix: now,
            last_used_unix: now,
        });
        atomic_write_toml(&self.index_path(), &index)?;

        self.evict_to_quota()?;
        Ok(prefix_hash)
    }

    /// Longest-exact-prefix lookup (spec §67, v1 exact-match only): checks
    /// whether the *entire* given token sequence has a stored snapshot.
    /// Callers wanting a true "longest prefix of this request" search
    /// should try successive truncations from longest to shortest (e.g. at
    /// their own checkpoint boundaries) and call this once per candidate.
    pub fn load(&self, tokens: &[u32], broker: &MemoryBroker) -> Result<Option<LoadedSnapshot>> {
        let prefix_hash_hex = hex_id(&Self::token_prefix_hash(tokens));
        let manifest_path = self.manifest_path(&prefix_hash_hex);
        let text = match std::fs::read_to_string(&manifest_path) {
            Ok(text) => text,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(e.into()),
        };
        let manifest: SnapshotManifest = toml::from_str(&text)
            .map_err(|e| ContextError::Invalid(format!("prefix manifest corrupt: {e}")))?;

        let mut full_attention = Vec::with_capacity(manifest.full_attention.len());
        for layer in &manifest.full_attention {
            let mut sealed = Vec::with_capacity(layer.page_content_ids.len());
            for id_hex in &layer.page_content_ids {
                let bytes = std::fs::read(self.blob_path(id_hex))?;
                sealed.push(SealedPage::from_bytes(&bytes)?);
            }
            full_attention.push(LoadedFullAttentionLayer {
                layer: LayerId(layer.layer),
                position: layer.position,
                sealed,
                tail_keys: le_bytes_to_f32_vec(&hex_decode(&layer.tail_key_hex)?),
                tail_values: le_bytes_to_f32_vec(&hex_decode(&layer.tail_value_hex)?),
            });
        }

        let mut gdn = Vec::with_capacity(manifest.gdn.len());
        for entry in &manifest.gdn {
            let bytes = std::fs::read(self.blob_path(&entry.content_id))?;
            gdn.push(LoadedGdnLayer {
                layer: LayerId(entry.layer),
                state: GdnState::from_bytes(broker, LayerId(entry.layer), &bytes)?,
            });
        }

        self.touch(&prefix_hash_hex)?;
        Ok(Some(LoadedSnapshot {
            token_count: manifest.token_count,
            full_attention,
            gdn,
        }))
    }

    fn touch(&self, prefix_hash_hex: &str) -> Result<()> {
        let mut index = self.load_index()?;
        if let Some(entry) = index
            .entries
            .iter_mut()
            .find(|e| e.prefix_hash == prefix_hash_hex)
        {
            entry.last_used_unix = now_unix();
            atomic_write_toml(&self.index_path(), &index)?;
        }
        Ok(())
    }

    fn remove_snapshot_internal(
        &self,
        refcounts: &mut RefcountTable,
        prefix_hash_hex: &str,
    ) -> Result<()> {
        let manifest_path = self.manifest_path(prefix_hash_hex);
        let text = match std::fs::read_to_string(&manifest_path) {
            Ok(text) => text,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(e) => return Err(e.into()),
        };
        let manifest: SnapshotManifest = toml::from_str(&text)
            .map_err(|e| ContextError::Invalid(format!("prefix manifest corrupt: {e}")))?;
        for layer in &manifest.full_attention {
            for id_hex in &layer.page_content_ids {
                self.release_blob(refcounts, id_hex)?;
            }
        }
        for entry in &manifest.gdn {
            self.release_blob(refcounts, &entry.content_id)?;
        }
        std::fs::remove_file(&manifest_path)?;
        Ok(())
    }

    fn release_blob(&self, refcounts: &mut RefcountTable, id_hex: &str) -> Result<()> {
        if let Some(count) = refcounts.counts.get_mut(id_hex) {
            *count = count.saturating_sub(1);
            if *count == 0 {
                refcounts.counts.remove(id_hex);
                let path = self.blob_path(id_hex);
                if path.exists() {
                    std::fs::remove_file(&path)?;
                }
            }
        }
        Ok(())
    }

    /// LRU disk quota enforcement (spec §66/§300): evicts oldest-used
    /// snapshots (manifest + any blobs whose refcount drops to zero) until
    /// total tracked bytes fits `quota_bytes`.
    fn evict_to_quota(&self) -> Result<()> {
        let mut index = self.load_index()?;
        let mut refcounts = self.load_refcounts()?;
        let mut total: u64 = index.entries.iter().map(|e| e.total_bytes).sum();
        if total <= self.quota_bytes {
            return Ok(());
        }
        index.entries.sort_by_key(|e| e.last_used_unix);
        let mut kept = Vec::new();
        for entry in index.entries {
            if total > self.quota_bytes {
                self.remove_snapshot_internal(&mut refcounts, &entry.prefix_hash)?;
                total = total.saturating_sub(entry.total_bytes);
            } else {
                kept.push(entry);
            }
        }
        atomic_write_toml(
            &self.index_path(),
            &StoreIndex {
                schema_version: SCHEMA_VERSION,
                entries: kept,
            },
        )?;
        atomic_write_toml(&self.refcounts_path(), &refcounts)?;
        Ok(())
    }

    /// Current tracked total bytes across all live snapshots (for tests
    /// and the quality/memory qualification doc).
    pub fn total_bytes(&self) -> Result<u64> {
        Ok(self
            .load_index()?
            .entries
            .iter()
            .map(|e| e.total_bytes)
            .sum())
    }

    pub fn snapshot_count(&self) -> Result<usize> {
        Ok(self.load_index()?.entries.len())
    }
}

fn f32_slice_to_le_bytes(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn le_bytes_to_f32_vec(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::Bytes;
    use crate::model::qwen36::attention::{BackendChoice, FullAttentionLayer};

    fn temp_dir(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("tqf-prefix-test-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    fn synthetic_tokens(n: usize) -> Vec<u32> {
        (0..n as u32).map(|i| 100 + i).collect()
    }

    /// A real, valid sealed-page blob (real header, real Q8 payload) for
    /// tests that exercise store-level mechanics (dedup, LRU, restart) and
    /// need `SealedPage::from_bytes` to actually succeed on load, without
    /// each test having to build a full `FullAttentionLayer` decode loop
    /// itself. `seed` varies the content so different calls produce
    /// different content IDs.
    fn synthetic_sealed_page(seed: f32) -> ([u8; 32], Vec<u8>) {
        let broker = MemoryBroker::new(Bytes(16 * 1024 * 1024));
        let mut layer = FullAttentionLayer::new_with_backend(
            &broker,
            LayerId(0),
            crate::context::tqkv::PAGE_TOKENS,
            BackendChoice::Tqkv(TqkvPrecision::Q8),
        )
        .unwrap();
        for i in 0..crate::context::tqkv::PAGE_TOKENS {
            let q = vec![0.0; 16 * 256];
            let gate = vec![4.0; 16 * 256];
            let k = vec![seed * (i as f32 + 1.0); 512];
            let v = vec![seed * (i as f32 + 1.0); 512];
            layer
                .decode_projected(q, &gate, k, &v, &vec![1.0; 256], &vec![1.0; 256])
                .unwrap();
        }
        let capture = layer.capture_tqkv_for_snapshot(LayerId(0)).unwrap();
        capture.pages.into_iter().next().unwrap()
    }

    #[test]
    fn token_prefix_hash_is_exact_and_order_sensitive() {
        let a = PrefixSnapshotStore::token_prefix_hash(&[1, 2, 3]);
        let b = PrefixSnapshotStore::token_prefix_hash(&[1, 2, 3]);
        let c = PrefixSnapshotStore::token_prefix_hash(&[1, 3, 2]);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    /// Builds a TQKV-backed `FullAttentionLayer` with a couple of sealed
    /// pages plus a tail, captures it, and round-trips it through the
    /// store to prove page-content-ID capture/restore is correct before
    /// testing the higher-level restart-reuse scenario below.
    #[test]
    fn full_attention_capture_round_trips_through_store_and_restore() {
        let dir = temp_dir("layer");
        let store = PrefixSnapshotStore::open(&dir, u64::MAX).unwrap();
        let broker = MemoryBroker::new(Bytes(64 * 1024 * 1024));

        let tokens = synthetic_tokens(300);
        let mut layer = FullAttentionLayer::new_with_backend(
            &broker,
            LayerId(3),
            crate::context::tqkv::PAGE_TOKENS + 20,
            BackendChoice::Tqkv(TqkvPrecision::Q8),
        )
        .unwrap();
        for i in 0..crate::context::tqkv::PAGE_TOKENS + 20 {
            let q = vec![0.1 * i as f32; 16 * 256];
            let gate = vec![4.0; 16 * 256];
            let k = vec![0.05 * i as f32; 512];
            let v = vec![0.02 * i as f32; 512];
            layer
                .decode_projected(q, &gate, k, &v, &vec![1.0; 256], &vec![1.0; 256])
                .unwrap();
        }
        let capture = layer.capture_tqkv_for_snapshot(LayerId(3)).unwrap();
        assert!(
            !capture.pages.is_empty(),
            "should have sealed at least one page"
        );

        store
            .store(&tokens, std::slice::from_ref(&capture), &[])
            .unwrap();
        let loaded = store.load(&tokens, &broker).unwrap().unwrap();
        assert_eq!(loaded.full_attention.len(), 1);
        let restored_layer = &loaded.full_attention[0];
        assert_eq!(restored_layer.sealed.len(), capture.pages.len());
        assert_eq!(restored_layer.tail_keys, capture.tail_keys);
        assert_eq!(restored_layer.position, capture.position);

        let mut fresh = FullAttentionLayer::new_with_backend(
            &broker,
            LayerId(3),
            crate::context::tqkv::PAGE_TOKENS + 20,
            BackendChoice::Tqkv(TqkvPrecision::Q8),
        )
        .unwrap();
        fresh
            .restore_tqkv_snapshot(
                restored_layer.sealed.clone(),
                restored_layer.tail_keys.clone(),
                restored_layer.tail_values.clone(),
                restored_layer.position,
            )
            .unwrap();
        assert_eq!(fresh.cache_len(), layer.cache_len());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn dedup_shares_one_blob_across_two_snapshots_with_an_identical_page() {
        let dir = temp_dir("dedup");
        let store = PrefixSnapshotStore::open(&dir, u64::MAX).unwrap();
        let shared_page_bytes = vec![7u8; 100];
        let shared_id = *blake3::hash(&shared_page_bytes).as_bytes();
        let capture_a = FullAttentionCapture {
            layer: LayerId(0),
            precision: TqkvPrecision::Q8,
            position: 5,
            pages: vec![(shared_id, shared_page_bytes.clone())],
            tail_keys: vec![],
            tail_values: vec![],
        };
        let capture_b = FullAttentionCapture {
            layer: LayerId(0),
            precision: TqkvPrecision::Q8,
            position: 5,
            pages: vec![(shared_id, shared_page_bytes.clone())],
            tail_keys: vec![],
            tail_values: vec![],
        };
        store.store(&[1, 2, 3], &[capture_a], &[]).unwrap();
        store.store(&[9, 9, 9], &[capture_b], &[]).unwrap();
        let refcounts = store.load_refcounts().unwrap();
        assert_eq!(refcounts.counts[&hex_id(&shared_id)], 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn lru_quota_evicts_the_least_recently_used_snapshot_first() {
        let dir = temp_dir("lru");
        let page_a = synthetic_sealed_page(1.0);
        let page_b = synthetic_sealed_page(2.0);
        // Each page is real (real header + Q8 payload); a quota just over
        // one page's size allows only one snapshot to survive.
        let store = PrefixSnapshotStore::open(&dir, page_a.1.len() as u64 + 32).unwrap();
        for (tokens, page) in [(&[1u32, 2, 3][..], page_a), (&[4, 5, 6][..], page_b)] {
            let capture = FullAttentionCapture {
                layer: LayerId(0),
                precision: TqkvPrecision::Q8,
                position: 0,
                pages: vec![page],
                tail_keys: vec![],
                tail_values: vec![],
            };
            store.store(tokens, &[capture], &[]).unwrap();
        }
        assert_eq!(store.snapshot_count().unwrap(), 1);
        // The most recently stored one ([4,5,6]) should have survived.
        let broker = MemoryBroker::new(Bytes(1024 * 1024));
        assert!(store.load(&[4, 5, 6], &broker).unwrap().is_some());
        assert!(store.load(&[1, 2, 3], &broker).unwrap().is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn restart_reuse_survives_dropping_and_reopening_the_store() {
        let dir = temp_dir("restart");
        let broker = MemoryBroker::new(Bytes(64 * 1024 * 1024));
        let tokens = synthetic_tokens(50);
        {
            let store = PrefixSnapshotStore::open(&dir, u64::MAX).unwrap();
            let page = synthetic_sealed_page(3.0);
            let capture = FullAttentionCapture {
                layer: LayerId(2),
                precision: TqkvPrecision::Q4,
                position: 50,
                pages: vec![page],
                tail_keys: vec![1.0, 2.0, 3.0],
                tail_values: vec![4.0, 5.0, 6.0],
            };
            store.store(&tokens, &[capture], &[]).unwrap();
            // `store` (the on-disk PrefixSnapshotStore handle) is dropped
            // here, simulating process exit — nothing survives in memory.
        }
        // A brand new store instance, as if the process restarted.
        let reopened = PrefixSnapshotStore::open(&dir, u64::MAX).unwrap();
        let loaded = reopened.load(&tokens, &broker).unwrap().unwrap();
        assert_eq!(loaded.full_attention.len(), 1);
        assert_eq!(loaded.full_attention[0].tail_keys, vec![1.0, 2.0, 3.0]);
        assert_eq!(loaded.full_attention[0].position, 50);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn gdn_state_round_trips_through_the_store() {
        let dir = temp_dir("gdn");
        let store = PrefixSnapshotStore::open(&dir, u64::MAX).unwrap();
        let broker = MemoryBroker::new(Bytes(64 * 1024 * 1024));
        let mut state = GdnState::new(&broker, LayerId(5)).unwrap();
        state.conv_tail_mut().reset();
        let capture = GdnCapture {
            layer: LayerId(5),
            bytes: state.to_bytes(),
        };
        store.store(&[11, 22], &[], &[capture]).unwrap();
        let loaded = store.load(&[11, 22], &broker).unwrap().unwrap();
        assert_eq!(loaded.gdn.len(), 1);
        assert_eq!(loaded.gdn[0].state.to_bytes(), state.to_bytes());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Audit finding C-04(a) (2026-08-20). `load` reads each blob and
    /// calls `SealedPage::from_bytes`, which never verifies the page's
    /// BLAKE3 (only `verify_sealed_pages` does, and the production
    /// restore caller in `model::qwen36::runtime` does not call it). The
    /// store is content-addressed, yet `load` never re-hashes the blob to
    /// check it against the filename it was found by, so a flipped
    /// payload byte becomes live context.
    ///
    /// Asserts spec §156's immutability guarantee at read time.
    /// Currently fails.
    #[test]
    fn a_tampered_blob_is_rejected_on_load() {
        let dir = temp_dir("tampered-blob");
        let store = PrefixSnapshotStore::open(&dir, u64::MAX).unwrap();
        let page = synthetic_sealed_page(1.0);
        let capture = FullAttentionCapture {
            layer: LayerId(0),
            precision: TqkvPrecision::Q8,
            position: 0,
            pages: vec![page],
            tail_keys: vec![],
            tail_values: vec![],
        };
        let tokens = synthetic_tokens(8);
        store.store(&tokens, &[capture], &[]).unwrap();

        let mut blob = None;
        for shard in std::fs::read_dir(dir.join("blobs")).unwrap() {
            for entry in std::fs::read_dir(shard.unwrap().path()).unwrap() {
                let path = entry.unwrap().path();
                if path.is_file() {
                    blob = Some(path);
                }
            }
        }
        let blob = blob.expect("one blob");
        let mut bytes = std::fs::read(&blob).unwrap();
        bytes[crate::context::tqkv::PAGE_HEADER_BYTES + 4] ^= 0xFF;
        std::fs::write(&blob, &bytes).unwrap();

        let broker = MemoryBroker::new(Bytes(64 * 1024 * 1024));
        assert!(
            store.load(&tokens, &broker).is_err(),
            "a blob whose content no longer matches its content address \
             must not be returned as live context"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Audit finding C-04(b) (2026-08-20). `restore_from_snapshot`
    /// assigns `self.sealed` wholesale: it checks neither the page
    /// header's `layer_id`, nor its key/value encodings against the
    /// cache's configured precision, nor the page count against
    /// `max_tokens`. Restoring two Q8 pages captured from layer 0 into a
    /// one-page Q4 cache on layer 7 therefore succeeds, doubles the
    /// resident token count, and never grows the broker reservation —
    /// violating locked invariant #4 (register before allocating) from an
    /// ordinary product path.
    ///
    /// Currently fails.
    #[test]
    fn restore_must_not_exceed_the_reserved_capacity() {
        use crate::context::tqkv::PAGE_TOKENS;

        let broker = MemoryBroker::new(Bytes(64 * 1024 * 1024));
        let mut source = FullAttentionLayer::new_with_backend(
            &broker,
            LayerId(0),
            PAGE_TOKENS * 2,
            BackendChoice::Tqkv(TqkvPrecision::Q8),
        )
        .unwrap();
        for i in 0..PAGE_TOKENS * 2 {
            let q = vec![0.0; 16 * 256];
            let gate = vec![4.0; 16 * 256];
            let k = vec![i as f32 + 1.0; 512];
            let v = vec![i as f32 + 1.0; 512];
            source
                .decode_projected(q, &gate, k, &v, &vec![1.0; 256], &vec![1.0; 256])
                .unwrap();
        }
        let capture = source.capture_tqkv_for_snapshot(LayerId(0)).unwrap();
        assert_eq!(capture.pages.len(), 2);

        let before = broker.snapshot().reserved;
        let mut victim = FullAttentionLayer::new_with_backend(
            &broker,
            LayerId(7),
            PAGE_TOKENS,
            BackendChoice::Tqkv(TqkvPrecision::Q4),
        )
        .unwrap();
        let reserved_for_one_page = broker.snapshot().reserved.0 - before.0;

        let pages: Vec<_> = capture
            .pages
            .iter()
            .map(|(_, bytes)| crate::context::tqkv::SealedPage::from_bytes(bytes).unwrap())
            .collect();
        let restored = victim.restore_tqkv_snapshot(pages, vec![], vec![], 2 * PAGE_TOKENS as u64);

        // Refusing the mismatched restore is correct; accepting it is
        // only correct if the broker was told about the extra page.
        if restored.is_ok() {
            assert!(
                broker.snapshot().reserved.0 - before.0 > reserved_for_one_page,
                "twice the reserved tokens are resident with no broker charge"
            );
        }
    }
}
