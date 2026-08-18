//! Phase 22 expert tile layouts (spec §294, §34): the canonical Phase 6
//! layout is one whole-region tile per matrix; a Phase 22 conversion may
//! emit finer neuron-width sub-tiles so the runtime can treat a tile, not a
//! whole expert, as its cache unit. The wire records are unchanged - a
//! tiled layout is simply more `ExpertTileRecord`s per expert - so the
//! format needs no migration (the Phase 22 exit-gate requirement).
//!
//! Q4_K block geometry fixes what is tileable. A Q4_K block is 256 columns
//! x 1 row (144 bytes):
//! - gate/up are [512 neurons, 2048 hidden] stored row-major, so the neuron
//!   dimension is the *row* dimension: any row width {64,128,256,512}
//!   divides cleanly.
//! - down is [2048 hidden, 512 neurons], so the neuron dimension is the
//!   *column* dimension: tiles must be multiples of the 256-column block,
//!   i.e. {256,512}. 64/128-neuron down tiling is impossible without
//!   splitting Q4_K blocks and is deliberately not offered.

use crate::format::tqf::records::{ExpertMatrix, ExpertTileRecord};
use crate::ids::TileId;

/// Bytes of one Q4_K block (256 columns x 1 row).
pub const Q4K_BLOCK_BYTES: u32 = 144;
/// Qwen3.6 routed-expert neuron count (spec §117 LOCKED geometry).
pub const EXPERT_NEURONS: u32 = 512;
/// Hidden width consumed by gate/up rows.
pub const EXPERT_HIDDEN: u32 = 2048;
/// Hidden width consumed by down rows.
pub const DOWN_HIDDEN: u32 = 2048;
/// Q4_K blocks per gate/up row (2048 columns / 256).
const GATEUP_BLOCKS_PER_ROW: u32 = 8;
/// Q4_K blocks per down row (512 columns / 256).
const DOWN_BLOCKS_PER_ROW: u32 = 2;

/// Tile width in the neuron dimension. `Whole` is the Phase 6 canonical
/// layout; the finer widths are Phase 22 A/B candidates. `Mixed128` keeps
/// the measured best row tiling for gate/up while leaving down whole - the
/// spec §34 allows different matrices to use different tilings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NeuronWidth {
    Whole,
    N64,
    N128,
    N256,
    Mixed128,
}

impl NeuronWidth {
    pub fn from_env(var: &str) -> Option<Self> {
        Self::from_env_value(&std::env::var(var).ok()?)
    }

    /// Parses the raw env value (`whole`, `64`, `128`, `256`, `mixed128`).
    pub fn from_env_value(value: &str) -> Option<Self> {
        match value.trim() {
            "" | "whole" => Some(Self::Whole),
            "64" => Some(Self::N64),
            "128" => Some(Self::N128),
            "256" => Some(Self::N256),
            "mixed128" | "mixed" => Some(Self::Mixed128),
            _ => None,
        }
    }

    /// Row width for gate/up tiles. `Whole` keeps the canonical merged
    /// GateUp region; `Mixed128` tiles gate/up at 128 rows.
    pub fn gateup_row_width(self) -> u32 {
        match self {
            Self::Whole => EXPERT_NEURONS,
            Self::Mixed128 | Self::N128 => 128,
            Self::N64 => 64,
            Self::N256 => 256,
        }
    }

    /// Column width for down tiles. 64/128 map to the smallest legal
    /// block-aligned width (256): splitting a Q4_K block across tiles is
    /// not representable.
    pub fn down_col_width(self) -> u32 {
        match self {
            Self::Whole | Self::Mixed128 => EXPERT_NEURONS,
            Self::N64 | Self::N128 | Self::N256 => 256,
        }
    }

    pub fn is_whole(self) -> bool {
        matches!(self, Self::Whole)
    }
}

/// Whether `width` divides the given region sizes into clean tiles.
/// `Whole` is always representable (one merged GateUp tile + one Down
/// tile, any sizes). The writer falls back to the canonical layout when
/// this is false, so synthetic/non-Qwen fixtures keep round-tripping.
pub fn layout_is_divisible(
    width: NeuronWidth,
    gate_bytes: u32,
    up_bytes: u32,
    down_bytes: u32,
) -> bool {
    if width == NeuronWidth::Whole {
        return true;
    }
    let gateup = gateup_tile_bytes(width.gateup_row_width());
    let down = down_tile_bytes(width.down_col_width());
    gate_bytes % gateup == 0 && up_bytes % gateup == 0 && down_bytes % down == 0
}

/// Number of tiles a layout emits for one expert's superextent.
pub fn tile_count(width: NeuronWidth) -> usize {
    if width == NeuronWidth::Whole {
        // Canonical Phase 6: one whole-region GateUp tile + one Down tile.
        return 2;
    }
    let gateup_tiles = (EXPERT_NEURONS / width.gateup_row_width()) as usize;
    let down_tiles = (EXPERT_NEURONS / width.down_col_width()) as usize;
    2 * gateup_tiles + down_tiles
}

fn gateup_tile_bytes(width: u32) -> u32 {
    width * GATEUP_BLOCKS_PER_ROW * Q4K_BLOCK_BYTES
}

fn down_tile_bytes(width: u32) -> u32 {
    DOWN_HIDDEN * (width / 256) * Q4K_BLOCK_BYTES
}

/// Generates the ordered tile-record table for one expert's canonical
/// `gate`/`up`/`down` byte regions under `width`. Records are emitted in
/// physical order (gate tiles, up tiles, down tiles) with monotonically
/// increasing `relative_offset`s and `tile_id`s; `neuron_start` is the row
/// offset within its own matrix region (gate/up) or column offset (down).
/// `Whole` collapses gate+up into the single whole-region GateUp tile of
/// the Phase 6 canonical layout (two records total).
pub fn tile_plan(
    width: NeuronWidth,
    gate_bytes: u32,
    up_bytes: u32,
    down_bytes: u32,
) -> Vec<ExpertTileRecord> {
    let gateup_width = width.gateup_row_width();
    let down_width = width.down_col_width();
    let expected = tile_count(width);
    let mut tiles = Vec::with_capacity(expected);
    let mut tile_id = 0u16;
    let mut offset = 0u32;

    if width == NeuronWidth::Whole {
        // Canonical Phase 6 layout, any region sizes: one merged GateUp
        // tile and one whole-region Down tile.
        tiles.push(ExpertTileRecord {
            matrix: ExpertMatrix::GateUp,
            tile_id: TileId(0),
            neuron_start: 0,
            neuron_count: 0,
            relative_offset: 0,
            stored_bytes: gate_bytes + up_bytes,
            quant_layout_id: 0, // filled by the writer, which owns quant_layout_id
            flags: 0,
        });
        tiles.push(ExpertTileRecord {
            matrix: ExpertMatrix::Down,
            tile_id: TileId(1),
            neuron_start: 0,
            neuron_count: 0,
            relative_offset: gate_bytes + up_bytes,
            stored_bytes: down_bytes,
            quant_layout_id: 0,
            flags: 0,
        });
        return tiles;
    }
    for (bytes, tile_bytes) in [
        (gate_bytes, gateup_tile_bytes(gateup_width)),
        (up_bytes, gateup_tile_bytes(gateup_width)),
    ] {
        debug_assert_eq!(bytes % tile_bytes, 0, "tile width must divide matrix bytes");
        let count = bytes / tile_bytes;
        for index in 0..count {
            tiles.push(ExpertTileRecord {
                matrix: ExpertMatrix::GateUp,
                tile_id: TileId(tile_id),
                neuron_start: (index * gateup_width) as u16,
                neuron_count: gateup_width as u16,
                relative_offset: offset,
                stored_bytes: tile_bytes,
                quant_layout_id: 0,
                flags: 0,
            });
            offset += tile_bytes;
            tile_id += 1;
        }
    }
    let down_tile_bytes = down_tile_bytes(down_width);
    debug_assert_eq!(
        down_bytes % down_tile_bytes,
        0,
        "tile width must divide matrix bytes"
    );
    for index in 0..down_bytes / down_tile_bytes {
        tiles.push(ExpertTileRecord {
            matrix: ExpertMatrix::Down,
            tile_id: TileId(tile_id),
            neuron_start: (index * down_width) as u16,
            neuron_count: down_width as u16,
            relative_offset: offset,
            stored_bytes: down_tile_bytes,
            quant_layout_id: 0,
            flags: 0,
        });
        offset += down_tile_bytes;
        tile_id += 1;
    }
    debug_assert_eq!(offset, gate_bytes + up_bytes + down_bytes);
    debug_assert_eq!(tiles.len(), expected);
    tiles
}

/// Outcome of validating a tile table against an expert's stored extent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TilePartition {
    /// Canonical Phase 6 layout: one whole-region GateUp tile followed by
    /// one whole-region Down tile.
    Canonical,
    /// A finer Phase 22 layout whose tiles exactly partition the extent in
    /// physical order, matrix-consistent (GateUp tiles cover the gate+up
    /// region, Down tiles cover the down region).
    Tiled,
}

/// Verifies that a tile table is a valid partition of an expert extent:
/// ordered, contiguous, non-overlapping, exactly covering
/// `gate+up+down` bytes, and that every record's matrix matches the region
/// its byte range falls in. The caller supplies the canonical matrix sizes.
pub fn classify_partition(
    tiles: &[ExpertTileRecord],
    gate_up_bytes: u32,
    down_bytes: u32,
) -> Result<TilePartition, String> {
    let total = gate_up_bytes + down_bytes;
    if tiles.is_empty() {
        return Err("expert extent has no tile records".to_string());
    }
    let mut expected_offset = 0u32;
    for (index, tile) in tiles.iter().enumerate() {
        if tile.stored_bytes == 0 {
            return Err(format!("tile {index} has zero stored bytes"));
        }
        if tile.relative_offset != expected_offset {
            return Err(format!(
                "tile {index} starts at byte {} but byte {} is the next unclaimed offset",
                tile.relative_offset, expected_offset
            ));
        }
        let end = tile
            .relative_offset
            .checked_add(tile.stored_bytes)
            .ok_or_else(|| format!("tile {index} byte range overflows u32"))?;
        let region = match tile.matrix {
            ExpertMatrix::GateUp => end <= gate_up_bytes,
            ExpertMatrix::Down => tile.relative_offset >= gate_up_bytes && end <= total,
        };
        if !region {
            return Err(format!(
                "tile {index} ({:?}) spans bytes {}..{end} outside its matrix region",
                tile.matrix, tile.relative_offset
            ));
        }
        expected_offset = end;
    }
    if expected_offset != total {
        return Err(format!(
            "tile table covers {expected_offset} bytes of a {total} byte extent"
        ));
    }
    if tiles.len() == 2 {
        return Ok(TilePartition::Canonical);
    }
    Ok(TilePartition::Tiled)
}

/// Generic partition sanity check (no matrix-region knowledge): tiles must
/// be ordered, contiguous, non-empty, and exactly cover `total` bytes. The
/// reader applies this to every expert at open time; Qwen-specific
/// region/matrix consistency is `classify_partition`'s job, applied later
/// by the weight loader.
pub fn partition_is_contiguous(tiles: &[ExpertTileRecord], total: u32) -> bool {
    let mut expected = 0u32;
    for tile in tiles {
        if tile.stored_bytes == 0 || tile.relative_offset != expected {
            return false;
        }
        match expected.checked_add(tile.stored_bytes) {
            Some(end) => expected = end,
            None => return false,
        }
    }
    expected == total
}

#[cfg(test)]
mod tests {
    use super::*;

    const GATE_BYTES: u32 = 512 * 8 * Q4K_BLOCK_BYTES;
    const UP_BYTES: u32 = GATE_BYTES;
    const DOWN_BYTES: u32 = 2048 * 2 * Q4K_BLOCK_BYTES;

    fn records_cover_extent(width: NeuronWidth) -> Vec<ExpertTileRecord> {
        let mut plan = tile_plan(width, GATE_BYTES, UP_BYTES, DOWN_BYTES);
        for (index, tile) in plan.iter_mut().enumerate() {
            tile.tile_id = TileId(index as u16);
        }
        plan
    }

    #[test]
    fn every_candidate_layout_partitions_the_extent_exactly() {
        for width in [
            NeuronWidth::Whole,
            NeuronWidth::N64,
            NeuronWidth::N128,
            NeuronWidth::N256,
            NeuronWidth::Mixed128,
        ] {
            let plan = records_cover_extent(width);
            assert_eq!(plan.len(), tile_count(width), "{width:?}");
            match classify_partition(&plan, GATE_BYTES + UP_BYTES, DOWN_BYTES) {
                Ok(partition) => assert_eq!(
                    partition,
                    if width.is_whole() {
                        TilePartition::Canonical
                    } else {
                        TilePartition::Tiled
                    },
                    "{width:?}"
                ),
                Err(error) => panic!("{width:?} layout rejected: {error}"),
            }
        }
    }

    #[test]
    fn tile_widths_describe_real_q4k_byte_sizes() {
        assert_eq!(tile_count(NeuronWidth::Whole), 2);
        // gate/up 128-row tiles: 128 x 8 blocks x 144 B; down 256-col tiles:
        // 2048 rows x 1 block x 144 B.
        assert_eq!(tile_count(NeuronWidth::N128), 10);
        let plan = tile_plan(NeuronWidth::N128, GATE_BYTES, UP_BYTES, DOWN_BYTES);
        assert_eq!(plan[0].stored_bytes, 128 * 8 * Q4K_BLOCK_BYTES);
        assert_eq!(plan[8].matrix, ExpertMatrix::Down);
        assert_eq!(plan[8].stored_bytes, 2048 * Q4K_BLOCK_BYTES);
        assert_eq!(plan[8].neuron_start, 0);
        assert_eq!(plan[9].neuron_start, 256);
        assert_eq!(
            plan[9].relative_offset,
            plan[8].relative_offset + plan[8].stored_bytes
        );
        // 64-row gate/up tiles plus two 256-col down tiles.
        assert_eq!(tile_count(NeuronWidth::N64), 18);
        // Mixed128 keeps down whole.
        assert_eq!(tile_count(NeuronWidth::Mixed128), 9);
        let mixed = tile_plan(NeuronWidth::Mixed128, GATE_BYTES, UP_BYTES, DOWN_BYTES);
        assert_eq!(mixed[8].matrix, ExpertMatrix::Down);
        assert_eq!(mixed[8].stored_bytes, DOWN_BYTES);
    }

    #[test]
    fn malformed_partitions_are_rejected() {
        let mut plan = records_cover_extent(NeuronWidth::N128);
        // Gap: first tile claims fewer bytes than its record's extent.
        plan[1].relative_offset += 1;
        assert!(classify_partition(&plan, GATE_BYTES + UP_BYTES, DOWN_BYTES).is_err());

        let mut plan = records_cover_extent(NeuronWidth::N128);
        // A GateUp tile pointing into the down region.
        let last = plan.len() - 1;
        plan[last].matrix = ExpertMatrix::GateUp;
        assert!(classify_partition(&plan, GATE_BYTES + UP_BYTES, DOWN_BYTES).is_err());

        // Truncated coverage.
        let mut plan = records_cover_extent(NeuronWidth::N128);
        plan.pop();
        assert!(classify_partition(&plan, GATE_BYTES + UP_BYTES, DOWN_BYTES).is_err());

        assert!(classify_partition(&[], GATE_BYTES + UP_BYTES, DOWN_BYTES).is_err());
    }

    #[test]
    fn generic_partition_check_rejects_gaps_overlaps_and_partial_coverage() {
        let total = GATE_BYTES + UP_BYTES + DOWN_BYTES;
        for width in [NeuronWidth::Whole, NeuronWidth::N128, NeuronWidth::Mixed128] {
            assert!(partition_is_contiguous(&records_cover_extent(width), total));
        }
        let mut plan = records_cover_extent(NeuronWidth::N128);
        plan[2].relative_offset += 64;
        assert!(!partition_is_contiguous(&plan, total));
        let mut plan = records_cover_extent(NeuronWidth::N128);
        plan[2].relative_offset -= 64;
        assert!(!partition_is_contiguous(&plan, total));
        let plan = records_cover_extent(NeuronWidth::N128);
        assert!(!partition_is_contiguous(&plan[..plan.len() - 1], total));
    }

    #[test]
    fn env_control_parses_all_candidate_widths() {
        let var = "TQF_TEST_TILE_NEURONS_PARSE";
        std::env::set_var(var, "64");
        assert_eq!(NeuronWidth::from_env(var), Some(NeuronWidth::N64));
        std::env::set_var(var, "128");
        assert_eq!(NeuronWidth::from_env(var), Some(NeuronWidth::N128));
        std::env::set_var(var, "256");
        assert_eq!(NeuronWidth::from_env(var), Some(NeuronWidth::N256));
        std::env::set_var(var, "mixed");
        assert_eq!(NeuronWidth::from_env(var), Some(NeuronWidth::Mixed128));
        std::env::set_var(var, "whole");
        assert_eq!(NeuronWidth::from_env(var), Some(NeuronWidth::Whole));
        std::env::set_var(var, "banana");
        assert_eq!(NeuronWidth::from_env(var), None);
        std::env::remove_var(var);
        assert_eq!(NeuronWidth::from_env(var), None);
    }
}
