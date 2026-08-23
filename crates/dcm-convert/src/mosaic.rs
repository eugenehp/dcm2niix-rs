//! Siemens / UIH mosaic demosaic (`nii_demosaic`).
//!
//! A mosaic stores many slices as tiles in one 2D DICOM frame. This module
//! unpacks them into a contiguous `[nz][ny][nx]` volume in scan order.

use rayon::prelude::*;

/// Demosaic a single 2D mosaic slab into `n_mosaic_slices` slices (row-major voxels).
///
/// Returns `(voxels, out_cols, out_rows)`. UIH may use a non-square tile grid.
pub fn demosaic_f32(
    in_img: &[f32],
    cols: usize,
    rows: usize,
    n_mosaic_slices: i32,
    is_uih: bool,
) -> (Vec<f32>, usize, usize) {
    if n_mosaic_slices < 2 {
        return (in_img.to_vec(), cols, rows);
    }
    let n = n_mosaic_slices as usize;
    let n_col = (n as f64).sqrt().ceil() as usize;
    let n_row = if is_uih {
        (n as f64 / n_col as f64).ceil() as usize
    } else {
        n_col
    };
    let out_cols = cols / n_col;
    let out_rows = rows / n_row;
    let slice_vox = out_cols * out_rows;
    let mut out = vec![0.0f32; slice_vox * n];
    let line = cols;
    let tile_row_stride = cols * out_rows;

    // Parallel over tiles: each writes a disjoint `[slice_vox]` region.
    out.par_chunks_mut(slice_vox)
        .enumerate()
        .for_each(|(tile, dst)| {
            let tile_col = tile % n_col;
            let tile_row = tile / n_col;
            let mut src = tile_row * tile_row_stride + tile_col * out_cols;
            let mut out_pos = 0usize;
            for _y in 0..out_rows {
                dst[out_pos..out_pos + out_cols].copy_from_slice(&in_img[src..src + out_cols]);
                src += line;
                out_pos += out_cols;
            }
        });
    (out, out_cols, out_rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn demosaic_2x2_grid() {
        let img: Vec<f32> = (0..16).map(|v| v as f32).collect();
        let (out, c, r) = demosaic_f32(&img, 4, 4, 4, false);
        assert_eq!((c, r), (2, 2));
        assert_eq!(out.len(), 2 * 2 * 4);
        // Tile (0,0) = [0,1 / 4,5]
        assert_eq!(&out[0..4], &[0.0, 1.0, 4.0, 5.0]);
    }

    #[test]
    fn demosaic_matches_serial_order() {
        // 3x3 tile grid, 8 slices (last tile unused in C++ style still written? we take n=8)
        let cols = 6;
        let rows = 6;
        let img: Vec<f32> = (0..cols * rows).map(|v| v as f32).collect();
        let (out, c, r) = demosaic_f32(&img, cols, rows, 8, false);
        assert_eq!((c, r), (2, 2));
        // First tile top-left 2x2
        assert_eq!(&out[0..4], &[0.0, 1.0, 6.0, 7.0]);
    }
}
