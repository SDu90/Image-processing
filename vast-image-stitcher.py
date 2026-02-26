#!/usr/bin/env python3
"""
Batch stitcher for Zeiss .CZI tiles (3D stacks), written as a reusable image-analysis pipeline.

===============================================================================
OVERVIEW
===============================================================================
This script batch-stitches tiled 3D microscopy acquisitions stored as Zeiss .CZI files.
It is designed for pipelines where tile placement is known (fixed overlap) but
tile-to-tile intensity can drift at borders (seams).

For each sample ("Animal") and for each channel group (e.g., BF and GFP), the script:
  1) Finds all .czi files in INPUT_PATH
  2) Splits them into groups using a BlockID parsed from the filename: "Block<integer>"
  3) Builds stitched mosaics by placing tiles along X using a FIXED overlap (pixel-count based)
  4) Normalizes intensity in the overlap (median gain matching; per-channel if channels exist)
  5) Feather-blends overlap pixels to reduce visible seams
  6) Auto-detects stitch direction (left→right vs right→left) from overlap similarity
     WITHOUT changing file order:
        - "forward": mosaic grows left→right
        - "reverse": mosaic grows right→left (still reading files in the same order)
  7) Saves a 3D BigTIFF stack and a 2D Z-max projection (MIP)

===============================================================================
WHAT THIS SCRIPT DOES *NOT* DO
===============================================================================
- No geometric registration (no feature matching, no phase correlation, no drift/rotation correction).
- No vertical stitching in Y (assumes tiles already align in Y).
- No non-rigid stitching.
If your tiles have positional drift, you need an alignment method in addition to this.

===============================================================================
FILENAME REQUIREMENTS
===============================================================================
The script uses filename tokens for grouping (and optionally ordering):
  - REQUIRED to split BF/GFP (or any groups): "Block<integer>" anywhere in the name
      Example: "...AcquisitionBlock2..." -> Block ID = 2
  - OPTIONAL for ordering within each sample: "pt<integer>"
      Example: "..._pt16..." -> tile index = 16

If SORT_TILES_BY_PT is False (default), ordering relies on filename sorting.
If SORT_TILES_BY_PT is True, the script orders tiles within each chunk by pt number.

===============================================================================
EDITING & USAGE
===============================================================================
- Prefer using the command-line options rather than editing the script.
  Example: `python vast-image-stitcher.py --input ./my_czis --output ./stitched --bf 1 --gfp 2 --overlap 0.10`
- If your CZI axis ordering differs, adjust `load_czi_clean()` only.

===============================================================================
DEPENDENCIES
===============================================================================
pip install numpy tifffile czifile
"""

import os
import re
import argparse
import logging
import numpy as np
import tifffile as tiff
import czifile
from tifffile import TiffWriter

__version__ = "0.1.0"


# =============================================================================
# 2) FILENAME PARSING + GENERAL HELPERS
# =============================================================================

def get_metadata(filename: str):
    """
    Parse a filename to extract:
      - Block ID: looks for 'Block<digits>' (case-insensitive)
      - pt ID:    looks for 'pt<digits>' (case-insensitive)

    Returns:
      (block_id:int, pt_id:int)
    Missing tokens return 0.

    Example:
      "20250815_xxx_AcquisitionBlock2_pt16.czi" -> (2, 16)
    """
    block_match = re.search(r"Block(\d+)", filename, re.IGNORECASE)
    pt_match = re.search(r"pt(\d+)", filename, re.IGNORECASE)
    block_id = int(block_match.group(1)) if block_match else 0
    pt_id = int(pt_match.group(1)) if pt_match else 0
    return block_id, pt_id


def natural_key(s: str):
    """
    Natural sort key (so 'file2' comes before 'file10').
    This keeps tiles in a human-expected order when filenames contain numbers.
    """
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"([0-9]+)", s)]


# =============================================================================
# 3) LOADING .CZI INTO A CONSISTENT SHAPE
# =============================================================================

def load_czi_clean(path: str):
    """
    Load a Zeiss CZI file as a NumPy array and normalize axes.

    Target output shapes:
      - (Z, Y, X)    for single-channel data
      - (Z, Y, X, C) for multi-channel data

    Why this is needed:
      Zeiss CZI files can contain extra dimensions (Scene, Time, Channel, etc.).
      czifile returns whatever dimension ordering is stored. We:
        1) squeeze singleton dims
        2) if remaining array is 4D, we assume the FIRST axis corresponds to channel
           and move it to the end (channels-last).
           This matches your earlier working approach.

    IMPORTANT:
      If you print shapes and see unexpected ordering, adjust here.
      This is the only place that should need dataset-specific axis changes.
    """
    with czifile.CziFile(path) as czi:
        data = np.squeeze(czi.asarray())

    # If 4D, assume axis0 behaves like channels and move to channels-last:
    # (C, Z, Y, X) -> (Z, Y, X, C)
    if data.ndim == 4:
        data = np.moveaxis(data, 0, -1)

    # Enforce expected dimensionality
    if data.ndim not in (3, 4):
        raise ValueError(
            f"Unexpected CZI shape after squeeze/normalize: {data.shape}\n"
            f"File: {path}\n"
            f"Adjust load_czi_clean() to match your CZI axis ordering."
        )

    return data


# =============================================================================
# 4) DIRECTION DECISION USING OVERLAP INTENSITY SIMILARITY
# =============================================================================

def _to_gray_2d(vol):
    """
    Convert a 3D tile volume into a robust 2D image for scoring overlaps.

    Input:
      vol: (Z,Y,X) or (Z,Y,X,C)

    Output:
      img2d: (Y,X) float32

    We use robust medians:
      - If channels exist: median across channels -> (Z,Y,X)
      - Then: median across Z -> (Y,X)

    This makes overlap scoring less sensitive to rare bright puncta or noise.
    """
    v = vol
    if v.ndim == 4:
        v = np.median(v, axis=-1)          # collapse channels
    return np.median(v, axis=0).astype(np.float32)  # collapse Z


def _edge_score(tileA, tileB, ov, side="forward", eps=1e-6):
    """
    Compute a mismatch score between overlap edges of two neighboring tiles.

    Lower score = better match.

    side="forward":
      compare right edge of tileA with left edge of tileB
    side="reverse":
      compare left edge of tileA with right edge of tileB

    Steps:
      1) create robust 2D projections from both tiles (median projections)
      2) extract overlap strips of width ov
      3) gain-normalize B strip to A strip using median ratio
      4) compute mean absolute difference as the mismatch score
    """
    A = _to_gray_2d(tileA)
    B = _to_gray_2d(tileB)

    if side == "forward":
        A_strip = A[:, -ov:]   # A right edge
        B_strip = B[:, :ov]    # B left edge
    elif side == "reverse":
        A_strip = A[:, :ov]    # A left edge
        B_strip = B[:, -ov:]   # B right edge
    else:
        raise ValueError("side must be 'forward' or 'reverse'")

    # Gain normalize to reduce brightness drift effects
    gain = np.median(A_strip) / (np.median(B_strip) + eps)
    Bn = B_strip * gain

    # Mean absolute difference (robust and easy to interpret)
    return float(np.mean(np.abs(A_strip - Bn)))


def choose_direction_by_overlap(vols, overlap=0.10):
    """
    Decide whether the file-ordered tile sequence is best interpreted as:
      - "forward" (left→right) or
      - "reverse" (right→left)

    We do NOT reorder files. We only decide how the mosaic grows on the canvas.

    Method:
      Compute total mismatch across all neighboring pairs for both hypotheses:
        forward_total = sum( right(tile_i) vs left(tile_{i+1}) )
        reverse_total = sum( left(tile_i)  vs right(tile_{i+1}) )
      Pick the smaller total mismatch.

    Returns:
      (direction:str, forward_score:float, reverse_score:float)
    """
    W = vols[0].shape[2]  # X dimension
    step = int(round(W * (1 - overlap)))
    ov = W - step
    if ov <= 0:
        raise ValueError(f"Invalid overlap={overlap}: computed ov={ov} for W={W}")

    forward_score = 0.0
    reverse_score = 0.0

    for i in range(len(vols) - 1):
        forward_score += _edge_score(vols[i], vols[i + 1], ov, side="forward")
        reverse_score += _edge_score(vols[i], vols[i + 1], ov, side="reverse")

    direction = "forward" if forward_score <= reverse_score else "reverse"
    return direction, forward_score, reverse_score


# =============================================================================
# 5) STITCHING: FIXED OVERLAP + GAIN MATCH + FEATHER BLEND
# =============================================================================

def stitch_fixed_overlap_autodir(tile_paths, overlap=0.10, eps=1e-6,
                                 direction="auto", verbose=True):
    """
    Stitch tiles using a fixed overlap fraction, with gain matching and feather blending.

    Inputs:
      tile_paths : list[str]
          Paths to tiles, in the desired order (file order is preserved).
      overlap : float
          Fractional overlap in X (0.10 means 10% overlap).
      eps : float
          Small value to avoid divide-by-zero in gain computations.
      direction : str
          "auto", "forward", or "reverse"
      verbose : bool
          Print direction decision details

    Output:
      out : np.ndarray float32
          Stitched volume:
            - (Z,Y,totalX) or
            - (Z,Y,totalX,C)

    Notes on "reverse":
      Files are processed in the same order, but the first tile is placed at the
      RIGHT end of the output canvas and subsequent tiles are prepended to the LEFT.
      This preserves file order while allowing reverse acquisition direction.
    """
    # Load all tiles as arrays (with error context)
    vols = []
    for p in tile_paths:
        try:
            vols.append(load_czi_clean(p))
        except Exception as e:
            raise RuntimeError(f"Failed to load CZI '{p}': {e}") from e
    first = vols[0]

    # Determine dimensionality and verify shapes match
    if first.ndim == 3:
        Z, H, W = first.shape
        C = None
    elif first.ndim == 4:
        Z, H, W, C = first.shape
    else:
        raise ValueError(f"Unexpected tile shape: {first.shape}")

    for i, v in enumerate(vols[1:], start=1):
        if v.shape != first.shape:
            raise ValueError(f"Tile {i} shape {v.shape} != first tile shape {first.shape}")

    # Convert overlap fraction to pixel step/overlap count
    step = int(round(W * (1 - overlap)))  # how many new pixels we advance per tile
    ov = W - step                         # overlap width in pixels
    if ov <= 0 or step <= 0:
        raise ValueError(f"Bad overlap={overlap}. step={step}, ov={ov}, W={W}")

    # Decide direction if auto
    if direction == "auto":
        direction, fwd, rev = choose_direction_by_overlap(vols, overlap=overlap)
        if verbose:
            print(f"      Auto-direction: {direction} (forward_score={fwd:.4g}, reverse_score={rev:.4g})")
    elif direction not in ("forward", "reverse"):
        raise ValueError("direction must be 'auto', 'forward', or 'reverse'")

    # Create output canvas
    total_width = W + (len(vols) - 1) * step
    out_shape = (Z, H, total_width) if C is None else (Z, H, total_width, C)
    out = np.zeros(out_shape, dtype=np.float32)

    # Feather weights across overlap: 0 → 1
    w = np.linspace(0.0, 1.0, ov, dtype=np.float32)
    w = w.reshape((1, 1, ov)) if C is None else w.reshape((1, 1, ov, 1))

    # -------------------------------------------------------------------------
    # FORWARD: mosaic grows left→right
    # -------------------------------------------------------------------------
    if direction == "forward":
        out[..., :W] = first.astype(np.float32)
        x = step  # where the next tile begins

        for i in range(1, len(vols)):
            tile = vols[i].astype(np.float32)

            # Overlap regions
            out_ov = out[..., x:x + ov]   # existing overlap in output
            tile_ov = tile[..., :ov]      # incoming tile's left overlap strip

            # Gain matching in overlap (median). If multi-channel, do per-channel.
            if C is None:
                gain = np.median(out_ov) / (np.median(tile_ov) + eps)
                tile *= gain
            else:
                gains = np.zeros((C,), dtype=np.float32)
                for c in range(C):
                    gains[c] = np.median(out_ov[..., c]) / (np.median(tile_ov[..., c]) + eps)
                tile *= gains.reshape((1, 1, 1, C))

            # Feather blend overlap + copy remainder
            out[..., x:x + ov] = out_ov * (1.0 - w) + tile[..., :ov] * w
            out[..., x + ov:x + W] = tile[..., ov:W]

            x += step

    # -------------------------------------------------------------------------
    # REVERSE: mosaic grows right→left (file order preserved)
    # -------------------------------------------------------------------------
    else:
        right_start = total_width - W
        out[..., right_start:right_start + W] = first.astype(np.float32)

        x = right_start - step  # where the next tile's non-overlap will go on the left

        # For reverse growth, the incoming overlap is on the RIGHT edge of the tile,
        # so we reverse the feather weights to blend correctly.
        wr = w[..., ::-1] if C is None else w[..., ::-1, :]

        for i in range(1, len(vols)):
            tile = vols[i].astype(np.float32)

            # Existing overlap is at the left edge of the already-placed region
            out_ov = out[..., x + step:x + step + ov]
            tile_ov = tile[..., -ov:]  # incoming tile's right overlap

            # Gain match in overlap
            if C is None:
                gain = np.median(out_ov) / (np.median(tile_ov) + eps)
                tile *= gain
            else:
                gains = np.zeros((C,), dtype=np.float32)
                for c in range(C):
                    gains[c] = np.median(out_ov[..., c]) / (np.median(tile_ov[..., c]) + eps)
                tile *= gains.reshape((1, 1, 1, C))

            # Blend overlap
            out[..., x + step:x + step + ov] = out_ov * (1.0 - wr) + tile[..., -ov:] * wr

            # Copy the non-overlapping left part of the tile
            out[..., x:x + step] = tile[..., :step]

            x -= step

    return out


# =============================================================================
# 6) MAIN BATCH RUNNER
# =============================================================================

def main():
    """
    Interactive entry point.
    User provides the Block IDs that correspond to BF and GFP.

    The script then:
      - collects all .czi files
      - selects BF and GFP subsets by Block ID
      - chunks each subset into groups of TILES_PER_SAMPLE
      - stitches each group and writes outputs into:
          OUTPUT_PATH/Animal_<n>/
    """
    # Parse CLI args
    parser = argparse.ArgumentParser(description="Batch stitcher for Zeiss .CZI tiles")
    parser.add_argument("--input", "-i", default='.', help="Input folder containing .czi files")
    parser.add_argument("--output", "-o", default='./stitched', help="Output folder for stitched results")
    parser.add_argument("--bf", type=int, help="BF Block ID (e.g., 1)")
    parser.add_argument("--gfp", type=int, help="GFP Block ID (e.g., 2)")
    parser.add_argument("--overlap", type=float, default=0.10, help="Fractional X overlap (e.g., 0.10)")
    parser.add_argument("--tiles-per-sample", type=int, default=4, help="Tiles per sample (default: 4)")
    parser.add_argument("--sort-pt", action="store_true", default=False, help="Sort tiles by pt index if present")
    parser.add_argument("--direction", choices=["auto", "forward", "reverse"], default="auto", help="Stitch direction: 'auto'|'forward'|'reverse' (default: auto)")
    parser.add_argument("--version", action="version", version=__version__)

    args = parser.parse_args()

    in_path = args.input
    out_base = args.output
    overlap = args.overlap
    tiles_per_sample = args.tiles_per_sample
    sort_tiles = args.sort_pt
    cli_direction = args.direction

    # configure basic logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Prompt for block IDs if not provided
    if args.bf is None:
        bf_id = int(input("BF Block ID (e.g., 1): "))
    else:
        bf_id = args.bf
    if args.gfp is None:
        gfp_id = int(input("GFP Block ID (e.g., 2): "))
    else:
        gfp_id = args.gfp

    # Validate input path
    if not os.path.isdir(in_path):
        raise RuntimeError(f"Input path not found or not a directory: {in_path}")

    # Find all .czi files and sort them naturally (only regular files)
    all_files = sorted(
        [f for f in os.listdir(in_path)
         if f.lower().endswith(".czi") and os.path.isfile(os.path.join(in_path, f))],
        key=natural_key
    )

    # Split files into channel groups using Block IDs
    bf_master = [f for f in all_files if get_metadata(f)[0] == bf_id]
    gfp_master = [f for f in all_files if get_metadata(f)[0] == gfp_id]

    # Determine number of samples based on BF count (keeps original behavior)
    num_samples = len(bf_master) // tiles_per_sample
    if num_samples <= 0:
        raise RuntimeError(
            "No samples detected.\n"
            "Check INPUT_PATH, file extensions, and whether filenames include the expected Block ID tokens."
        )

    # Warn about leftover tiles or mismatched BF/GFP counts
    if len(bf_master) != num_samples * tiles_per_sample:
        print(f"WARNING: BF file count ({len(bf_master)}) is not a multiple of {tiles_per_sample}; extra files ignored.")
    if len(gfp_master) < num_samples * tiles_per_sample:
        print(f"WARNING: GFP has fewer files ({len(gfp_master)}) than expected ({num_samples*tiles_per_sample}).")

    # Ensure output base exists
    try:
        os.makedirs(out_base, exist_ok=True)
    except Exception as e:
        raise RuntimeError(f"Unable to create output directory '{out_base}': {e}") from e

    # Process each sample ("Animal")
    for i in range(num_samples):
        animal_dir = os.path.join(out_base, f"Animal_{i + 1}")
        os.makedirs(animal_dir, exist_ok=True)

        dir_label = cli_direction
        print(f"\n>>> Animal {i + 1} | overlap={overlap*100:.1f}% | direction={dir_label}")

        # Stitch BF and GFP separately
        for name, master in [("BF", bf_master), ("GFP", gfp_master)]:
            # Select this animal's tile filenames
            chunk_files = master[i * tiles_per_sample:(i + 1) * tiles_per_sample]
            if not chunk_files:
                continue

            # Optional re-ordering using pt index
            if sort_tiles:
                chunk_files = sorted(chunk_files, key=lambda f: get_metadata(f)[1])

            # Convert filenames to full paths
            chunk_paths = [os.path.join(in_path, f) for f in chunk_files]

            print(f"    Stitching {name} (file order preserved):")
            for f in chunk_files:
                b, pt = get_metadata(f)
                print(f"      - {f}  (Block={b}, pt={pt})")

            # Perform stitching (float32 output). stitch_fixed_overlap_autodir accepts 'auto' or fixed.
            try:
                stitched_f = stitch_fixed_overlap_autodir(
                    chunk_paths,
                    overlap=overlap,
                    direction=cli_direction,
                    verbose=True
                )
            except Exception as e:
                logging.error("Stitching failed for %s sample %d (%s): %s", name, i + 1, chunk_files, e)
                continue

            # Convert to uint16 for saving (common microscopy format)
            final = np.clip(stitched_f, 0, 65535).astype(np.uint16)

            # Save Z max projection for quick QC
            mip = final.max(axis=0)
            # Save MIP
            try:
                tiff.imwrite(os.path.join(animal_dir, f"{name}_MIP.tif"), mip)
            except Exception as e:
                logging.error("Failed to write MIP for %s: %s", os.path.join(animal_dir, f"{name}_MIP.tif"), e)
                continue

            # Save 3D volume as BigTIFF, writing one Z-slice per TIFF page
            out3d = os.path.join(animal_dir, f"{name}_3D.tif")
            try:
                with TiffWriter(out3d, bigtiff=True) as tif:
                    for z in range(final.shape[0]):
                        tif.write(final[z], photometric="minisblack")
            except Exception as e:
                logging.error("Failed to write 3D TIFF for %s: %s", out3d, e)
                # Attempt to remove partial file if created
                try:
                    if os.path.exists(out3d):
                        os.remove(out3d)
                except Exception:
                    pass
                continue

    print("\nBatch Complete.")


if __name__ == "__main__":
    main()
