# VAST Image Stitcher

Lightweight batch stitcher for Zeiss .CZI tiled 3D acquisitions.

Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python vast-image-stitcher.py --help
```

Example usage (non-interactive):

```bash
python vast-image-stitcher.py --input /path/to/czi_files --output ./stitched --bf 1 --gfp 2 --overlap 0.10
```

Dependencies are listed in `requirements.txt` (numpy, tifffile, czifile).

Notes

- Filenames must contain `Block<integer>` tokens to group channel stacks.
- If present, `pt<integer>` can be used to order tiles when `--sort-pt` is enabled.
- The tool performs intensity gain-matching and feather blending, but not
  geometric registration.
