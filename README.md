# virdentallab Blender Add-ons

Two complementary Blender add-ons for mesh alignment and deviation analysis,
built for dental and orthodontic workflows.

| Add-on | Panel | Purpose |
| --- | --- | --- |
| `mesh_alignment_part1.py` | View3D > Sidebar > Mesh Align | Align a test mesh to a reference mesh |
| `mesh_distance_3.py` | View3D > Sidebar > Distance Analysis | Measure and visualise the deviation between them |

They are designed to be used in that order: align first, then measure. Distance
Analysis picks up the pair already set in Mesh Alignment, so the two meshes do
not have to be selected twice.

## Requirements

- Blender 4.5 LTS or newer
- [Open3D](https://www.open3d.org/) 0.19 or newer

Open3D is not bundled with Blender and must be installed into the Python
environment Blender can see. With Blender closed:

```
pip install open3d --user
```

Both add-ons add the user site-packages directory to `sys.path` on load and
report the detected Open3D version in their panel.

## Mesh Alignment Pro

- **Paint selection** - paint a region on each mesh, then *Accept Selection*
  converts the painted vertices into a vertex selection that drives alignment.
- **Landmark pairs** - place matching points for an initial coarse alignment
  (SVD, centroid based).
- **ICP** - Open3D point-to-plane ICP, globally or restricted to the painted
  region, with optional downsampling.

## Mesh Distance Analysis

- Signed distance via Open3D raycasting, in both directions.
- **Hybrid sampling** - statistics from uniform or Poisson-disk surface samples
  (unbiased), visualisation from mesh vertices (sharp).
- Per-direction statistics, reported so they can be quoted directly:
  - `Mean (signed)` - trueness / bias
  - `Std Dev` - precision; pairs with the signed mean
  - `MAD` - mean absolute distance, the magnitude of the error
  - `RMS`, `95th Perc.`, `Max`
- Symmetric figures follow the standard definitions:
  Hausdorff is `max(h(A->B), h(B->A))` and HD95 is `max(P95(A->B), P95(B->A))`.
  The two directions are never pooled into a single percentile, which would
  weight whichever direction has more samples.
- Colour mapping with adjustable range, outlier highlighting, and optional
  Gaussian / Laplacian smoothing of the distance field.
- On-screen colour legend, histogram, and an interactive deviation probe.

### A note on Hausdorff

Maximum and Hausdorff distance are dominated by a handful of vertices and are
sensitive to loose vertices, small floating islands, and geometry merged in
later (attachments, buttons). MAD, RMS and HD95 are far more robust. Prefer
HD95 when reporting, and clean the mesh before relying on the maximum.

## Installation

Edit > Preferences > Add-ons > Install..., pick the `.py` file, enable it.
Install both to get the alignment-to-distance hand-off.

## Licence

GPL-3.0-or-later. Blender add-ons link against Blender's Python API and are
distributed under GPL-compatible terms.

## Author

[virdentallab](https://virdentallab.com)

Repository: https://github.com/zazaven/virdentallab-blender-addons
