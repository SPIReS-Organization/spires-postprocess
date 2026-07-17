# spires-postprocess

Postprocessing for the [SPIReS](https://github.com/SPIReS-Organization) package
family. The implemented scientific core adjusts inversion snow fraction for
shade, viewable canopy obstruction, and fractional glacier ice. The example
notebooks also sketch future cloud gap-fill and tree-inpainting workflows.

Consumes the inversion `results` boundary defined in
[`spires-contract`](https://github.com/SPIReS-Organization/spires-contract).

## Canopy and ice adjustment

The public xarray-first API accepts inversion results plus ancillary and sensor
geometry arrays:

```python
from spires_postprocess import apply_snow_fraction_adjustments

adjusted = apply_snow_fraction_adjustments(
    inversion_results,
    canopy_fraction=io_data.ancillary["canopy_fraction"],
    ice_fraction=io_data.ancillary["ice_fraction"],
    sensor_zenith=io_data.scene["sensor_zenith"],
    sensor_azimuth=io_data.scene["sensor_azimuth"],
    slope=io_data.ancillary["slope"],
    aspect=io_data.ancillary["aspect"],
    average_vertical_crown_radius=4.644,
    average_horizontal_crown_radius=1.72,
)
```

Supplying `canopy_fraction` adds `canopy_adjusted_fsca`. Supplying
`ice_fraction` adds `ice_adjusted_fsca`. Supplying both adds both layers. The
input dataset and its original `fsca` and `fshade` variables remain unchanged.

The canopy-only output is

```text
O_canopy = clip(fshade + viewable_canopy_fraction, 0, 0.99)
canopy_adjusted_fsca = clip(fsca / (1 - O_canopy), 0, 1)
```

The ice output is calculated directly from the original `fsca`, rather than
sequentially from `canopy_adjusted_fsca`:

```text
O_ice = clip(fshade + viewable_canopy_fraction + ice_fraction, 0, 0.99)
ice_adjusted_fsca = max(clip(fsca / (1 - O_ice), 0, 1), ice_fraction)
```

When no canopy is supplied, the viewable-canopy term is zero. This reproduces
the daily shade/canopy/ice behavior of `SPIRES_2025_0_1` while keeping canopy
and ice independently selectable by pipeline configuration.

### Geometry and dimensions

Canopy view adjustment uses sensor zenith and the configured average vertical
and horizontal crown radii. When slope and aspect are omitted, the flat-terrain
Liu et al. (2004) GO-VGF form is used. When both are supplied, the terrain-aware
form also requires sensor azimuth. Aspect and sensor azimuth are expected in
degrees clockwise from north.

Results may use `(y, x)` or `(time, y, x)` dimensions. Static `(y, x)` canopy,
ice, slope, and aspect layers broadcast over time. Time-dependent sensor
geometry may use `(time, y, x)`. All supplied coordinates must match exactly;
reprojection and normalization belong to `spires-io`.

The implementation preserves NaNs and Dask-backed lazy arrays. Fractional
inputs and outputs are clipped to `[0, 1]`.

### Provenance

The added data variables record the formula identifier, included obstruction
terms, source labels, terrain use, crown radii where applicable, the 0.99
obscuration guard, and whether the fractional-ice lower bound was applied.

Fractional ice is not an inversion mask. `spires-io` preserves it as numeric
ancillary data without modifying `valid_inversion_mask`.

Note: the `lama` dependency (tree inpainting) may need to be sourced from a
specific distribution; the dependency list here is a placeholder for collaborators
to refine.
