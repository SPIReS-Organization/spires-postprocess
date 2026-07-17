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

## Snow radiative products

Three independent xarray-first functions evaluate normalized lookup datasets:

```python
from spires_postprocess import (
    compute_delta_vis,
    compute_radiative_forcing,
    compute_snow_albedo,
)

albedo = compute_snow_albedo(
    inversion_results,
    cosine_solar_zenith=io_data.scene["cosine_solar_zenith"],
    cosine_illumination=io_data.scene["cosine_illumination"],
    lookup=albedo_lookup,
)
delta_vis = compute_delta_vis(
    inversion_results,
    cosine_solar_zenith=io_data.scene["cosine_solar_zenith"],
    lookup=forcing_lookup,
)
forcing = compute_radiative_forcing(
    inversion_results,
    cosine_solar_zenith=io_data.scene["cosine_solar_zenith"],
    lookup=forcing_lookup,
)
```

`grain_size` is effective snow grain radius in micrometers. Recognized unit
metadata are `um`, `µm`, `μm`, `micrometer`, and `micrometers`; absent metadata
uses this documented SPIReS convention. `dust_concentration` is ppm (or
`parts per million`) and is transformed to LUT mass fraction by division by
1,000,000. Soot mass fraction is fixed to zero.

The albedo call adds exactly these dimensionless variables:

- `albedo_clean_flat`
- `albedo_dirty_flat`
- `albedo_clean_terrain_corrected`
- `albedo_dirty_terrain_corrected`

Flat products evaluate `(cosine_solar_zenith, cosine_solar_zenith)`. The two
terrain-corrected albedos evaluate
`(cosine_solar_zenith, cosine_illumination)`. `compute_delta_vis()` adds the
dimensionless flat-surface `delta_vis` product, and
`compute_radiative_forcing()` adds the independent flat-surface
`radiative_forcing` product in `W m-2`. Neither forcing call requires a runtime
albedo product.

### Normalized lookup contract

Lookups are supplied as `xarray.Dataset` objects, normally normalized from
production NetCDF by `spires-io`. `spires-postprocess` does not read lookup
files. Variables have these exact ordered dimensions and units:

| Variable | Ordered dimensions | Units |
| --- | --- | --- |
| `clean_albedo` | `cosine_solar_zenith, cosine_illumination, sqrt_grain_radius_um` | `1` |
| `dirty_albedo` | the preceding three, `dust_mass_fraction, soot_mass_fraction` | `1` |
| `delta_vis` | the dirty-albedo dimensions | `1` |
| `radiative_forcing` | the dirty-albedo dimensions | `W m-2` |

Each consumed variable is validated independently. Coordinates must be finite,
real numeric, unique, and strictly increasing; table values must be finite real
numbers. Zero must lie in every consumed soot domain. Albedo LUT values outside
`[-1e-6, 1 + 1e-6]` are rejected.

Interpolation is linear with no extrapolation. LUT boundary values are valid;
missing or out-of-domain inputs produce `NaN` only at affected pixels. Albedo
roundoff is clipped to `[0, 1]` after valid LUT content has been checked. Result
layouts may be `(y, x)` or `(time, y, x)`, with static `(y, x)` geometry
broadcast over time. Shared coordinates must match exactly. Dask-backed scene
inputs remain lazy; lookup axes and values are intentionally eager.

Every product records its lookup variable and axis ranges, grain-radius and
dust transformations, zero-soot assumption, and flat or terrain-corrected
geometry. Lookup `source`, SHA-256 checksum, atmosphere model, and dust model
attributes are propagated when present. Input dataset variables and attributes
remain unchanged, and existing same-named products are deterministically
replaced in the returned copy.

The grain-radius meaning, dust conversion, geometry labels, and output names
are candidates for migration into `spires-contract` when that contract work
resumes.

Note: the `lama` dependency (tree inpainting) may need to be sourced from a
specific distribution; the dependency list here is a placeholder for collaborators
to refine.
