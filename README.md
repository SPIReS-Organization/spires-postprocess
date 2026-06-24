# spires-postprocess

Postprocessing for the [SPIReS](https://github.com/SPIReS-Organization) package
family: cloud gap-fill (temporal interpolation, shade correction) and tree
masking / inpainting of inversion results.

Consumes the inversion `results` boundary defined in
[`spires-contract`](https://github.com/SPIReS-Organization/spires-contract).

> **Status:** scaffolding only. The package layout and dependency on
> `spires-contract` are in place; the implementation is to be filled in. See
> `examples/` for the workflows this package will cover.

Note: the `lama` dependency (tree inpainting) may need to be sourced from a
specific distribution; the dependency list here is a placeholder for collaborators
to refine.
