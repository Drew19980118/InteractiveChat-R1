"""Compatibility shim for outlines 0.0.46.

The old outlines release required by vLLM 0.6.3 imports
``pyairports.airports`` eagerly, even when airport constrained generation is
not used. The package published as pyairports==0.0.1 contains only metadata.
This local module supplies the unused symbol so JSON guided decoding can load.
"""