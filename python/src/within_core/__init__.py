"""within_core: convert Gemma 4 E4B into chunked Core AI assets for the Apple Neural Engine.

The pipeline, one module per step:

* :mod:`within_core.checkpoint` reads a pinned local checkpoint, verified byte for byte.
* :mod:`within_core.layers`, :mod:`within_core.attention` and :mod:`within_core.chunks`
  define the graph: decoder layers in channels-first form, grouped into chunks with
  host-held KV, plus a vocabulary-head chunk.
* :mod:`within_core.quantize` compresses the projections (MLP lut6, other projections
  INT8, head INT4).
* :mod:`within_core.export` authors each chunk as a Core AI asset and writes the chunk
  contract and runtime configuration.
* :mod:`within_core.tables` writes the FP16 embedding tables the host reads outside the
  graph.
* :mod:`within_core.package` assembles the downloadable package and its manifests.
* :mod:`within_core.convert` is the command line.
"""
