"""Placement and numerics verification (ANE-EXEC-v1) of the converted chunks on a device.

Pipeline position: after conversion (:mod:`within_core.convert`) and a device recording. A
host app runs every chunk's ``step_1`` on three arms of the same asset (Neural Engine,
GPU, CPU) inside an all-process ``xctrace`` recording, and two idle recordings follow; see
the README for the recording and the run folder it produces.

* :mod:`within_core.verify.trace` reads the exported trace tables.
* :mod:`within_core.verify.review` computes the verdict from them.
* :mod:`within_core.verify.reference` recomputes each chunk's outputs in FP32 from the
  recorded input bytes, for the numerics criterion.

``python -m within_core.verify review|reference`` is the command line.

What this can and cannot prove: the proof is by exclusion plus coincidence, not
attribution. It cannot show which operation ran on the ANE; the idle control bounds, and
does not exclude, ANE rows from other processes inside a window; it covers only the
instrumented stages.
"""
