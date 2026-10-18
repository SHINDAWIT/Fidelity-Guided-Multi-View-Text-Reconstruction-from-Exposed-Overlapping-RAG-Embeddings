# Fidelity-Guided Multi-View Text Reconstruction
## from Exposed Overlapping RAG Embeddings

[한국어](README.md) · [Paper PDF](paper/ICISC_2026_paper__52.pdf) · [Implementation guide](docs/IMPLEMENTATION.md) · [Results](docs/EXPERIMENTS.md)

**Paper:** Submitted to ICISC 2026; under review as of October 18, 2026.

## Research overview

Independently inverting overlapping text embeddings produces several imperfect reconstructions of shared content. Their insertions and deletions can also shift boundaries. This study investigates how to select and assemble these correlated outputs.

The project reuses the public pretrained GTR/Vec2Text models and the whole-view cosine fidelity score already computed by Vec2Text. Project-specific processing validates raw-window inputs, selects overlap owners, assembles outputs, and separates reconstruction metrics from evaluation-only diagnostics. It does not train a new inverter or confidence model.

## Main result

Both fixed-coordinate methods use the **same eight Top-1 views per span**, the same coordinates, and the same clipping, null, length, and tie handling. Only the ownership rule changes.

Evaluation: **60 overlap-stable English Natural Questions spans × 136 reference positions = 8,160 positions**.

| Method | Matched reference positions | Recovery | Total edits |
| --- | ---: | ---: | ---: |
| FC–L: lower-index owner | 7,823 / 8,160 | 95.87% | 415 |
| FC–F: corresponding-fidelity owner | 8,045 / 8,160 | **98.59%** | **137** |

The within-sample difference is **+222 matches, +2.72 percentage points, and 278 fewer edits**. This is a **post-hoc exploratory** analysis. Recovery measures matched reference positions, not full-document exact reconstruction.

Progressive CP–F adds 16 matches over FC–F but increases insertions from 22 to 41 and edits from 137 to 140, while exact spans decrease from 38/60 to 35/60. This is a whole-method comparison: boundary inference and length handling also differ.

## What to inspect

- [Input contracts](src/overlap_poc/chunking.py#L226): compare independently tokenized raw substrings against intended token slices; validate 31 content tokens plus EOS.
- [Existing-model adapter](src/overlap_poc/reconstruction.py#L175): pretrained Vec2Text inference and observation of existing candidate scores.
- [Fixed-coordinate assembly](src/overlap_poc/stage5d_assembly.py#L163): deterministic ownership, nulls, clipping, and triple coverage.
- [Progressive assembly](src/overlap_poc/stage6a_source_free_assembly.py#L235): accumulated suffix / next-prefix alignment and whole-view ownership.
- [Evaluation](src/overlap_poc/stage5d_assembly.py#L200): matched positions, substitutions, insertions, deletions, edit totals, and span exact match.
- [Resume guards](src/overlap_poc/stage6c_strong.py#L149): run fingerprint and completed-key order validation.

I studied the existing Vec2Text input and inference pipeline, worked on output selection and assembly, and compared reconstruction errors. The analysis separates input-representation differences from reconstruction errors and measures how assembly rules affect each error type.

The recorded stack includes Python, pretrained Vec2Text, PyTorch GPU inference in a CUDA environment, Transformers/tokenizers, Hugging Face datasets, NumPy, YAML/JSON, and Linux.

## Inspect the published numbers

```bash
python3 tools/verify_summary.py
```

This uses only Python's standard library to check aggregate arithmetic, CSV agreement, the paper's checksum, and copied-source integrity. Research-pipeline execution requires a separate environment.

The publication includes selected source code, configurations, existing synthetic-input tests, and aggregate JSON/CSV. It omits benchmark text, actual reconstruction/token arrays, embeddings, model weights, and checkpoints. It is therefore not a self-contained reproduction of the paper's end-to-end GPU experiment. Some original runners refer to withheld frozen artifacts.

The aggregate checker runs separately from the original tests and GPU inference. Results remain specific to the pretrained GTR/Vec2Text setting and the selected stable English NQ subset. Source-free assembly is distinct from reference-based evaluation and oracle diagnostics. The eight-versus-five-vector overlap/no-overlap comparison is not an equal-budget estimate.

See [reproducibility boundaries](docs/REPRODUCIBILITY.md), [aggregate records](results/README.md), and [materials and attribution](NOTICE.md).
