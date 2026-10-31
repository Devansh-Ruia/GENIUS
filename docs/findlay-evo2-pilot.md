# Findlay/Evo 2 timing pilot

This pilot freezes 20 calibration variants before any model scoring. It measures
descriptive separation and runtime for full `evo2_7b`. It does not fit confidence
calibration or compare the evidence panel with another system.

## Prepare the data

From the repository root, using the existing project environment:

```powershell
.venv/Scripts/python -m vus_panel.eval.pilot prepare --download
```

Preparation downloads missing files, verifies existing files, validates all
Findlay rows, and creates or verifies the frozen artifacts. It refuses corrupt,
incompatible, or incomplete freezes without replacing them. On Linux, use
`.venv/bin/python` in place of `.venv/Scripts/python`.

The publisher workbook remains in ignored `data/raw/`. The pinned chromosome
`NC_000017.10`, GRCh37.p13, remains in ignored `data/cache/reference/`. Download
receipts record file and sequence hashes. Validation checks one-based genomic
positions against `sequence[position - 1]` and confirms transcript alleles are
complements of the unchanged genomic alleles. No ClinVar annotations enter the
extracted records.

`data/cache/pilot/validation.json` contains the complete validation report.
`data/splits/calibration.txt` and `test.txt` retain the seed-0 stratified SHA256
split. `split_meta.json` records its provenance and checksums. `pilot20.json`
contains only selected IDs and aggregate metadata, selecting 7 FUNC, 6 INT, and
7 LOF calibration variants by SHA256 of `pilot-v1:0:{variant_id}`.

Copy the frozen `data/splits/` artifacts along with this code when reproducing
the run. Git preserves their bytes without line-ending conversion.

## Prepare the scoring service on the GPU instance

The service requires an existing compatible CUDA/Evo 2 environment on
`dying-azure-toucan`. CPU tests do not install CUDA dependencies or download
weights. The local application dependency list is unchanged.

The backend verifies the installed Arc source files and config against revision
`53f195997257c56c00e5ef8d33a54f5baad143a6`. The required checkpoint is the full
`evo2_7b` from revision `bda0089f92582d5baabf0f22d9fc85f3588f6b58`, with SHA256
`c66645929dc1b9c631f5be656da8726f38946315dc9167000a615dd626fcecf4`.

In that environment, materialize the revision-specific checkpoint explicitly:

```bash
python -m vus_panel.scoring.service materialize --directory data/cache/models
```

Start one service process, after transferring or preparing the verified reference:

```bash
python -m vus_panel.scoring.service serve \
  --checkpoint data/cache/models/bda0089f92582d5baabf0f22d9fc85f3588f6b58/evo2_7b.pt \
  --reference data/cache/reference/NC_000017.10.fasta \
  --reference-sequence-sha256 2463d54fb95c959a3fe545545fd11313986278881c7ded54721e172e7df1a41a
```

The default bind address is `127.0.0.1:8001`. Use an authenticated SSH tunnel to
reach it from the pilot runner. The service has no application authentication;
keep it on loopback. It exposes `GET /metadata` and `POST /score`. Score requests
contain exactly `chrom`, `pos`, `ref`, `alt`, and `genome_build`. Extra fields,
including answer-key fields with null values, are rejected.

## Frozen scoring protocol

Each context contains 8,192 genomic-forward bases with the variant at index
4096. REF and ALT differ at exactly one nucleotide. Out-of-bounds windows and
unsupported bases fail without shifting or complementing the context.

REF and ALT are scored separately with `batch_size=1`, `prepend_bos=False`,
`reduce_method="sum"`, and `average_reverse_complement=False`. Each total sums
8,191 next-token natural-log probabilities, including downstream terms. The
delta is the saved ALT total minus the saved REF total. These semantics follow
the [pinned Arc scoring source](https://github.com/ArcInstitute/evo2/blob/53f195997257c56c00e5ef8d33a54f5baad143a6/evo2/scoring.py).

One model load and one fixed synthetic REF/ALT warm-up pair precede pilot
scoring. The metadata records actual precision, dependencies, GPU information,
checkpoint/config hashes, model-loading time, and separate warm-up scores and
times. The client rejects changes in metadata or service session. Reference
score caching is disabled.

## Run and inspect artifacts

With an authenticated tunnel to the service:

```powershell
$env:EVO2_BACKEND_URL = 'http://127.0.0.1:8001'
.venv/Scripts/python -m vus_panel.eval.pilot run --selection data/splits/pilot20.json
```

Each invocation creates a unique ignored directory under `runs/`. The runner
retains every selected ID, including partial scores, failures, and unattempted
records after interruption. It uses no automatic retries or replacements.
Label-free raw outcomes are saved before Findlay labels are joined locally.

Inspect `outcomes.jsonl`, the joined evaluation table, `run_record.json`,
`metrics.json`, and `report.md` together. The report includes coverage, class
summaries, and descriptive LOF-positive AUROC using negative delta. Remaining
calibration runtime is estimated from successful scoring times, with startup
costs separate. Supply `--billing-rate-usd-per-hour` only when the actual instance
rate is known. An unavailable backend yields an incomplete run, not model scores.

## Sources

- [Publisher workbook](https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-018-0461-z/MediaObjects/41586_2018_461_MOESM3_ESM.xlsx)
- [NC_000017.10 reference sequence](https://www.ncbi.nlm.nih.gov/nuccore/NC_000017.10)
- [Pinned checkpoint revision](https://huggingface.co/arcinstitute/evo2_7b/tree/bda0089f92582d5baabf0f22d9fc85f3588f6b58)
