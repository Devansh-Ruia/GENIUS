# Demo reviews

`panel_reports.json` holds three real panel reviews (`mock: false`) that the page opens with, through `GET /demo`. They come from live `POST /review` calls on 2026-11-01, saved in `runs/20261101T171637Z-demo-reviews/` on the machine that ran them. That folder holds each response as received, `run_record.json` and `selection.json`. `runs/` is not committed.

All three are calibration-split variants, picked by the first match in `data/splits/calibration.txt` order:

| Role | Variant |
|---|---|
| FUNC | `GRCh37-17-41197722-T-C` |
| LOF | `GRCh37-17-41197729-T-C` |
| Evo 2 and phyloP point opposite ways | `GRCh37-17-41201135-T-A` |

"Opposite" means the Evo 2 calibration percentile is at most 10 with phyloP at most 0.055, or at least 90 with phyloP at least 4.269. Those are the bottom and top calibration quintiles in `data/cache/calibration_refs.json`.

Confidence is the chair's own judgement and is not calibrated. The sae agent is planned, not in this build, so it reports unavailable.
