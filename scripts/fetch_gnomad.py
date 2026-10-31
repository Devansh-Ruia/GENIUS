"""Cache gnomAD v2.1.1 (GRCh37) allele frequencies for the BRCA1 region.

    python scripts/fetch_gnomad.py                          # gnomAD API
    python scripts/fetch_gnomad.py --bigquery MY_GCP_PROJECT  # also pull from BigQuery to cross-check

Outputs in data/cache/:
    gnomad_v2_1_1.csv           one row per variant from the gnomAD API
    gnomad_v2_1_1_bq.csv        same from BigQuery (only with --bigquery)
    gnomad_v2_1_1_coverage.csv  per-position coverage, if the API returns it

A variant missing from the cache means "not seen in gnomAD", which only
counts as evidence where coverage is good. Never read it as AF = 0 blindly.
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd
import requests

from cache_common import UA, region, write_cache

API = "https://gnomad.broadinstitute.org/api"
DATASET = "gnomad_r2_1"  # = v2.1.1 in the API

VARIANTS_Q = """
query ($chrom: String!, $start: Int!, $stop: Int!) {
  region(chrom: $chrom, start: $start, stop: $stop, reference_genome: GRCh37) {
    variants(dataset: %s) {
      variant_id pos ref alt
      exome { ac an filters }
      genome { ac an filters }
    }
  }
}""" % DATASET

COVERAGE_Q = """
query ($chrom: String!, $start: Int!, $stop: Int!) {
  region(chrom: $chrom, start: $start, stop: $stop, reference_genome: GRCh37) {
    coverage(dataset: %s) {
      exome { pos mean over_20 }
      genome { pos mean over_20 }
    }
  }
}""" % DATASET


def gql(query: str, variables: dict) -> dict:
    r = requests.post(API, json={"query": query, "variables": variables}, headers=UA, timeout=120)
    r.raise_for_status()
    body = r.json()
    if body.get("errors"):
        raise RuntimeError(body["errors"])
    return body["data"]


def fetch_api(chrom: str, start: int, end: int, chunk: int = 20_000) -> pd.DataFrame:
    rows = []
    for s in range(start, end + 1, chunk):  # chunks keep each query under the API's region limit
        e = min(s + chunk - 1, end)
        data = gql(VARIANTS_Q, {"chrom": chrom, "start": s, "stop": e})
        for v in data["region"]["variants"]:
            ex, ge = v.get("exome") or {}, v.get("genome") or {}
            rows.append({
                "variant_id": v["variant_id"], "chrom": chrom, "pos": v["pos"],
                "ref": v["ref"], "alt": v["alt"],
                "exome_ac": ex.get("ac"), "exome_an": ex.get("an"),
                "exome_filters": ";".join(ex.get("filters") or []) if ex else None,
                "genome_ac": ge.get("ac"), "genome_an": ge.get("an"),
                "genome_filters": ";".join(ge.get("filters") or []) if ge else None,
            })
    df = pd.DataFrame(rows).drop_duplicates("variant_id")
    ac = df[["exome_ac", "genome_ac"]].fillna(0).sum(axis=1)
    an = df[["exome_an", "genome_an"]].fillna(0).sum(axis=1)
    df["ac"], df["an"] = ac.astype(int), an.astype(int)
    df["af"] = (ac / an.where(an > 0)).astype(float)
    return df.sort_values(["pos", "ref", "alt"]).reset_index(drop=True)


def fetch_coverage(chrom: str, start: int, end: int, chunk: int = 20_000) -> pd.DataFrame | None:
    try:
        parts = []
        for s in range(start, end + 1, chunk):
            e = min(s + chunk - 1, end)
            cov = gql(COVERAGE_Q, {"chrom": chrom, "start": s, "stop": e})["region"]["coverage"]
            for kind in ("exome", "genome"):
                for c in cov.get(kind) or []:
                    parts.append({"chrom": chrom, "pos": c["pos"], "kind": kind,
                                  "mean": c["mean"], "over_20": c["over_20"]})
        if not parts:
            return None
        return (pd.DataFrame(parts)
                .pivot_table(index=["chrom", "pos"], columns="kind", values=["mean", "over_20"])
                .pipe(lambda d: d.set_axis([f"{k}_{m}" for m, k in d.columns], axis=1))
                .reset_index())
    except Exception as exc:  # noqa: BLE001
        print(f"Coverage query failed ({exc}). Skipping coverage; flag this to the population agent.")
        return None


def fetch_bigquery(project: str, chrom: str, start: int, end: int) -> pd.DataFrame:
    from google.cloud import bigquery  # pip install google-cloud-bigquery

    client = bigquery.Client(project=project)
    frames = []
    for kind in ("exomes", "genomes"):
        table = f"bigquery-public-data.gnomAD.v2_1_1_{kind}__chr{chrom}"
        sql = f"""
        SELECT reference_name, start_position, reference_bases, alt.alt AS alt, alt.AC AS ac, AN AS an
        FROM `{table}` v, UNNEST(v.alternate_bases) alt
        WHERE start_position BETWEEN @s AND @e"""
        cfg = bigquery.QueryJobConfig(query_parameters=[
            bigquery.ScalarQueryParameter("s", "INT64", start - 1),  # BigQuery start_position is 0-based
            bigquery.ScalarQueryParameter("e", "INT64", end - 1),
        ])
        try:
            df = client.query(sql, job_config=cfg).to_dataframe()
        except Exception as exc:  # noqa: BLE001
            sys.exit(f"BigQuery failed on {table}: {exc}\nCheck the schema with: bq show --schema {table}")
        df["pos"] = df["start_position"] + 1
        df["kind"] = kind[:-1]
        frames.append(df)
    df = pd.concat(frames)
    df["variant_id"] = chrom + "-" + df["pos"].astype(str) + "-" + df["reference_bases"] + "-" + df["alt"]
    wide = df.pivot_table(index="variant_id", columns="kind", values=["ac", "an"], aggfunc="first")
    wide.columns = [f"{k}_{m}" for m, k in wide.columns]
    return wide.reset_index()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bigquery", metavar="GCP_PROJECT", help="also pull from BigQuery for a cross-check")
    args = ap.parse_args()

    chrom, start, end, how = region()
    print(f"region {chrom}:{start}-{end} ({how})")
    meta = {"region": f"{chrom}:{start}-{end}", "region_basis": how, "dataset": "gnomAD v2.1.1"}

    df = fetch_api(chrom, start, end)
    print(f"API: {len(df)} variants, {(df['ac'] > 0).sum()} with AC > 0, max AF {df['af'].max():.4g}")
    write_cache(df, "gnomad_v2_1_1", {**meta, "source": f"{API} ({DATASET})"})

    cov = fetch_coverage(chrom, start, end)
    if cov is not None:
        write_cache(cov, "gnomad_v2_1_1_coverage", {**meta, "source": f"{API} coverage ({DATASET})"})

    if args.bigquery:
        bq = fetch_bigquery(args.bigquery, chrom, start, end)
        write_cache(bq, "gnomad_v2_1_1_bq", {**meta, "source": "bigquery-public-data.gnomAD v2_1_1"})


if __name__ == "__main__":
    main()
