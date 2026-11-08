"""Download LAZ and/or DOP tiles listed in tiles_duelmen.csv.

- LAZ -> data/raw/laz/   (issue #5)
- DOP -> data/raw/dop/   (issue #6)

Re-runnable: files that already exist with the expected size (from the
index, stored in the CSV) are skipped. Downloads go to a .part file first
and are renamed only when complete, so aborted runs leave no broken files.
Files whose size does not match are retried, failures are logged to
data/raw/<dataset>/download_failures.log and the script exits with code 1.

Usage (inside the activated env):
    conda run -n pcsearch python pipeline/01_download/download_tiles.py laz
    conda run -n pcsearch python pipeline/01_download/download_tiles.py dop
    conda run -n pcsearch python pipeline/01_download/download_tiles.py all --workers 4
    conda run -n pcsearch python pipeline/01_download/download_tiles.py laz --limit 1   # quick test
    conda run -n pcsearch python pipeline/01_download/download_tiles.py all --check    # only verify
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[2]
TILE_LIST = REPO / "pipeline/01_download/tiles_duelmen.csv"

DATASETS = {
    "laz": {
        "base_url": "https://www.opengeodata.nrw.de/produkte/geobasis/hm/3dm_l_las/3dm_l_las/",
        "file_col": "laz_file",
        "size_col": "laz_size",
        "out_dir": REPO / "data/raw/laz",
    },
    "dop": {
        "base_url": "https://www.opengeodata.nrw.de/produkte/geobasis/lusat/akt/dop/dop_jp2_f10/",
        "file_col": "dop_file",
        "size_col": "dop_size",
        "out_dir": REPO / "data/raw/dop",
    },
}

CHUNK = 1 << 20  # 1 MiB
RETRIES = 3
TIMEOUT = (30, 120)  # connect, read


@dataclass
class Job:
    dataset: str
    name: str
    url: str
    size: int
    path: Path


def load_jobs(dataset: str, tile_list: Path) -> list[Job]:
    cfg = DATASETS[dataset]
    jobs = []
    with tile_list.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            name = row[cfg["file_col"]].strip()
            if not name:
                print(f"  [{dataset}] no file for tile {row['tile_id']} in tile list, skipped")
                continue
            jobs.append(Job(
                dataset=dataset,
                name=name,
                url=cfg["base_url"] + name,
                size=int(row[cfg["size_col"]]),
                path=cfg["out_dir"] / name,
            ))
    return jobs


def is_complete(job: Job) -> bool:
    return job.path.exists() and job.path.stat().st_size == job.size


def download(job: Job, session: requests.Session) -> tuple[Job, str]:
    """Return (job, status) with status 'skipped', 'ok' or 'FAILED: <reason>'."""
    if is_complete(job):
        return job, "skipped"
    job.path.parent.mkdir(parents=True, exist_ok=True)
    part = job.path.with_name(job.path.name + ".part")
    last_err = ""
    for attempt in range(1, RETRIES + 1):
        try:
            with session.get(job.url, stream=True, timeout=TIMEOUT) as r:
                r.raise_for_status()
                with part.open("wb") as f:
                    for chunk in r.iter_content(CHUNK):
                        f.write(chunk)
            got = part.stat().st_size
            if got != job.size:
                raise IOError(f"size mismatch: got {got}, expected {job.size}")
            part.replace(job.path)
            return job, "ok"
        except Exception as e:  # noqa: BLE001 - log any failure and retry
            last_err = f"{type(e).__name__}: {e}"
            part.unlink(missing_ok=True)
            if attempt < RETRIES:
                time.sleep(5 * attempt)
    return job, f"FAILED: {last_err}"


def run(dataset: str, args: argparse.Namespace) -> int:
    jobs = load_jobs(dataset, args.tile_list)
    if args.limit:
        jobs = jobs[: args.limit]
    total_gb = sum(j.size for j in jobs) / 1e9
    todo = [j for j in jobs if not is_complete(j)]
    todo_gb = sum(j.size for j in todo) / 1e9
    print(f"[{dataset}] {len(jobs)} files ({total_gb:.1f} GB), "
          f"{len(jobs) - len(todo)} complete, {len(todo)} to download ({todo_gb:.1f} GB) "
          f"-> {DATASETS[dataset]['out_dir'].relative_to(REPO)}")

    if args.check:
        for j in todo:
            state = "size mismatch" if j.path.exists() else "missing"
            print(f"  {state}: {j.name}")
        return 0 if not todo else 1

    failures = []
    done_bytes = 0
    t0 = time.time()
    with requests.Session() as session, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download, j, session) for j in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            job, status = fut.result()
            if status == "ok":
                done_bytes += job.size
            else:
                failures.append((job, status))
            rate = done_bytes / 1e6 / max(time.time() - t0, 1e-6)
            print(f"  [{dataset} {i}/{len(todo)}] {status:8.60s} {job.name}  ({rate:.1f} MB/s)")

    if failures:
        log = DATASETS[dataset]["out_dir"] / "download_failures.log"
        with log.open("a", encoding="utf-8") as f:
            for job, status in failures:
                f.write(f"{datetime.now().isoformat(timespec='seconds')}\t{job.name}\t{job.url}\t{status}\n")
        print(f"[{dataset}] {len(failures)} failed, see {log.relative_to(REPO)}. Re-run to retry.")
        return 1
    print(f"[{dataset}] all {len(jobs)} files complete.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=["laz", "dop", "all"])
    ap.add_argument("--tile-list", type=Path, default=TILE_LIST)
    ap.add_argument("--workers", type=int, default=3, help="parallel downloads (default 3)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N tiles (testing)")
    ap.add_argument("--check", action="store_true", help="only verify existing files, no download")
    args = ap.parse_args()

    if not args.tile_list.exists():
        print(f"Tile list not found: {args.tile_list}. Run make_tile_list.py first.", file=sys.stderr)
        return 1
    datasets = ["laz", "dop"] if args.dataset == "all" else [args.dataset]
    return max(run(d, args) for d in datasets)


if __name__ == "__main__":
    sys.exit(main())
