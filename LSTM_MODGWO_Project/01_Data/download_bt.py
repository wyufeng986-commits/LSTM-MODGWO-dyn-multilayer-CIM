#!/usr/bin/env python3
"""Robust resume download for CNS bluetooth CSV zip from Netzschleuder."""

from __future__ import annotations

import time
import urllib.request
from pathlib import Path

URL = "https://networks.skewed.de/net/copenhagen/files/bt.csv.zip"
OUT = Path(__file__).resolve().parent / "raw" / "cns" / "bt.csv.zip"
EXPECTED_MIN = 8 * 1024 * 1024  # ~9.09 MiB


def download() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    attempt = 0
    while True:
        attempt += 1
        existing = OUT.stat().st_size if OUT.exists() else 0
        headers = {"User-Agent": "Mozilla/5.0", "Accept": "*/*"}
        if existing:
            headers["Range"] = f"bytes={existing}-"
        print(f"attempt={attempt} existing={existing}")
        req = urllib.request.Request(URL, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                mode = "ab" if existing and resp.status == 206 else "wb"
                if mode == "wb" and existing:
                    print("server ignored Range; rewriting")
                with open(OUT, mode) as f:
                    while True:
                        chunk = resp.read(256 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                        if f.tell() % (1024 * 1024) < 256 * 1024:
                            print(f"size={OUT.stat().st_size / 1024 / 1024:.2f}MB")
        except Exception as exc:
            print(f"error: {exc}")
            time.sleep(min(30, attempt * 2))
            continue

        size = OUT.stat().st_size
        print(f"finished pass size={size}")
        if size >= EXPECTED_MIN:
            print("OK")
            return
        time.sleep(2)


if __name__ == "__main__":
    download()
