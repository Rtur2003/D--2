"""
abdulkader Hemorrhagic hastalarinda edge slice noise'u temizle.

Kural:
  - Normal hastalarin tum dilimleri tutulur (beyin her kesitte normaldir).
  - Vbookshelf dilimleri zaten slice-level etiketli, dokunma.
  - abdulkader Hemorrhagic: her hastanin dilimlerini numaraya gore sirala,
    ilk %15 ve son %15'i at (toplam %30 kirp).

labels.csv -> labels.csv (orijinal yedeklenir) + labels.raw.csv
train/val/test.csv yeniden split.py ile uretilmelidir.
"""

from __future__ import annotations
import csv
import re
import shutil
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "labels.csv"
RAW_BACKUP = ROOT / "labels.raw.csv"

TRIM_FRACTION = 0.15  # ilk %15 + son %15 kirpilir


def slice_num(path: str) -> int:
    m = re.search(r"_(\d+)\.jpg$", path)
    return int(m.group(1)) if m else -1


def main():
    if not RAW_BACKUP.exists():
        shutil.copy(LABELS, RAW_BACKUP)
        print(f"[BACKUP] {RAW_BACKUP.name} olusturuldu")

    with open(RAW_BACKUP, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    # Sadece abdulkader Hemorrhagic -> hasta bazli grupla
    groups = defaultdict(list)
    keep_others = []
    for r in rows:
        if r["source"] == "abdulkader" and r["label"] == "1":
            groups[r["patient_id"]].append(r)
        else:
            keep_others.append(r)

    # Her hasta icin orta %70
    kept = []
    dropped = 0
    for pid, items in groups.items():
        items.sort(key=lambda r: slice_num(r["path"]))
        n = len(items)
        lo = int(round(n * TRIM_FRACTION))
        hi = n - lo
        middle = items[lo:hi]
        kept.extend(middle)
        dropped += (n - len(middle))

    final = keep_others + kept

    with open(LABELS, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "source", "patient_id"])
        w.writeheader()
        w.writerows(final)

    n0 = sum(1 for r in final if r["label"] == "0")
    n1 = sum(1 for r in final if r["label"] == "1")
    print(f"abdulkader Hemorrhagic: {dropped} edge-slice atildi")
    print(f"Toplam: {len(final)} (normal={n0}, hem={n1})")
    print("NEXT: python scripts/split.py")


if __name__ == "__main__":
    main()
