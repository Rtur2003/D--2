"""
v2 icin tek bir labels.csv uretir.

Iki kaynak:
  1) abdulkader90/brain-ct-hemorrhage-dataset  -> Data/NORMAL, Data/Hemorrhagic
  2) vbookshelf/computed-tomography-ct-images  -> Patients_CT + hemorrhage_diagnosis.csv

Kopyalama yok; absolute path + label (0=normal, 1=hemorrhage) + source + patient_id yazilir.
Hasta bazinda split yapabilmek icin patient_id onemli.
"""

from __future__ import annotations
import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data_raw"
OUT = ROOT / "labels.csv"

ABD_ROOT = RAW / "Data"
VBOOK_ROOT = RAW / "computed-tomography-images-for-intracranial-hemorrhage-detection-and-segmentation-1.0.0"


def collect_abdulkader() -> list[dict]:
    rows = []
    normal_dir = ABD_ROOT / "NORMAL"
    for patient_dir in sorted(normal_dir.iterdir()):
        if not patient_dir.is_dir():
            continue
        pid = f"abd_{patient_dir.name.split('[')[0]}"
        for img in sorted(patient_dir.rglob("*.jpg")):
            rows.append({"path": str(img.resolve()), "label": 0, "source": "abdulkader", "patient_id": pid})

    hem_root = ABD_ROOT / "Hemorrhagic" / "KANAMA"
    for patient_dir in sorted(hem_root.iterdir()):
        if not patient_dir.is_dir():
            continue
        pid = f"abd_H_{patient_dir.name}"
        for img in sorted(patient_dir.rglob("*.jpg")):
            rows.append({"path": str(img.resolve()), "label": 1, "source": "abdulkader", "patient_id": pid})
    return rows


def collect_vbookshelf() -> list[dict]:
    csv_path = VBOOK_ROOT / "hemorrhage_diagnosis.csv"
    patients_root = VBOOK_ROOT / "Patients_CT"

    rows = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pnum = int(row["PatientNumber"])
            snum = int(row["SliceNumber"])
            no_hem = int(row["No_Hemorrhage"])
            label = 0 if no_hem == 1 else 1

            patient_dir = patients_root / f"{pnum:03d}" / "brain"
            img = patient_dir / f"{snum}.jpg"
            if not img.exists():
                continue
            rows.append({
                "path": str(img.resolve()),
                "label": label,
                "source": "vbookshelf",
                "patient_id": f"vbook_{pnum:03d}",
            })
    return rows


def main():
    rows = collect_abdulkader() + collect_vbookshelf()

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "source", "patient_id"])
        w.writeheader()
        w.writerows(rows)

    n0 = sum(1 for r in rows if r["label"] == 0)
    n1 = sum(1 for r in rows if r["label"] == 1)
    n_src = {}
    for r in rows:
        n_src[r["source"]] = n_src.get(r["source"], 0) + 1
    n_pat = len({r["patient_id"] for r in rows})

    print(f"Toplam: {len(rows)} goruntu, {n_pat} hasta")
    print(f"  Normal (0)     : {n0}")
    print(f"  Hemorrhage (1) : {n1}")
    for k, v in n_src.items():
        print(f"  source={k:12s} -> {v}")
    print(f"\nYazildi: {OUT}")


if __name__ == "__main__":
    main()
