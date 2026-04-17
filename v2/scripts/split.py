"""
labels.csv -> train.csv / val.csv / test.csv (70/15/15)

Hasta bazinda ayrilir (ayni hastanin dilimleri ayni split'te).
Sinif dengesi mumkun oldugunca korunur (hastalarin sahip oldugu
etiketlerin cogunluguna gore stratifikasyon).
"""

from __future__ import annotations
import csv
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "labels.csv"

SEED = 42
RATIOS = (0.70, 0.15, 0.15)


def load_rows():
    with open(LABELS, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def patient_dominant_label(rows):
    per_patient = defaultdict(lambda: [0, 0])
    for r in rows:
        per_patient[r["patient_id"]][int(r["label"])] += 1
    dom = {}
    for pid, (n0, n1) in per_patient.items():
        dom[pid] = 1 if n1 >= n0 else 0
    return dom


def split_patients(dom, seed=SEED):
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for pid, lab in dom.items():
        by_class[lab].append(pid)

    train, val, test = [], [], []
    for lab, pids in by_class.items():
        rng.shuffle(pids)
        n = len(pids)
        n_train = int(round(n * RATIOS[0]))
        n_val = int(round(n * RATIOS[1]))
        train += pids[:n_train]
        val += pids[n_train:n_train + n_val]
        test += pids[n_train + n_val:]
    return set(train), set(val), set(test)


def write_split(rows, pids, out_path):
    subset = [r for r in rows if r["patient_id"] in pids]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "source", "patient_id"])
        w.writeheader()
        w.writerows(subset)
    n0 = sum(1 for r in subset if r["label"] == "0")
    n1 = sum(1 for r in subset if r["label"] == "1")
    return len(subset), n0, n1, len(pids)


def main():
    rows = load_rows()
    dom = patient_dominant_label(rows)
    tr, va, te = split_patients(dom)

    for name, pids in [("train", tr), ("val", va), ("test", te)]:
        out = ROOT / f"{name}.csv"
        n, n0, n1, np_ = write_split(rows, pids, out)
        print(f"{name:5s}: {n:5d} goruntu | normal={n0:5d}  hem={n1:5d}  | {np_:3d} hasta")


if __name__ == "__main__":
    main()
