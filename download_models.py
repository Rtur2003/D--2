"""
Egitilmis model agirliklarini GitHub Releases'ten indirir.

Modeller (.pth) GitHub'in 100MB dosya limiti sebebiyle repository'ye dahil
edilmemistir; release asset olarak saklanir.

Kullanim:
    python download_models.py

Kendi GitHub hesabiniza release yuklediginizde RELEASE_BASE_URL'yi guncelleyin.
"""

import hashlib
import sys
import urllib.request
from pathlib import Path


RELEASE_BASE_URL = "https://github.com/Rtur2003/D--2/releases/download/v1.0"

MODEL_FILES = {
    "convnext_tiny_best.pth": {
        "url": f"{RELEASE_BASE_URL}/convnext_tiny_best.pth",
        "expected_size_mb": 106,
    },
    "custom_cnn_best.pth": {
        "url": f"{RELEASE_BASE_URL}/custom_cnn_best.pth",
        "expected_size_mb": 5,
    },
    "train_stats.json": {
        "url": f"{RELEASE_BASE_URL}/train_stats.json",
        "expected_size_mb": 0.01,
    },
}

MODELS_DIR = Path(__file__).parent / "models"


def _progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100.0, downloaded * 100.0 / total_size)
        mb_done = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        sys.stdout.write(f"\r    {percent:5.1f}%  ({mb_done:6.1f} / {mb_total:6.1f} MB)")
        sys.stdout.flush()


def download_one(fname: str, meta: dict) -> bool:
    dest = MODELS_DIR / fname
    if dest.exists():
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"[SKIP] {fname} mevcut ({size_mb:.1f} MB)")
        return True

    print(f"[GET ] {fname}")
    print(f"        {meta['url']}")
    try:
        urllib.request.urlretrieve(meta["url"], str(dest), reporthook=_progress)
        print()
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"[DONE] {fname} ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"\n[FAIL] {fname}: {e}")
        if dest.exists():
            dest.unlink()
        return False


def main() -> int:
    MODELS_DIR.mkdir(exist_ok=True)
    print(f"Hedef klasor: {MODELS_DIR}")
    print(f"Release: {RELEASE_BASE_URL}\n")

    ok = 0
    for fname, meta in MODEL_FILES.items():
        if download_one(fname, meta):
            ok += 1

    total = len(MODEL_FILES)
    print(f"\n{ok}/{total} dosya hazir.")
    if ok < total:
        print("Eksik dosyalar icin release URL'sini ve hesap adini kontrol edin.")
        return 1
    print("Tum modeller indirildi. Arayuzu baslatabilirsiniz: python main.py --app")
    return 0


if __name__ == "__main__":
    sys.exit(main())
