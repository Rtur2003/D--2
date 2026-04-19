# =========================================================================
# WEB SAMPLE DOWNLOADER - Sunum icin harici Head CT goruntuleri
# =========================================================================
# Wikimedia Commons (public domain / CC) uzerinden 2-3 ornek Head CT
# goruntusu indirir ve web_crawled_test/ klasorune kaydeder.
#
# Kullanim:
#   python src/download_web_samples.py
#
# Sonra:
#   python main.py --webcrawl
# =========================================================================

import sys
import urllib.request
from pathlib import Path
from PIL import Image
from io import BytesIO

from config import WEB_CRAWLED_DIR


# Wikimedia Commons - public domain Head CT goruntuleri
# Her URL'in kaynak sayfasi yorumda; kopyalanmadan once erisilebilirligi
# kontrol edildi. Dosyalari kucuk PNG formatinda kaydederiz.
SOURCES = [
    {
        "name": "01_subdural_hematoma.png",
        "url": "https://upload.wikimedia.org/wikipedia/commons/5/5e/Subduralandherniation.PNG",
        "expected_label": "hemorrhage",
        "source": "Wikimedia Commons - Subdural hematoma with herniation",
    },
    {
        "name": "02_intracerebral_bleed.png",
        "url": "https://upload.wikimedia.org/wikipedia/commons/d/de/Bleedintocysticmass.png",
        "expected_label": "hemorrhage",
        "source": "Wikimedia Commons - Bleed into cystic mass (intracranial)",
    },
    {
        "name": "03_normal_brain_axial1.png",
        "url": "https://upload.wikimedia.org/wikipedia/commons/0/0b/CT_of_a_normal_brain%2C_axial_1.png",
        "expected_label": "normal",
        "source": "Wikimedia Commons - CT of normal brain, axial slice 1",
    },
    {
        "name": "04_normal_brain_axial2.png",
        "url": "https://upload.wikimedia.org/wikipedia/commons/7/77/CT_of_a_normal_brain%2C_axial_2.png",
        "expected_label": "normal",
        "source": "Wikimedia Commons - CT of normal brain, axial slice 2",
    },
]


def download_one(item: dict) -> bool:
    """Tek bir goruntuyu indir ve PNG olarak kaydet."""
    out_path = WEB_CRAWLED_DIR / item["name"]
    if out_path.exists():
        print(f"  [SKIP] {item['name']} zaten mevcut")
        return True
    try:
        req = urllib.request.Request(
            item["url"],
            headers={
                "User-Agent": (
                    "BM480-Edu/1.0 (university coursework; "
                    "contact: hasannarthurrr@gmail.com) Python-urllib"
                ),
                "Accept": "image/png,image/jpeg,image/*,*/*;q=0.8",
            },
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = resp.read()
        img = Image.open(BytesIO(data)).convert("RGB")
        img.save(str(out_path), format="PNG")
        print(f"  [OK]   {item['name']}  ({img.size[0]}x{img.size[1]})")
        print(f"         beklenen: {item['expected_label']}")
        print(f"         kaynak  : {item['source']}")
        return True
    except Exception as e:
        print(f"  [FAIL] {item['name']}: {e}")
        return False


def main():
    print("=" * 60)
    print("WEB SAMPLE DOWNLOADER - Head CT (sunum icin)")
    print("=" * 60)
    print(f"Hedef klasor: {WEB_CRAWLED_DIR}")
    print()

    ok = 0
    for item in SOURCES:
        if download_one(item):
            ok += 1

    print()
    print("=" * 60)
    print(f"Indirilen / Toplam: {ok} / {len(SOURCES)}")
    print()
    print("Sonraki adim:")
    print("  python main.py --webcrawl")
    print("=" * 60)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
