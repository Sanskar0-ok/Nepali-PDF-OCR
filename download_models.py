from pathlib import Path
from urllib.request import urlretrieve

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

models = {
    "craft_mlt_25k.pth": "https://github.com/Sanskar0-ok/Nepali-PDF-OCR/releases/download/v1.0/craft_mlt_25k.pth",
    "devanagari.pth": "https://github.com/Sanskar0-ok/Nepali-PDF-OCR/releases/download/v1.0/devanagari.pth",
}

for filename, url in models.items():
    destination = MODELS_DIR / filename

    print(f"Downloading {filename}...")
    urlretrieve(url, destination)
    print(f"Finished: {filename}")