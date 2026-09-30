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
from pathlib import Path
from urllib.request import Request, urlopen

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

models = {
    "craft_mlt_25k.pth": "https://github.com/Sanskar0-ok/Nepali-PDF-OCR/releases/download/v1.0/craft_mlt_25k.pth",
    "devanagari.pth": "https://github.com/Sanskar0-ok/Nepali-PDF-OCR/releases/download/v1.0/devanagari.pth",
}

for filename, url in models.items():

    destination = MODELS_DIR / filename

    if destination.exists():
        print(f"{filename} already exists. Skipping.")
        continue

    print(f"Downloading {filename}...")

    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0"
        }
    )

    with urlopen(request, timeout=300) as response:
        with open(destination, "wb") as output:
            while True:
                chunk = response.read(1024 * 1024)

                if not chunk:
                    break

                output.write(chunk)

    print(
        f"Finished {filename} "
        f"({destination.stat().st_size / 1024 / 1024:.1f} MB)"
    )