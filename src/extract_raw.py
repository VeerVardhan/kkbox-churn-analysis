import glob
import os
import zipfile
import py7zr

DOWNLOADS = os.path.join(os.path.expanduser("~"), "Downloads")
RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

TARGET_FILES = {
    "train_v2.csv.7z",
    "transactions_v2.csv.7z",
    "user_logs_v2.csv.7z",
    "members_v3.csv.7z",
}

zips = glob.glob(os.path.join(DOWNLOADS, "*kkbox*.zip"))
if not zips:
    raise FileNotFoundError(f"No kkbox zip file found in {DOWNLOADS}")
zip_path = zips[0]
print(f"Found zip: {zip_path}")

with zipfile.ZipFile(zip_path, "r") as zf:
    for member in zf.namelist():
        base = os.path.basename(member)
        if base in TARGET_FILES:
            out_7z = os.path.join(RAW_DIR, base)
            print(f"Extracting {base} from zip...")
            with zf.open(member) as src, open(out_7z, "wb") as dst:
                while chunk := src.read(1024 * 1024):
                    dst.write(chunk)

            print(f"Decompressing {base} -> CSV...")
            with py7zr.SevenZipFile(out_7z, "r") as archive:
                archive.extractall(path=RAW_DIR)

            os.remove(out_7z)  # remove .7z after CSV is extracted

print("\nDone! Files in data/raw:")
for f in sorted(os.listdir(RAW_DIR)):
    size_mb = os.path.getsize(os.path.join(RAW_DIR, f)) / (1024 * 1024)
    print(f"  {f:<25} {size_mb:,.1f} MB")