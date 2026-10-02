import os
import time
from concurrent.futures import ThreadPoolExecutor

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

BASE = "https://physionet.org/files/challenge-2019/1.0.0/training/"
SAVE = "PhysioNet_Challenge2019"
MAX_WORKERS = 16
REQUEST_TIMEOUT = (10, 60)
PER_FILE_ATTEMPTS = 3

session = requests.Session()


def setup_session():
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=MAX_WORKERS, pool_maxsize=MAX_WORKERS)
    session.mount("https://", adapter)
    session.mount("http://", adapter)


setup_session()


def get_files(folder):
    url = BASE + folder + "/"
    print(f"Reading {folder}...")

    r = session.get(url, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")

    files = []

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.endswith(".psv"):
            files.append((url + href, folder, href))

    print(f"Found {len(files)} files in {folder}")
    return files


def download(item):
    url, folder, filename = item

    outdir = os.path.join(SAVE, folder)
    os.makedirs(outdir, exist_ok=True)

    outfile = os.path.join(outdir, filename)
    partfile = outfile + ".part"

    if os.path.exists(outfile):
        return "skipped", folder, filename, ""

    last_error = ""
    for attempt in range(1, PER_FILE_ATTEMPTS + 1):
        try:
            r = session.get(url, stream=True, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()

            with open(partfile, "wb") as f:
                for chunk in r.iter_content(65536):
                    if chunk:
                        f.write(chunk)

            os.replace(partfile, outfile)
            return "downloaded", folder, filename, ""
        except requests.exceptions.RequestException as exc:
            last_error = str(exc)
            if os.path.exists(partfile):
                try:
                    os.remove(partfile)
                except OSError:
                    pass

            if attempt < PER_FILE_ATTEMPTS:
                time.sleep(min(2 ** attempt, 8))

    return "failed", folder, filename, last_error


files = []

files += get_files("training_setA")
files += get_files("training_setB")

print(f"\nTotal files: {len(files)}")
print("Downloading...\n")

downloaded = 0
skipped = 0
failed = []

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
    for status, folder, filename, error in tqdm(pool.map(download, files), total=len(files)):
        if status == "downloaded":
            downloaded += 1
        elif status == "skipped":
            skipped += 1
        else:
            failed.append((folder, filename, error))

print("\nSummary:")
print(f"Downloaded: {downloaded}")
print(f"Skipped (already exists): {skipped}")
print(f"Failed: {len(failed)}")

if failed:
    failed_log = os.path.join(SAVE, "failed_downloads.txt")
    with open(failed_log, "w", encoding="utf-8") as f:
        for folder, filename, error in failed:
            f.write(f"{folder}/{filename}\t{error}\n")
    print(f"Failed file list saved to: {failed_log}")

print("\nDone!")