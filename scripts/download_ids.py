"""Download BabelStone IDS database."""

import argparse
import requests
from pathlib import Path
from tqdm import tqdm

# BabelStone IDS file URL
IDS_URL = "https://www.babelstone.co.uk/CJK/IDS.TXT"


def download_file(url: str, output_path: Path) -> None:
    """Download file with progress bar.

    Args:
        url: URL to download from
        output_path: Path to save file
    """
    print(f"Downloading from: {url}")
    print(f"Saving to: {output_path}")

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f, tqdm(
        total=total_size,
        unit="B",
        unit_scale=True,
        desc="Downloading"
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))

    print(f"Download complete: {output_path}")
    print(f"File size: {output_path.stat().st_size:,} bytes")


def main():
    parser = argparse.ArgumentParser(description="Download BabelStone IDS database")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/ids.txt"),
        help="Output file path (default: data/raw/ids.txt)"
    )
    parser.add_argument(
        "--url",
        type=str,
        default=IDS_URL,
        help=f"IDS file URL (default: {IDS_URL})"
    )
    args = parser.parse_args()

    try:
        download_file(args.url, args.output)
        print("\nSuccess! You can now run:")
        print(f"  python scripts/01_build_ids_lexicon.py --ids_file {args.output}")
    except Exception as e:
        print(f"Error downloading IDS file: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
