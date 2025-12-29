"""Download and extract Chinese Wikipedia dump.

This script downloads the latest Chinese Wikipedia dump and extracts
plain text using WikiExtractor.
"""

import argparse
import subprocess
import requests
import re
from pathlib import Path
from tqdm import tqdm
from datetime import datetime


# Wikipedia dump base URL
WIKI_DUMP_URL = "https://dumps.wikimedia.org/zhwiki/"
LATEST_DUMP_URL = WIKI_DUMP_URL + "latest/zhwiki-latest-pages-articles.xml.bz2"


def get_latest_dump_date() -> str:
    """Get the date of the latest Wikipedia dump.

    Returns:
        Date string in YYYYMMDD format
    """
    print("Fetching latest dump date...")
    response = requests.get(WIKI_DUMP_URL)
    response.raise_for_status()

    # Parse HTML to find latest date directory
    dates = re.findall(r'href="(\d{8})/"', response.text)
    if not dates:
        raise ValueError("Could not find latest dump date")

    latest = max(dates)
    print(f"Latest dump date: {latest}")
    return latest


def download_file(url: str, output_path: Path) -> None:
    """Download file with progress bar and resume support.

    Args:
        url: URL to download from
        output_path: Path to save file
    """
    print(f"Downloading from: {url}")
    print(f"Saving to: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if file exists (for resume)
    if output_path.exists():
        existing_size = output_path.stat().st_size
        print(f"Found existing file ({existing_size:,} bytes). Attempting to resume...")
        headers = {"Range": f"bytes={existing_size}-"}
        mode = "ab"
    else:
        existing_size = 0
        headers = {}
        mode = "wb"

    response = requests.get(url, stream=True, headers=headers)

    # Check if resume is supported
    if response.status_code == 416:
        print("File already fully downloaded.")
        return
    elif response.status_code not in (200, 206):
        response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0)) + existing_size

    with open(output_path, mode) as f, tqdm(
        initial=existing_size,
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


def install_wikiextractor() -> None:
    """Install WikiExtractor if not already installed."""
    try:
        import wikiextractor
        print("WikiExtractor already installed.")
    except ImportError:
        print("WikiExtractor not found. Installing...")
        subprocess.run(
            ["pip", "install", "wikiextractor"],
            check=True
        )
        print("WikiExtractor installed successfully.")


def extract_text(dump_file: Path, output_dir: Path) -> Path:
    """Extract plain text from Wikipedia dump.

    Args:
        dump_file: Path to .xml.bz2 dump file
        output_dir: Directory to save extracted text

    Returns:
        Path to extracted text directory
    """
    print(f"Extracting text from: {dump_file}")
    print(f"Output directory: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Run WikiExtractor
    cmd = [
        "python", "-m", "wikiextractor.WikiExtractor",
        str(dump_file),
        "--json",  # JSON format for easier parsing
        "--no-templates",  # Skip templates
        "--output", str(output_dir),
        "--bytes", "100M",  # Split into 100MB files
        "--processes", "8",  # Parallel processing
    ]

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    print(f"Extraction complete: {output_dir}")
    return output_dir


def merge_extracted_files(extracted_dir: Path, output_file: Path) -> None:
    """Merge all extracted wiki_* files into single text file.

    Args:
        extracted_dir: Directory containing extracted files
        output_file: Output merged text file
    """
    print(f"Merging extracted files...")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    import json

    sentence_count = 0

    with open(output_file, "w", encoding="utf-8") as f_out:
        # Find all wiki_* JSON files
        for wiki_file in sorted(extracted_dir.rglob("wiki_*")):
            print(f"Processing: {wiki_file}")
            with open(wiki_file, "r", encoding="utf-8") as f_in:
                for line in f_in:
                    try:
                        article = json.loads(line)
                        text = article.get("text", "")

                        # Split into sentences (simple split on periods/newlines)
                        sentences = text.replace("\n", "。").split("。")

                        for sent in sentences:
                            sent = sent.strip()
                            if sent:
                                f_out.write(sent + "\n")
                                sentence_count += 1

                    except json.JSONDecodeError:
                        continue

    print(f"Merged {sentence_count:,} sentences to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Download and extract Chinese Wikipedia dump"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/wiki_zh.txt"),
        help="Output text file (default: data/raw/wiki_zh.txt)"
    )
    parser.add_argument(
        "--keep-dump",
        action="store_true",
        help="Keep intermediate .xml.bz2 dump file"
    )
    parser.add_argument(
        "--dump-dir",
        type=Path,
        default=Path("data/raw/wikipedia_dump"),
        help="Directory for dump files (default: data/raw/wikipedia_dump)"
    )
    parser.add_argument(
        "--use-existing",
        type=Path,
        help="Use existing dump file instead of downloading"
    )
    args = parser.parse_args()

    try:
        # Install WikiExtractor if needed
        install_wikiextractor()

        # Download dump
        if args.use_existing:
            dump_file = args.use_existing
            print(f"Using existing dump: {dump_file}")
        else:
            dump_file = args.dump_dir / "zhwiki-latest-pages-articles.xml.bz2"
            download_file(LATEST_DUMP_URL, dump_file)

        # Extract text
        extracted_dir = args.dump_dir / "extracted"
        extract_text(dump_file, extracted_dir)

        # Merge into single file
        merge_extracted_files(extracted_dir, args.output)

        # Cleanup
        if not args.keep_dump:
            print(f"Removing dump file: {dump_file}")
            dump_file.unlink(missing_ok=True)

        print("\nSuccess! You can now run:")
        print(f"  python scripts/02_prepare_corpus.py --input {args.output}")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
