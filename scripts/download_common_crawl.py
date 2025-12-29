"""Download Chinese text from Common Crawl.

This script downloads Chinese language segments from Common Crawl
using the WARC format and extracts clean text.
"""

import argparse
import gzip
import requests
from pathlib import Path
from tqdm import tqdm
from typing import List, Iterator
import re


# Common Crawl index URLs (latest crawls)
CC_INDEX_URL = "https://index.commoncrawl.org/CC-MAIN-{crawl_id}-index"
CC_S3_BASE = "https://data.commoncrawl.org/"

# Recent crawl IDs (update periodically)
RECENT_CRAWLS = [
    "2024-10",  # Example crawl ID
    "2024-05",
]


def get_available_crawls() -> List[str]:
    """Fetch list of available Common Crawl datasets.

    Returns:
        List of crawl IDs
    """
    print("Fetching available Common Crawl datasets...")
    response = requests.get("https://index.commoncrawl.org/collinfo.json")
    response.raise_for_status()

    crawls = response.json()
    crawl_ids = [crawl["id"].replace("CC-MAIN-", "") for crawl in crawls]

    print(f"Found {len(crawl_ids)} available crawls")
    return crawl_ids


def search_chinese_domains(
    crawl_id: str,
    domains: List[str] = None,
    max_results: int = 1000
) -> List[dict]:
    """Search Common Crawl index for Chinese domains.

    Args:
        crawl_id: Common Crawl ID
        domains: List of domains to search (if None, use default Chinese sites)
        max_results: Maximum results to return

    Returns:
        List of WARC record metadata
    """
    if domains is None:
        # Default Chinese domains
        domains = [
            "baike.baidu.com",
            "zhihu.com",
            "163.com",
            "sina.com.cn",
        ]

    print(f"Searching crawl {crawl_id} for Chinese content...")

    index_url = f"https://index.commoncrawl.org/CC-MAIN-{crawl_id}-index"
    results = []

    for domain in domains:
        query_url = f"{index_url}?url={domain}/*&output=json"
        print(f"  Querying: {domain}")

        try:
            response = requests.get(query_url)
            if response.status_code == 200:
                for line in response.text.strip().split("\n")[:max_results]:
                    if line:
                        import json
                        results.append(json.loads(line))
        except Exception as e:
            print(f"  Error querying {domain}: {e}")

    print(f"Found {len(results)} WARC records")
    return results


def download_warc_record(warc_url: str) -> bytes:
    """Download and decompress WARC record.

    Args:
        warc_url: URL to WARC file

    Returns:
        Decompressed WARC content
    """
    full_url = CC_S3_BASE + warc_url
    response = requests.get(full_url)
    response.raise_for_status()

    # Decompress if gzipped
    if warc_url.endswith(".gz"):
        return gzip.decompress(response.content)
    return response.content


def extract_text_from_warc(warc_content: bytes) -> str:
    """Extract plain text from WARC content.

    Args:
        warc_content: Raw WARC content

    Returns:
        Extracted text
    """
    try:
        # Simple extraction: look for text between HTML tags
        text = warc_content.decode("utf-8", errors="ignore")

        # Remove HTML tags
        text = re.sub(r"<[^>]+>", " ", text)

        # Remove URLs
        text = re.sub(r"http\S+", "", text)

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)

        return text.strip()
    except Exception as e:
        return ""


def download_chinese_corpus(
    output_file: Path,
    max_pages: int = 10000,
    crawl_id: str = None
) -> None:
    """Download Chinese text corpus from Common Crawl.

    Args:
        output_file: Path to save corpus
        max_pages: Maximum number of pages to download
        crawl_id: Specific crawl ID to use (if None, use latest)
    """
    print(f"Downloading Chinese corpus from Common Crawl")
    print(f"Max pages: {max_pages}")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Get crawl ID
    if crawl_id is None:
        available = get_available_crawls()
        crawl_id = available[0] if available else "2024-10"
        print(f"Using latest crawl: {crawl_id}")

    # Search for Chinese content
    warc_records = search_chinese_domains(crawl_id, max_results=max_pages)

    if not warc_records:
        print("No WARC records found. Try different domains or crawl ID.")
        return

    # Download and extract text
    sentence_count = 0
    with open(output_file, "w", encoding="utf-8") as f:
        for record in tqdm(warc_records[:max_pages], desc="Processing"):
            try:
                # Download WARC
                warc_url = record.get("filename")
                offset = record.get("offset", 0)
                length = record.get("length", 0)

                # For simplicity, we download full WARC (in production, use byte range)
                warc_content = download_warc_record(warc_url)

                # Extract text
                text = extract_text_from_warc(warc_content)

                # Split into sentences
                sentences = text.split("。")
                for sent in sentences:
                    sent = sent.strip()
                    if len(sent) > 10:  # Minimum length filter
                        f.write(sent + "\n")
                        sentence_count += 1

            except Exception as e:
                continue

    print(f"Downloaded {sentence_count:,} sentences to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Download Chinese text from Common Crawl"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/cc_zh.txt"),
        help="Output text file (default: data/raw/cc_zh.txt)"
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=10000,
        help="Maximum pages to download (default: 10000)"
    )
    parser.add_argument(
        "--crawl-id",
        type=str,
        help="Specific Common Crawl ID to use"
    )
    parser.add_argument(
        "--list-crawls",
        action="store_true",
        help="List available crawls and exit"
    )
    args = parser.parse_args()

    try:
        if args.list_crawls:
            crawls = get_available_crawls()
            print("\nAvailable Common Crawl datasets:")
            for crawl in crawls[:10]:
                print(f"  - {crawl}")
            return 0

        print("=" * 70)
        print("NOTICE: Common Crawl download is computationally intensive")
        print("and may take several hours for large corpora.")
        print("Consider using Wikipedia-zh as primary corpus.")
        print("=" * 70)

        download_chinese_corpus(
            args.output,
            max_pages=args.max_pages,
            crawl_id=args.crawl_id
        )

        print("\nSuccess! You can now run:")
        print(f"  python scripts/02_prepare_corpus.py --input {args.output}")

    except Exception as e:
        print(f"Error: {e}")
        print("\nNote: Common Crawl downloads are complex. If you encounter issues,")
        print("consider using Wikipedia-zh as the primary corpus instead.")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
