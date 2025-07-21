import argparse
import asyncio
import gzip
import os
import re
import sys
from io import BytesIO
from typing import Any, Dict, List

import openai
import requests
from dotenv import load_dotenv
from xml.etree import ElementTree

from supabase import Client, create_client
from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CrawlerRunConfig,
    CacheMode,
    MemoryAdaptiveDispatcher,
)

# Load environment variables
load_dotenv()

# XML namespace for sitemaps
NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}


def smart_chunk_markdown(markdown: str, max_len: int = 1000) -> List[str]:
    """
    Splits markdown hierarchically by headers (#, ##, ###), then by fixed size,
    ensuring each chunk is at most max_len characters.
    """
    def split_by_pattern(text: str, pattern: str) -> List[str]:
        positions = [m.start() for m in re.finditer(pattern, text, re.MULTILINE)] + [len(text)]
        return [text[positions[i]:positions[i+1]].strip()
                for i in range(len(positions) - 1)
                if text[positions[i]:positions[i+1]].strip()]

    chunks: List[str] = []
    for level1 in split_by_pattern(markdown, r'^# .+$'):
        if len(level1) <= max_len:
            chunks.append(level1)
            continue
        for level2 in split_by_pattern(level1, r'^## .+$'):
            if len(level2) <= max_len:
                chunks.append(level2)
                continue
            for level3 in split_by_pattern(level2, r'^### .+$'):
                if len(level3) <= max_len:
                    chunks.append(level3)
                else:
                    # final split by fixed size
                    for i in range(0, len(level3), max_len):
                        chunks.append(level3[i:i+max_len].strip())

    # Ensure no chunk exceeds max_len
    final_chunks: List[str] = []
    for chunk in chunks:
        if len(chunk) > max_len:
            for i in range(0, len(chunk), max_len):
                final_chunks.append(chunk[i:i+max_len].strip())
        else:
            final_chunks.append(chunk)

    return [c for c in final_chunks if c]


def _fetch_bytes(url: str) -> bytes:
    """
    Download a URL and return its raw bytes. Decompress .gz content if needed.
    """
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    data = response.content
    if url.endswith('.gz'):
        with gzip.GzipFile(fileobj=BytesIO(data)) as gz:
            return gz.read()
    return data


def parse_sitemap(sitemap_url: str) -> List[str]:
    """
    Parse a sitemap (XML or .gz) or sitemap-index and return all <loc> URLs.
    """
    raw = _fetch_bytes(sitemap_url)
    root = ElementTree.fromstring(raw)

    urls: List[str] = []
    # Recurse into sitemap-index entries
    for sitemap in root.findall('sm:sitemap', NS):
        loc = sitemap.find('sm:loc', NS)
        if loc is not None and loc.text:
            urls.extend(parse_sitemap(loc.text))

    # Collect URLs from urlset
    for url_node in root.findall('sm:url', NS):
        loc = url_node.find('sm:loc', NS)
        if loc is not None and loc.text:
            urls.append(loc.text)

    return urls


async def crawl_batch(urls: List[str], max_concurrent: int = 10) -> List[Dict[str, Any]]:
    """
    Batch crawl multiple URLs asynchronously using Crawl4AI.
    """
    browser_cfg = BrowserConfig(headless=True, verbose=False)
    run_cfg = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=70.0,
        check_interval=1.0,
        max_session_permit=max_concurrent
    )

    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        results = await crawler.arun_many(
            urls=urls,
            run_config=run_cfg,
            dispatcher=dispatcher
        )

    return [
        {'url': res.url, 'markdown': res.markdown}
        for res in results
        if res.success and res.markdown
    ]


def extract_section_info(chunk: str) -> Dict[str, Any]:
    """
    Extract header info and stats from a text chunk.
    """
    headers = re.findall(r'^(#+)\s+(.+)$', chunk, re.MULTILINE)
    header_str = '; '.join(f'{level} {title}' for level, title in headers)

    return {
        'headers': header_str,
        'char_count': len(chunk),
        'word_count': len(chunk.split()),
    }


async def get_embedding(text: str) -> List[float]:
    """
    Fetch embedding vector for given text using OpenAI.
    """
    try:
        response = await openai.embeddings.create(
            model='text-embedding-3-small',
            input=text
        )
        return response.data[0].embedding
    except Exception:
        # Return a zero vector on error
        return [0.0] * 1536


def load_env() -> Dict[str, str]:
    """
    Load required environment variables and validate presence.
    """
    openai_key = os.getenv('OPENAI_API_KEY')
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_SERVICE_KEY')
    if not all([openai_key, supabase_url, supabase_key]):
        sys.exit('Error: Please set OPENAI_API_KEY, SUPABASE_URL, and SUPABASE_SERVICE_KEY in .env')
    return {'openai': openai_key, 'url': supabase_url, 'key': supabase_key}


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Crawl a site map, chunk content, generate embeddings, and insert into Supabase.'
    )
    parser.add_argument('url', help='Site map URL to crawl (XML or .gz)')
    parser.add_argument('--collection', default='site_pages', help='Supabase table name')
    parser.add_argument('--chunk-size', type=int, default=1000, help='Max chunk size in chars')
    parser.add_argument('--max-concurrent', type=int, default=10, help='Max parallel sessions')
    parser.add_argument('--batch-size', type=int, default=100, help='Insert batch size')
    args = parser.parse_args()

    env = load_env()
    openai.api_key = env['openai']
    supabase: Client = create_client(env['url'], env['key'])

    urls = parse_sitemap(args.url)
    if not urls:
        sys.exit('No URLs found in sitemap.')

    crawl_results = asyncio.run(crawl_batch(urls, max_concurrent=args.max_concurrent))
    if not crawl_results:
        sys.exit('No crawl results to process.')

    # Prepare documents and metadata
    records = []
    chunk_idx = 0
    for entry in crawl_results:
        source = entry['url']
        for chunk in smart_chunk_markdown(entry['markdown'], max_len=args.chunk_size):
            meta = extract_section_info(chunk)
            records.append({
                'url': source,
                'chunk_number': chunk_idx,
                'content': chunk,
                'metadata': {**meta, 'source': source, 'chunk_index': chunk_idx},
                'embedding': None,  # placeholder
            })
            chunk_idx += 1

    if not records:
        sys.exit('No content to insert.')

    print(f'Generating embeddings for {len(records)} chunks...')
    batch_inputs = [r['content'] for r in records]
    emb_response = openai.embeddings.create(
        input=batch_inputs,
        model='text-embedding-3-small'
    )
    embeddings = [item.embedding for item in emb_response.data]

    for rec, emb in zip(records, embeddings):
        rec['embedding'] = emb

    print(f'Inserting {len(records)} records into "{args.collection}" in batches of {args.batch_size}...')
    for i in range(0, len(records), args.batch_size):
        batch = records[i:i + args.batch_size]
        try:
            resp = supabase.table(args.collection).insert(batch).execute()
            if getattr(resp, 'status_code', 0) >= 400:
                raise RuntimeError(f'HTTP {resp.status_code}: {resp}')
            print(f'✓ Inserted batch {i // args.batch_size}')
        except Exception as err:
            sys.exit(f'Batch {i // args.batch_size} failed: {err}')


if __name__ == '__main__':
    main()
