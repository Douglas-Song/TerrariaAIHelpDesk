from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
from crawl4ai.deep_crawling import BestFirstCrawlingStrategy
from crawl4ai.deep_crawling.filters import (
    FilterChain,
    DomainFilter,
    URLPatternFilter,
    ContentTypeFilter
)
from crawl4ai.deep_crawling.scorers import KeywordRelevanceScorer

from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse

from supabase import create_client, Client
from urllib.parse import urlparse, urldefrag

import os
import asyncio
import json
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

#Initialize .env variables
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

#Use deep crawl to scrape all website pages from Terraria Wiki
async def crawl_site(start_url: str, max_depth: int = 1):
    # Create a sophisticated filter chain
    filter_chain = FilterChain([
        # Domain boundaries
        DomainFilter(
            allowed_domains=["terraria.wiki.gg"]
        ),

        #Exclude all none English version of wiki
        URLPatternFilter(
            patterns=[
                "*terraria.wiki.gg/cs*",   # Čeština
                "*terraria.wiki.gg/de*",   # Deutsch
                "*terraria.wiki.gg/el*",   # Ελληνικά
                "*terraria.wiki.gg/es*",   # Español (formal & informal)
                "*terraria.wiki.gg/fi*",   # Suomi
                "*terraria.wiki.gg/fr*",   # Français
                "*terraria.wiki.gg/hi*",   # हिन्दी
                "*terraria.wiki.gg/hu*",   # Magyar
                "*terraria.wiki.gg/id*",   # Bahasa Indonesia
                "*terraria.wiki.gg/it*",   # Italiano
                "*terraria.wiki.gg/ja*",   # 日本語
                "*terraria.wiki.gg/ko*",   # 한국어
                "*terraria.wiki.gg/lt*",   # Lietuvių
                "*terraria.wiki.gg/lv*",   # Latviešu
                "*terraria.wiki.gg/nl*",   # Nederlands
                "*terraria.wiki.gg/no*",   # Norsk
                "*terraria.wiki.gg/pl*",   # Polski
                "*terraria.wiki.gg/pt*",   # Português
                "*terraria.wiki.gg/ru*",   # Русский
                "*terraria.wiki.gg/sv*",   # Svenska
                "*terraria.wiki.gg/th*",   # ไทย
                "*terraria.wiki.gg/tr*",   # Türkçe
                "*terraria.wiki.gg/uk*",   # Українська
                "*terraria.wiki.gg/vi*",   # Tiếng Việt
                "*terraria.wiki.gg/yue*",  # 粵語 (Cantonese)
                "*terraria.wiki.gg/zh*",   # 中文
                "*terraria.wiki.gg/wiki/Mobile_version*", # Mobile
                "*terraria.wiki.gg/wiki/cs*",
                "*terraria.wiki.gg/wiki/de*",
                "*terraria.wiki.gg/wiki/el*",
                "*terraria.wiki.gg/wiki/es*",
                "*terraria.wiki.gg/wiki/fi*",
                "*terraria.wiki.gg/wiki/fr*",
                "*terraria.wiki.gg/wiki/hi*",
                "*terraria.wiki.gg/wiki/hu*",
                "*terraria.wiki.gg/wiki/id*",
                "*terraria.wiki.gg/wiki/it*",
                "*terraria.wiki.gg/wiki/ja*",
                "*terraria.wiki.gg/wiki/ko*",
                "*terraria.wiki.gg/wiki/lt*",
                "*terraria.wiki.gg/wiki/lv*",
                "*terraria.wiki.gg/wiki/nl*",
                "*terraria.wiki.gg/wiki/no*",
                "*terraria.wiki.gg/wiki/pl*",
                "*terraria.wiki.gg/wiki/pt*",
                "*terraria.wiki.gg/wiki/ru*",
                "*terraria.wiki.gg/wiki/sv*",
                "*terraria.wiki.gg/wiki/th*",
                "*terraria.wiki.gg/wiki/tr*",
                "*terraria.wiki.gg/wiki/uk*",
                "*terraria.wiki.gg/wiki/vi*",
                "*terraria.wiki.gg/wiki/yue*",
                "*terraria.wiki.gg/wiki/zh*",
            ],
            reverse=True
        ),

        # Content type filtering
        ContentTypeFilter(allowed_types=["text/html"])
    ])

    # Create a relevance scorer
    keyword_scorer = KeywordRelevanceScorer(
        keywords=["wiki"],
        weight=0.7
    )

    # Set up the configuration
    config = CrawlerRunConfig(
        deep_crawl_strategy=BestFirstCrawlingStrategy(
            max_depth=max_depth,
            include_external=False,
            filter_chain=filter_chain,
            url_scorer=keyword_scorer,
            max_pages=100
        ),
        scraping_strategy=LXMLWebScrapingStrategy(),
        stream=True,
        verbose=True
    )

    # Execute the crawl
    results = []
    async with AsyncWebCrawler() as crawler:
        async for result in await crawler.arun(url = start_url, config=config, sesson_id = "session1"):
            score = result.metadata.get("score", 0)
            depth = result.metadata.get("depth", 0)
            print(f"Depth: {depth} | Score: {score:.2f} | {result.url}")

            if result.success:
                results.append(result)
                await process_and_store_document(result.url, result.markdown.raw_markdown)

    # Analyze the results
    print(f"Crawled {len(results)} high-value pages")
    print(f"Average score: {sum(r.metadata.get('score', 0) for r in results) / len(results):.2f}")

    # Group by depth
    depth_counts = {}
    for result in results:
        depth = result.metadata.get("depth", 0)
        depth_counts[depth] = depth_counts.get(depth, 0) + 1

    print("Pages crawled by depth:")
    for depth, count in sorted(depth_counts.items()):
        print(f"  Depth {depth}: {count} pages")

   
@dataclass
class ProcessedChunk:
    url: str
    chunk_number: int
    title: str
    summary: str
    content:str
    metadata: Dict[str, Any]
    embedding: List[float]

#Chunk the text obtained
def chunk_text(text: str, chunk_size: int = 5000) -> List[str]:
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        #Find the end position of each chunk
        end = start + chunk_size

        #If we reach the end of the text, take whatever is left
        if end >= text_length:
            chunks.append(text[start:].strip())
            break
        
        #Try to find code block
        chunk = text[start:end]
        code_block = chunk.rfind('```')
        if code_block != -1 and code_block > chunk_size * 0.3:
            end = start + code_block

        #If no code block, try break at paragraph
        elif '\n\n' in chunk:
            #Find the last paragraph break
            last_break = chunk.rfind('\n\n')
            if last_break > chunk_size * 0.3:
                end = start + last_break
        
        elif '. ' in chunk:
            #Find the last sentence break
            last_period = chunk.rfind('. ')
            if last_period > chunk_size * 0.3:
                end = start + last_period
        
        #Extract chunk and clean up
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        #Move to start position of next chunk
        start = max(start + 1, end)

        return chunks

#Use AI to help to conclude title and summary of each documentation chunk.
async def get_title_and_summary(chunk : str, url: str):
    """Extract title and summary using GPT-4."""
    system_prompt = """You are an AI that extracts titles and summaries from documentation chunks.
    Return a JSON object with 'title' and 'summary' keys.
    For the title: If this seems like the start of a document, extract its title. If it's a middle chunk, derive a descriptive title.
    For the summary: Create a concise summary of the main points in this chunk.
    Keep both title and summary concise but informative."""
    
    try:
        response = await openai_client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"URL: {url}\n\nContent:\n{chunk[:1000]}..."}  # Send first 1000 chars for context
            ],
            response_format={ "type": "json_object" }
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error getting title and summary: {e}")
        return {"title": "Error processing title", "summary": "Error processing summary"}

#Get the embedding model from OPENAI
async def get_embedding(text: str) -> List[float]:
    try:
        result = await openai_client.embeddings.create(
            model = "text-embedding-3-small",
            input = text
        )
        return result.data[0].embedding
    except Exception as error:
        print(f"Failed to fetch embedding: {error}")
        return [0.0] * 1536 #return a zero vector of size 1536
    
#Process and embbed the chunk
async def process_chunk(chunk: str, chunk_number: int, url:str) -> ProcessedChunk:
    extracted = await get_title_and_summary(chunk, url)

    embedding = await get_embedding(chunk)

    metadata = {
        "source" : "Terraria Wiki",
        "chunk_size" : len(chunk),
        "crawled_at": datetime.now(timezone.utc).isoformat(),
        "url_path" : urlparse(url).path
    }

    return ProcessedChunk(
        url = url,
        chunk_number = chunk_number,
        title = extracted['title'],
        summary= extracted['summary'],
        content=chunk,
        metadata=metadata,
        embedding=embedding
    )

#Insert a chunk into Supabase
async def insert_chunk(chunk: ProcessedChunk):
    try:
        data = {
            "url": chunk.url,
            "chunk_number": chunk.chunk_number,
            "title": chunk.title,
            "summary": chunk.summary,
            "content": chunk.content,
            "metadata": chunk.metadata,
            "embedding": chunk.embedding
        }
        
        #Insert all the chunk into our SQL supabase.
        result = supabase.table("site_pages").insert(data).execute()
        print(f"Inserted chunk {chunk.chunk_number} for link {chunk.url}")
        return result
    except Exception as e:
        print(f"Encountered error when insterting chunk: {e}")
        return None

async def process_and_store_document(url:str, markdown:str):
    #split into chunks
    chunks = chunk_text(markdown)

    #process chunks in parrallel
    tasks = [
        process_chunk(chunk, i, url) 
        for i, chunk in enumerate(chunks)
    ]
    processed_chunks = await asyncio.gather(*tasks)

    insert_tasks = [
        insert_chunk(chunk)
        for chunk in processed_chunks
    ]
    await asyncio.gather(*insert_tasks) 

async def main():
    await crawl_site("https://terraria.wiki.gg/")

if __name__ == "__main__":
    asyncio.run(main())
