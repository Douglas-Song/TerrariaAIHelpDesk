# TerrariaAIHelpDesk
This project builds an AI-powered helpdesk for the popular 2D sandbox game **Terraria** using **Retrieval - Augmented - Generation (RAG)**
to deliver context - aware, accurate, and game - specific assistance to players. It enables user to ask questions and receive helpful answers grounded in
content from the Terraria Wiki.

## Features
- **RAG-based Question Answering:** Uses embeddings and a vector database to retrieve the most relevant chunks of data ffrom the Terraria Wiki.
- **Context-Aware Responses:** Queries are answered in OpenAI's language models, ensuring accurate and wiki-grounded information.
- **Web Crawling with Crawl4AI:** Sitemap-based batch crawling from the **terraria.wiki.gg** to acquire and preprocess relevant pages.
- **Smart chunking and Embedding:** Large web content is segmented into embedding-friendly pieces using a chunking algorithm and embeded via OpenAI.
- **Supabase Integration:** Chunked data and metadata are stored and queried from Supabase using SQL and vector search.
- **Streamlit Interface** Clean, minimal frontend for querying the AI and viewing responses with clickable sources

## Setup Instruction

## 1. Clone repository

```bash
git clone https://github.com/Douglas-Song/TerrariaAIHelpDesk.git
cd TerrariaAIHelpDesk
```

## 2. Install UV if you haven't
```bash
pip install uv
```

## 3. Create and activate a virtual environment:
```bash
uv venv
.venv/Scripts/active
# on Mac or Linux: source .venv/bin/activate
```

## 4. Install dependencies
```bash
uv pip install -r requirements.txt
```

## Option 1: Setup your own Database and AI model.
Create your own .env file under the **TerrariaAIHelpdesk** folder. Then create and add your own API keys from openai.com and Supabase.com
In your .env file, you should have:
OPENAI_API_KEY= your-api-key
SUPABASE_URL= your-project-url
SUPABASE_SERVICE_KEY= your-service-role-api-key

To get Supabase project URL, go to project - configuration - Data API - Project URL
To get Supabase service role API key, go to - project - project settings - API keys - service_role (secret)

After all API keys are set up, run: python crawl_and_insert.py https://terraria.wiki.gg/sitemaps/sitemap-terraria_en-NS_0-0.xml.gz
to crawl your own Terraria Wiki data.

## Option 2: Use already exsited Supabase with Terraria Wiki data.
Create your own .env file under the **TerrariaAIHelpdesk** folder. Then create and add your own API keys from openai.com and Supabase.com
In your .env file, you should have:
OPENAI_API_KEY= your-api-key
SUPABASE_URL=https://iykddopovgfrnzepkmxm.supabase.co

After all API keys are set up, move on to next step.

## Streamlit Interface
Once everything is set up, run: streamlit run streamlit_ui.py
to open the streamlit interface, and start asking questions about Terraria!


## How it works
The system utilizes Crawl4AI, a widely adopted and LLM-compatible web crawler, to retrieve structured data from the Terraria Wiki. It employs the batch crawl method, which supports crawling multiple URLs in parallel. This approach offers substantial time savings compared to deep crawling or single-pages crawling, provided that the target URLs are known in advance. These URLs are typically obtained from a sitemap.xml or sitemap.xml.gz file – standard resources maintained by websites that index all accessible pages. 

After the web is collected, it is processed through a smart chunk function that segments long website text into smaller, embedding-friendly units. These chunks are then passed to an embedding model provided by OpenAI, accessed via an API key. The resulting embeddings are then stored along with their associated metadata, including the page title, source URL, and textual content. 

All processed reports are then inserted into a Supabase database, which offers a structured and queryable SQL-based interface for storage and retrieval. A Retrieval-Augmented-Generation (RAG) AI agent – powered by GPT-4.1-mini – queries the database to retrieve relevant information in response to user questions, enabling accurate and context-aware assistance for Terraria players. Finally, a Streamlit UI is set up to provide a user-friendly and easy to manipulate interface.

## Future Improvements
- Agentic RAG & MCP server integration for multi-step queries
- Advanced filtering on sitemap URLs.
- Improved Streamlit UI design for game immersion.