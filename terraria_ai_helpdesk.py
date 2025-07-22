from __future__ import annotations as _annotations

from dataclasses import dataclass
from dotenv import load_dotenv
import logfire
import asyncio
import httpx
import os

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIModel
from openai import AsyncOpenAI
from supabase import Client
from typing import List

llm = os.getenv('LLM_MODEL', 'gpt-4o-mini')
model = OpenAIModel(llm)

@dataclass
class TerrariaAiDeps:
    supabase: Client
    openai_client: AsyncOpenAI

system_prompt = """
You are Terraria Helpdesk, an AI assistant specialized in all things Terraria. You have read‑only access to a Supabase database pre‑loaded with chunked and embedded content from the official Terraria Wiki. Whenever a user asks a question about Terraria—whether it’s about items, NPCs, bosses, mechanics, biomes, crafting recipes, progression, or modding—you must:

Retrieve the most relevant chunks from Supabase by performing a vector‑similarity search over the embeddings.

Synthesize a clear, accurate answer based primarily on those retrieved chunks.

Cite each chunk you used (by chunk ID or title) so the user knows which Wiki passages informed your response.

If the database returns no relevant information, fall back on your internal knowledge of Terraria.

Structure answers with headings, bullet points, and step‑by‑step instructions as needed.

Ask clarifying questions if the user’s query is ambiguous or could refer to multiple game versions.

Always strive for concise, friendly, and precise guidance to help players resolve their Terraria questions.
"""

Terraria_ai_helpdesk = Agent(
    model,
    system_prompt=system_prompt,
    deps_type=TerrariaAiDeps,
    retries=2
)

async def get_embedding(text: str, openai_client: AsyncOpenAI) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1536  # Return zero vector on error

@Terraria_ai_helpdesk.tool
async def retrieve_relevant_documentation(ctx: RunContext[TerrariaAiDeps], user_query: str) -> str:
    """
    Retrieve relevant documentation chunks based on the query with RAG.
    
    Args:
        ctx: The context including the Supabase client and OpenAI client
        user_query: The user's question or query
        
    Returns:
        A formatted string containing the top 5 most relevant documentation chunks
    """
    try:
        # Get the embedding for the query
        query_embedding = await get_embedding(user_query, ctx.deps.openai_client)
        
        # Query Supabase for relevant documents
        result = ctx.deps.supabase.rpc(
            'match_site_pages',
            {
                'query_embedding': query_embedding,
                'match_count': 5
            }
        ).execute()
        
        if not result.data:
            return "No relevant documentation found."
        
                    
        # Format the results
        formatted_chunks = []
        for doc in result.data:
            chunk_text = f"""
# {doc['title']}

{doc['content']}
"""
            formatted_chunks.append(chunk_text)
            
        # Join all chunks with a separator
        return "\n\n---\n\n".join(formatted_chunks)
        
    except Exception as e:
        print(f"Error retrieving documentation: {e}")
        return f"Error retrieving documentation: {str(e)}"