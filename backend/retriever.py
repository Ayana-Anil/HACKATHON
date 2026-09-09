"""
retriever.py - Role 5

Handles semantic search by taking a user query, generating an embedding via Embedder,
searching the VectorStore (FAISS), and formatting the retrieved chunks for the LLM.
"""

from typing import List, Dict, Any, Optional
from embedder import Embedder
from vector_store import VectorStore


class SyllabusRetriever:
    def __init__(self, embedder: Embedder, vector_store: VectorStore):
        """
        Initializes the retriever with instances of Embedder and VectorStore.
        
        Args:
            embedder (Embedder): Active embedder instance.
            vector_store (VectorStore): FAISS vector store instance.
        """
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieves top_k relevant text chunks for a given query string.
        
        Args:
            query (str): Question asked by the student.
            top_k (int): Number of most relevant context chunks to fetch.
            
        Returns:
            List[Dict[str, Any]]: List of retrieved documents/chunks with text and metadata.
        """
        if not query or not query.strip():
            return []

        # Step 1: Embed the query string into a vector using Role 3's embedder
        query_vector = self.embedder.get_embedding(query)

        # Step 2: Perform similarity search in FAISS using Role 4's vector store
        results = self.vector_store.search(query_vector, k=top_k)

        return results

    def format_context(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """
        Formats retrieved chunks into a clean, structured string block 
        ready to be injected into the Ollama prompt inside llm.py (Role 6).
        
        Args:
            retrieved_chunks (List[Dict[str, Any]]): Result list from self.retrieve()
            
        Returns:
            str: Combined context block or fallback message.
        """
        if not retrieved_chunks:
            return "No relevant syllabus information found."

        context_blocks = []
        for idx, chunk in enumerate(retrieved_chunks, start=1):
            text = chunk.get("text", chunk.get("content", "")).strip()
            page = chunk.get("metadata", {}).get("page", None)
            
            page_str = f" (Page {page})" if page is not None else ""
            block = f"[Chunk {idx}{page_str}]\n{text}"
            context_blocks.append(block)

        return "\n\n".join(context_blocks)

    def get_context_for_query(self, query: str, top_k: int = 3) -> str:
        """
        Convenience method that runs retrieve and format_context in a single call.
        
        Args:
            query (str): Question asked by user.
            top_k (int): Number of chunks to retrieve.
            
        Returns:
            str: Formatted context string for prompt building.
        """
        chunks = self.retrieve(query=query, top_k=top_k)
        return self.format_context(chunks)