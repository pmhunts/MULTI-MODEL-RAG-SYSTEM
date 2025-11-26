from typing import List, Dict, Any
import time
import logging
import os

class QAEngine:
    """
    Generates answers using retrieved context from a vector store.
    """

    def __init__(self, vector_store, api_key=None):
        """
        Initialize QAEngine with a vector store.

        Args:
            vector_store: Vector store for retrieving relevant contexts.
            api_key (str, optional): Anthropic API key. If not provided, falls back to environment variable.
        """
        self.vector_store = vector_store
        self.logger = logging.getLogger(__name__)

        # Check if Anthropic API key is available (from parameter or environment)
        self.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
        self.use_llm = self.api_key is not None

    def generate_answer(self, query: str, use_hybrid: bool = True, top_k: int = 5) -> Dict[str, Any]:
        """
        Generate an answer with citations using the retrieved context.

        Args:
            query (str): The user query string.
            use_hybrid (bool, optional): Whether to use hybrid search. Defaults to True.
            top_k (int, optional): Number of top documents to retrieve. Defaults to 5.

        Returns:
            Dict[str, Any]: Dictionary with answer, sources, context, and timings.
        """
        start_time = time.time()

        try:
            # Retrieve relevant context documents
            if use_hybrid:
                retrieved = self.vector_store.hybrid_search(query, top_k=top_k)
            else:
                retrieved = self.vector_store.retrieve(query, top_k=top_k)
        except Exception as e:
            self.logger.error(f"Error during document retrieval: {e}")
            retrieved = []

        retrieval_time = time.time() - start_time

        if not retrieved:
            self.logger.warning("No documents retrieved for the query.")
            return {
                'answer': "I couldn't find any relevant information in the document to answer your question. Please try rephrasing or asking about a different topic.",
                'sources': [],
                'context': "",
                'retrieval_time_ms': int(retrieval_time * 1000),
                'generation_time_ms': 0
            }

        context = self._build_context(retrieved)
        sources = self._extract_sources(retrieved)

        generation_start = time.time()
        
        # Generate answer using LLM if available, otherwise use smart fallback
        if self.use_llm:
            answer = self._generate_with_anthropic(query, context)
        else:
            answer = self._generate_smart_fallback(query, retrieved)
            
        generation_time = time.time() - generation_start

        return {
            'answer': answer,
            'sources': sources,
            'context': context,
            'retrieval_time_ms': int(retrieval_time * 1000),
            'generation_time_ms': int(generation_time * 1000)
        }

    def _build_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """Build a concatenated context string from retrieved documents."""
        context_parts = []
        for i, doc in enumerate(retrieved_docs):
            page = doc.get('metadata', {}).get('page', 'unknown')
            doc_type = doc.get('metadata', {}).get('type', 'text')
            content = doc.get('content', '')[:800]  # Increased snippet size
            context_parts.append(f"[Document {i+1} - Page {page} - Type: {doc_type}]\n{content}")
        return "\n\n".join(context_parts)

    def _extract_sources(self, retrieved_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract source metadata from retrieved documents."""
        sources = []
        for doc in retrieved_docs:
            metadata = doc.get('metadata', {})
            source_info = {
                'type': metadata.get('type', 'unknown'),
                'page': metadata.get('page', 'unknown'),
                'confidence': 1 - doc.get('distance', 0),
                'content': (doc.get('content', '')[:300] + '...').replace('\n', ' ')
            }
            sources.append(source_info)
        return sources

    def _generate_with_anthropic(self, query: str, context: str) -> str:
        """Generate answer using Anthropic Claude API."""
        try:
            from anthropic import Anthropic
            
            client = Anthropic(api_key=self.api_key)
            
            prompt = f"""Based on the following context from a document, please answer the user's question directly and concisely.

Context from the document:
{context}

User's question: {query}

Please provide a clear, direct answer based solely on the information in the context above. If the context doesn't contain enough information to answer the question, say so. Cite specific pages when relevant."""

            message = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            return message.content[0].text
            
        except ImportError:
            self.logger.warning("Anthropic library not installed. Using fallback.")
            return self._generate_smart_fallback(query, None)
        except Exception as e:
            self.logger.error(f"Error calling Anthropic API: {e}")
            return self._generate_smart_fallback(query, None)

    def _generate_smart_fallback(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        Generate a smart fallback answer by extracting the most relevant content.
        This is used when LLM API is not available.
        """
        if not retrieved_docs:
            return "I couldn't find relevant information to answer your question."
        
        # Get the top result
        top_doc = retrieved_docs[0]
        page = top_doc.get('metadata', {}).get('page', 'unknown')
        content = top_doc.get('content', '')
        
        # Extract the most relevant sentences (simple approach)
        sentences = content.split('.')
        relevant_sentences = []
        query_terms = set(query.lower().split())
        
        for sentence in sentences[:10]:  # Look at first 10 sentences
            sentence_terms = set(sentence.lower().split())
            overlap = len(query_terms & sentence_terms)
            if overlap > 0:
                relevant_sentences.append(sentence.strip())
        
        if relevant_sentences:
            answer = '. '.join(relevant_sentences[:3])  # Take top 3 sentences
            answer = answer.strip() + '.'
            answer += f"\n\n(Source: Page {page})"
            return answer
        else:
            # Just return the beginning of the top document
            preview = content[:500].strip()
            return f"{preview}...\n\n(Source: Page {page})"