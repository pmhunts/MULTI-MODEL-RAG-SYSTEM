import faiss
import numpy as np
import pickle
import uuid
from typing import List, Dict, Any
from pathlib import Path

class MultiModalVectorStore:
    """Vector store using FAISS - works on Windows without build tools"""
    
    def __init__(self, collection_name: str = "multimodal_docs", persist_directory: str = "./vector_db"):
        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(exist_ok=True)
        
        # Initialize FAISS index (384 dimensions for all-MiniLM-L6-v2)
        self.dimension = 384
        self.index = faiss.IndexFlatL2(self.dimension)
        
        # Storage for documents and metadata
        self.documents = []
        self.metadatas = []
        self.ids = []
        
        # Initialize embedder
        from embedding.multimodal_embedder import MultiModalEmbedder
        self.embedder = MultiModalEmbedder()
        
        # Try to load existing index
        self._load_index()
    
    def add_documents(self, chunks: List[Dict[str, Any]]):
        """Add document chunks to vector store"""
        if not chunks:
            return
        
        embeddings = []
        
        for chunk in chunks:
            chunk_id = str(uuid.uuid4())
            
            # Generate embedding based on type
            if chunk['type'] in ['text', 'table']:
                embedding = self.embedder.embed_text(chunk['content'])
            elif chunk['type'] == 'image':
                # For images, embed the OCR text or description
                embedding = self.embedder.embed_text(chunk.get('ocr_text', ''))
            else:
                continue
            
            embeddings.append(embedding)
            self.ids.append(chunk_id)
            self.documents.append(chunk['content'])
            self.metadatas.append({
                'type': chunk['type'],
                'page': chunk.get('page', 0),
                'source': chunk.get('source', 'unknown')
            })
        
        # Convert to numpy array and add to FAISS index
        if embeddings:
            embeddings_array = np.array(embeddings).astype('float32')
            self.index.add(embeddings_array)
            
            # Save index after adding
            self._save_index()
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks for a query"""
        if self.index.ntotal == 0:
            return []
        
        # Generate query embedding
        query_embedding = self.embedder.embed_text(query)
        query_array = np.array([query_embedding]).astype('float32')
        
        # Search in FAISS index
        distances, indices = self.index.search(query_array, min(top_k, self.index.ntotal))
        
        # Format results
        retrieved_docs = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):  # Valid index
                retrieved_docs.append({
                    'id': self.ids[idx],
                    'content': self.documents[idx],
                    'metadata': self.metadatas[idx],
                    'distance': float(distances[0][i])
                })
        
        return retrieved_docs
    
    def hybrid_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Hybrid search combining vector and keyword matching"""
        # Vector search
        vector_results = self.retrieve(query, top_k=top_k * 2)
        
        if not vector_results:
            return []
        
        # Simple keyword filtering
        query_terms = set(query.lower().split())
        
        # Score results
        for result in vector_results:
            content_terms = set(result['content'].lower().split())
            keyword_overlap = len(query_terms & content_terms) / len(query_terms) if query_terms else 0
            
            # Combine scores (adjust weights as needed)
            # Lower distance is better, so we invert it
            vector_score = 1.0 / (1.0 + result['distance'])
            result['hybrid_score'] = (
                0.7 * vector_score +      # Vector similarity
                0.3 * keyword_overlap     # Keyword overlap
            )
        
        # Sort by hybrid score and return top_k
        vector_results.sort(key=lambda x: x['hybrid_score'], reverse=True)
        return vector_results[:top_k]
    
    def _save_index(self):
        """Save FAISS index and metadata to disk"""
        try:
            # Save FAISS index
            index_path = self.persist_directory / f"{self.collection_name}.index"
            faiss.write_index(self.index, str(index_path))
            
            # Save metadata
            metadata_path = self.persist_directory / f"{self.collection_name}.pkl"
            with open(metadata_path, 'wb') as f:
                pickle.dump({
                    'documents': self.documents,
                    'metadatas': self.metadatas,
                    'ids': self.ids
                }, f)
        except Exception as e:
            print(f"Warning: Could not save index: {e}")
    
    def _load_index(self):
        """Load FAISS index and metadata from disk"""
        try:
            index_path = self.persist_directory / f"{self.collection_name}.index"
            metadata_path = self.persist_directory / f"{self.collection_name}.pkl"
            
            if index_path.exists() and metadata_path.exists():
                # Load FAISS index
                self.index = faiss.read_index(str(index_path))
                
                # Load metadata
                with open(metadata_path, 'rb') as f:
                    data = pickle.load(f)
                    self.documents = data['documents']
                    self.metadatas = data['metadatas']
                    self.ids = data['ids']
                
                print(f"✅ Loaded existing index with {self.index.ntotal} vectors")
        except Exception as e:
            print(f"No existing index found or error loading: {e}")
    
    def clear(self):
        """Clear the vector store"""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents = []
        self.metadatas = []
        self.ids = []
        self._save_index()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        return {
            'total_vectors': self.index.ntotal,
            'total_documents': len(self.documents),
            'dimension': self.dimension,
            'collection_name': self.collection_name
        }


# For backwards compatibility with existing code
class FAISSVectorStore(MultiModalVectorStore):
    """Alias for MultiModalVectorStore"""
    pass
