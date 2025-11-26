from typing import List, Tuple, Dict, Any
import re

class SemanticChunker:
    """Intelligently chunks documents preserving semantic meaning"""
    
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_text(self, text: str, page_num: int) -> List[Dict[str, Any]]:
        """Chunk text with semantic boundaries"""
        # Split by sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            
            if current_length + sentence_length > self.chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    'type': 'text',
                    'content': chunk_text,
                    'page': page_num,
                    'word_count': current_length
                })
                
                # Add overlap
                overlap_sentences = current_chunk[-2:] if len(current_chunk) >= 2 else current_chunk
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(s.split()) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Add remaining chunk
        if current_chunk:
            chunks.append({
                'type': 'text',
                'content': ' '.join(current_chunk),
                'page': page_num,
                'word_count': current_length
            })
        
        return chunks
    
    def chunk_table(self, table: List[List[str]], page_num: int) -> Dict[str, Any]:
        """Convert table to searchable format"""
        # Convert table to markdown-like text
        text_representation = []
        
        # Headers
        if table and len(table) > 0:
            headers = table[0]
            text_representation.append(' | '.join(headers))
            
            # Rows
            for row in table[1:]:
                text_representation.append(' | '.join(str(cell) for cell in row))
        
        return {
            'type': 'table',
            'content': '\n'.join(text_representation),
            'page': page_num,
            'metadata': {
                'rows': len(table),
                'columns': len(table[0]) if table else 0
            }
        }
