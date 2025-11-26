from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
import torch
from typing import Union, List
import numpy as np

class MultiModalEmbedder:
    """Generates embeddings for text, tables, and images"""
    
    def __init__(self):
        # Text embeddings - use a model with good performance
        self.text_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Vision embeddings - CLIP for image understanding
        self.vision_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.vision_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate text embeddings"""
        embedding = self.text_model.encode(text, convert_to_numpy=True)
        return embedding
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Batch embed multiple texts"""
        embeddings = self.text_model.encode(texts, convert_to_numpy=True)
        return embeddings
    
    def embed_image(self, image) -> np.ndarray:
        """Generate image embeddings using CLIP"""
        inputs = self.vision_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            image_features = self.vision_model.get_image_features(**inputs)
        return image_features.numpy().flatten()
    
    def embed_table(self, table_text: str) -> np.ndarray:
        """Generate table embeddings (treated as structured text)"""
        # Tables are embedded as text but can be weighted differently
        return self.embed_text(table_text)
