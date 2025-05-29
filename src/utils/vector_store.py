"""Vector store utilities for embedding and storing medical text."""
from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from dataclasses import dataclass
import pickle
import os

@dataclass
class Document:
    """Document class to store text and its metadata."""
    text: str
    metadata: dict = None

class VectorStore:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the vector store with a sentence transformer model.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        self.documents: List[Document] = []
        
    def add_texts(self, texts: List[str], metadatas: Optional[List[dict]] = None) -> None:
        """
        Add texts to the vector store.
        
        Args:
            texts: List of text strings to add
            metadatas: Optional list of metadata dicts for each text
        """
        if not texts:
            return
            
        if metadatas is None:
            metadatas = [{} for _ in texts]
            
        # Create Document objects
        documents = [Document(text=text, metadata=meta) 
                    for text, meta in zip(texts, metadatas)]
        
        # Get embeddings
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        
        # Add to FAISS index
        self.index.add(np.array(embeddings).astype('float32'))
        self.documents.extend(documents)
        
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Search for similar documents using the query.
        
        Args:
            query: Query text
            k: Number of documents to return
            
        Returns:
            List of most similar documents
        """
        # Check if index is empty
        if self.index.ntotal == 0:
            return []
            
        # Adjust k if we have fewer documents than requested
        k = min(k, self.index.ntotal)
        
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        
        # Search the index
        scores, indices = self.index.search(
            np.array(query_embedding).astype('float32'), k
        )
        
        # Return the documents
        return [self.documents[i] for i in indices[0]]
    
    def save(self, directory: str) -> None:
        """
        Save the vector store to disk.
        
        Args:
            directory: Directory to save the vector store
        """
        os.makedirs(directory, exist_ok=True)
        
        # Save the FAISS index
        faiss.write_index(self.index, os.path.join(directory, "index.faiss"))
        
        # Save the documents
        with open(os.path.join(directory, "documents.pkl"), "wb") as f:
            pickle.dump(self.documents, f)
            
    @classmethod
    def load(cls, directory: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Load a vector store from disk.
        
        Args:
            directory: Directory containing the vector store
            model_name: Name of the sentence transformer model
            
        Returns:
            Loaded VectorStore instance
        """
        vector_store = cls(model_name)
        
        # Load the FAISS index
        vector_store.index = faiss.read_index(os.path.join(directory, "index.faiss"))
        
        # Load the documents
        with open(os.path.join(directory, "documents.pkl"), "rb") as f:
            vector_store.documents = pickle.load(f)
            
        return vector_store 