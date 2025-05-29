"""PDF processing utilities for medical text extraction."""
import fitz  # PyMuPDF
from typing import List
import re

def extract_text_from_pdf(file_path: str) -> List[str]:
    """
    Extract text from PDF and split into chunks.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        List of text chunks from the PDF
    """
    doc = fitz.open(file_path)
    text_chunks = []
    
    for page in doc:
        text = page.get_text()
        # Clean text
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Split into smaller chunks (roughly 500 characters each)
        chunks = split_into_chunks(text, chunk_size=500)
        text_chunks.extend(chunks)
    
    doc.close()
    return text_chunks

def split_into_chunks(text: str, chunk_size: int = 500) -> List[str]:
    """
    Split text into smaller chunks while trying to preserve sentence boundaries.
    
    Args:
        text: Input text to split
        chunk_size: Target size for each chunk
        
    Returns:
        List of text chunks
    """
    chunks = []
    current_chunk = ""
    
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += sentence + " "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
        
    return chunks 