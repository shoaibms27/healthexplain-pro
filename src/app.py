"""Streamlit app for medical text simplification."""
import streamlit as st
import os
from tempfile import NamedTemporaryFile
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from utils.pdf_processor import extract_text_from_pdf
from utils.vector_store import VectorStore
from utils.llm_chain import MedicalExplanationChain

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = VectorStore()

if 'explanation_chain' not in st.session_state:
    try:
        st.session_state.explanation_chain = MedicalExplanationChain()
    except ValueError as e:
        st.error("⚠️ Groq API key not found. Please add your GROQ_API_KEY to the .env file.")
        st.stop()

def process_pdf(uploaded_file):
    """Process an uploaded PDF file."""
    with NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        # Extract text from PDF
        text_chunks = extract_text_from_pdf(tmp_path)
        
        if not text_chunks:
            st.warning("No text could be extracted from the PDF. Please check if the file is text-based and not scanned.")
            return 0
            
        # Add to vector store
        st.session_state.vector_store.add_texts(
            texts=text_chunks,
            metadatas=[{"source": uploaded_file.name} for _ in text_chunks]
        )
        
        return len(text_chunks)
    finally:
        # Clean up temporary file
        os.unlink(tmp_path)

def main():
    st.title("Medical Text Simplifier 🏥")
    st.write("""
    Upload medical documents or enter medical text to get simplified explanations.
    Powered by Groq AI and LangChain.
    """)
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload a medical PDF document (optional)",
        type=['pdf']
    )
    
    if uploaded_file:
        with st.spinner("Processing PDF..."):
            num_chunks = process_pdf(uploaded_file)
            if num_chunks > 0:
                st.success(f"Successfully processed PDF into {num_chunks} chunks!")
    
    # Text input
    query = st.text_area(
        "Enter medical text to explain:",
        height=100,
        placeholder="e.g., The patient suffers from pleural effusion and hemothorax."
    )
    
    if st.button("Generate Explanation"):
        if not query:
            st.error("Please enter some text to explain.")
            return
            
        with st.spinner("Generating explanation..."):
            # Search for relevant context
            relevant_docs = st.session_state.vector_store.similarity_search(
                query=query,
                k=4
            )
            
            if not relevant_docs:
                st.warning("No relevant medical context found in the database. The explanation will be generated without additional context.")
            
            # Generate explanation
            explanation = st.session_state.explanation_chain.generate_explanation(
                query=query,
                relevant_docs=relevant_docs
            )
            
            # Display results
            st.markdown("### Simplified Explanation")
            st.write(explanation)
            
            # Optionally show sources
            if relevant_docs:
                with st.expander("View Source Documents"):
                    for i, doc in enumerate(relevant_docs, 1):
                        st.markdown(f"**Source {i}:**")
                        st.write(doc.text)
                        if doc.metadata:
                            st.write(f"*Source: {doc.metadata.get('source', 'Unknown')}*")
                        st.markdown("---")

if __name__ == "__main__":
    main() 