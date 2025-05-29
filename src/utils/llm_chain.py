"""LangChain integration with Groq LLM for medical text simplification."""
from typing import List, Optional
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema import Document as LangchainDocument
import os

from .vector_store import Document

# Default prompt template for medical text simplification
DEFAULT_PROMPT_TEMPLATE = """You are a helpful medical assistant that explains complex medical concepts in simple terms.

Context information is below:
-------------------
{context}
-------------------

Given the context above, please explain the following in simple, patient-friendly terms:
{query}

Your explanation should be:
1. Easy to understand for non-medical professionals
2. Accurate and grounded in the provided context
3. Empathetic and clear
4. Include any relevant precautions or next steps if applicable

Explanation:"""

class MedicalExplanationChain:
    def __init__(
        self, 
        groq_api_key: Optional[str] = None,
        model_name: str = "qwen-qwq-32b",
        temperature: float = 0.3,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE
    ):
        """
        Initialize the medical explanation chain.
        
        Args:
            groq_api_key: Groq API key (defaults to env var GROQ_API_KEY)
            model_name: Name of the Groq model to use
            temperature: Temperature for generation
            prompt_template: Custom prompt template (optional)
        """
        if groq_api_key is None:
            groq_api_key = os.getenv("GROQ_API_KEY")
            if groq_api_key is None:
                raise ValueError("GROQ_API_KEY not found in environment variables")
        
        # Initialize Groq LLM
        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=model_name,
            temperature=temperature
        )
        
        # Create prompt
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "query"]
        )
        
        # Create the document chain
        self.chain = create_stuff_documents_chain(llm, prompt)
        
    def generate_explanation(
        self, 
        query: str, 
        relevant_docs: List[Document]
    ) -> str:
        """
        Generate a simplified medical explanation.
        
        Args:
            query: The medical text or concept to explain
            relevant_docs: List of relevant documents for context
            
        Returns:
            Simplified explanation string
        """
        # Convert our Document objects to LangChain Document objects
        langchain_docs = [
            LangchainDocument(
                page_content=doc.text,
                metadata=doc.metadata or {}
            )
            for doc in relevant_docs
        ]
        
        # Run the chain
        result = self.chain.invoke({
            "context": "\n\n".join(doc.text for doc in relevant_docs),
            "query": query
        })
        
        return result 