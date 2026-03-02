"""Helper utilities for embeddings, vector store, and QA chain."""

from __future__ import annotations

import os
from typing import Optional

from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore

from src.prompt import SYSTEM_PROMPT


def download_hugging_face_embeddings(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> HuggingFaceEmbeddings:
    """Return a configured embedding model."""
    return HuggingFaceEmbeddings(model_name=model_name)


def _build_prompt() -> PromptTemplate:
    return PromptTemplate(template=SYSTEM_PROMPT, input_variables=["context", "question"])


def build_qa_chain(index_name: str, namespace: Optional[str] = None) -> RetrievalQA:
    """Create RetrievalQA chain backed by Pinecone.

    Requires:
      - OPENAI_API_KEY
      - PINECONE_API_KEY
      - PINECONE_INDEX_NAME (passed as index_name)
    """
    embeddings = download_hugging_face_embeddings()
    vector_store = PineconeVectorStore(index_name=index_name, embedding=embeddings, namespace=namespace)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": _build_prompt()},
        return_source_documents=False,
    )
    return qa_chain
