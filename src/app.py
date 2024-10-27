import os

from logging import logging
from exception import AppException

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
    


def create_rag_chain(llm, retriever):
    """
    Runs the RAG chain with the provided LLM and retriever.

    Args:
        llm: The language model to use.
        retriever: The retriever to use.

    Returns:
        The results of the RAG chain.
    """

    system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt), 
            ("human", "{input}")
        ]
    )
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain
