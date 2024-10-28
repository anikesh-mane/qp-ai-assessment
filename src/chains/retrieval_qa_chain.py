import os

from logger import logger
from exception import AppException

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


# function to load the RAG chain
def create_retreival_qa_chain(llm, retriever):
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
    "Use the following pieces of retrieved context to answer the question."
    "If you don't know the answer, say 'I DONT KNOW THE ANSWER TO THIS QUESTION !'."
    "If you cannot find the answer in the given context, say 'THIS INFO IS NOT PROVIDED IN GIVEN CONTEXT !'"
    "Use three sentences maximum and keep the answer concise."
    "\n\n"
    "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt), 
            ("human", "{input}")
        ]
    )
    
    try:
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    except Exception as e:
        raise AppException(e, sys)

    return rag_chain
