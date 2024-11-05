import os
import shutil
import sys
from src.logger import logger
from src.exception import AppException

from dotenv import dotenv_values

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.databases.db_api import router as db_router
from src.databases.milvus import create_milvus, convert_collection_to_retriever
from src.ai_models.embedding import load_hf_embed_func, load_mistral_embed_func, load_sparse_embedding_func
from src.ai_models.text_generation import load_hf_llm_model
from src.chains.retrieval_qa_chain import create_retreival_qa_chain

env_vars = dotenv_values('.env')

# 
class QuestionRequest(BaseModel):
    question: str
    collection_name: str
    k: int = 3


app = FastAPI(title='QP-AI-Chatbot', 
                version='0.0.1')

app.include_router(db_router, tags=["MilvusDB"])

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    # Create milvus client, embedding func and llm
    try:

        app.milvus_client = create_milvus(db_uri=env_vars['MILVUS_LOCAL_URI'])

        logger.info("MILVUS CLIENT ADDED to app")

        app.dense_embed, app.embed_dim =  load_hf_embed_func( 
                                                                model_name=env_vars['EMBED_MODEL_HF_PATH'],
                                                                device='cuda'
                                                                )
        logger.info(f"DENSE EMBEDD MODEL {env_vars['EMBED_MODEL_HF_PATH']} ADDED to app")

        app.llm = load_hf_llm_model(hf_api_key=str(env_vars['HF_TOKEN']),
                                    model_id=env_vars['LLM_HF_PATH'],
                                    )

        logger.info(f"LLM ADDED {env_vars['LLM_HF_PATH']} to app")

        app.env_vars = env_vars

    except Exception as e:
        raise AppException(e, sys)

@app.get('/')
def home():
    return "API is ready to use !"


@app.post("/query_by_collection")
async def ask_question(request: Request, question_request: QuestionRequest):

    collection_name = question_request.collection_name
    k=question_request.k
    
    # Convert collection into retriever
    retiever = convert_collection_to_retriever(collection_name=collection_name,
                                                 env_vars=request.app.env_vars, 
                                                 embed_model=request.app.dense_embed, 
                                                 sparse_embed_model=request.app.sparse_embed, 
                                                 k=k
                                                 )
    
    # Log the incoming request data
    logger.info("Received question:", question_request.question)

    # Create Chain
    rag_chain = create_retreival_qa_chain(llm=request.app.llm, retriever=retiever)

    # Run Chain
    resp = rag_chain.invoke({'input':question_request.question})
    
    return resp


# if __name__ == "__main__":
#     uvicorn.run(app, host="localhost", port=8080)
    

    



