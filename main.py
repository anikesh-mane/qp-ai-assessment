from logger import logger
from exception import AppException

from dotenv import dotenv_values

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.databases.db_api import db_router
from src.databases.milvus import create_milvus, convert_collection_to_retriever
from src.ai_models.embedding import load_bge_embed_func, load_sparse_embedding_func
from src.ai_models.text_generation import load_hf_llm_model
from src.chains.retrieval_qa_chain import create_qa_chain

env_vars = dotenv_values('.env')

# 
class QuestionRequest(BaseModel):
    question: str
    collection_name: str
    k: int = 3


app = FastAPI()

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
        app.milvus_client = create_milvus(uri=env_vars['MILVUS_URI'])

        app.dense_embed, app.embed_dim =  load_bge_embed_func(model_name='BAAI/bge-m3', 
                                                                device='cpu', 
                                                                use_fp16=False
                                                                )

        app.llm = load_hf_llm_model(hf_api_key=env_vars['HF_TOKEN'])

    except Exception as e:
        raise AppException(e, sys)


@app.post("/query_by_collection")
async def ask_question(request: Request, question_request: QuestionRequest):

    collection_name = question_request.collection_name
    k=question_request.k
    
    # Convert collection into retriever
    retiever = convert_collection_to_retriever(collection_name=collection_name,
                                                 client=request.app.milvus_client, 
                                                 embed_model=request.app.dense_embed, 
                                                 sparse_embed_model=request.app.sparse_embed, 
                                                 k=k
                                                 )
    
    # Log the incoming request data
    logging.info("Received question:", question_request.question)

    # Create Chain
    rag_chain = create_qa_chain(llm=request.app.llm, retriever=retiever)

    # Run Chain
    resp = rag_chain.run(question_request.question)
    
    return resp


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)
    

    



