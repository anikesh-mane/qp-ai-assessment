import os
import shutil

from logging import logging
from exception import AppException

from extration.pdf import langchain_pdf_loader
from databases.milvus import create_or_load_collection, add_documents_to_collection
from ai_models.embedding import load_bge_embed_func, load_sparse_embedding_func

from fastapi import APIRouter, UploadFile, Request, HTTPException, File

db_router = APIRouter()

@router.post("/upload")
async def upload_file(collection_name: str, request: Request, file: UploadFile = File(...)):
    '''
    Uploads a PDF file, chunks it, generates embeddings, and stores it in Milvus.
    
    Args:
        collection_name (str): Name of the collection to store the file in.
        request (Request): FastAPI Request object.
        file (UploadFile, optional): The uploaded PDF file. Defaults to File(...).

    Returns:
        dict: A dictionary containing a success message or an error message. 
               e.g., {"message": "File uploaded and processed successfully"}
    '''

    try:
        if file.filename.split(".")[-1] != "pdf":
            raise HTTPException(status_code=400, detail="File must be a PDF")

        # Save the uploaded file to a temporary directory
        with open(f"temp/{file.filename}", "wb") as f:
            shutil.copyfileobj(file.file, f)

        logging.info("File saved to temporary directory")

        # Chunk the file
        documents = langchain_pdf_loader(file_path=f"./temp/{file.filename}")
        logging.info("File chunked")

        # Create an embedding functions
        sparse_embed = load_sparse_embedding_func(documents)
        request.app.sparse_embed = sparse_embed

        # initialize milvus collection
        collection_status = create_or_load_collection(collection_name = collection_name, 
                                                        client = request.app.milvus_client, 
                                                        embed_dim = request.app.embed_dim
                                                        )
        logging.info(collection_status)
        
        # Push the documents and embeddings to collection
        status = add_documents_to_collection(collection_name = collection_name, 
                                                client = request.app.milvus_client, 
                                                embed_model = request.app.dense_embed, 
                                                sparse_embed_model = request.app.sparse_embed,
                                                documents = documents
                                                )

        # Remove the temporary file\
        os.remove(f"./temp/{file.filename}")
        
        logging.info("Temporary file removed")

        return {"message": "Document uploaded successfully!"} | status

    except Exception as e:
        raise AppException(e, sys)


@router.get("/list_collections")
def get_collections(request: Request):
    return {"collections": request.app.milvus_client.list_collections()}


@router.get('/delete_collection/{collection_name}')
def delete_collection(collection_name: str, request: Request):
    request.app.milvus_client.drop_collection(collection_name)
    return {"message": "Collection deleted successfully!"}
