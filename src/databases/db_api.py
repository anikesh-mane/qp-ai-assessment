import os
import shutil

from logging import logging
from exception import AppException

from extration.pdf import langchain_pdf_loader

from fastapi import APIRouter, UploadFile, Request, HTTPException, File

router = APIRouter()

@router.post("/upload")
async def upload_file(collection_name: str, request: Request, file: UploadFile = File(...)):

    try:
        if file.filename.split(".")[-1] != "pdf":
            raise HTTPException(status_code=400, detail="File must be a PDF")

        # Save the uploaded file to a temporary directory
        with open(f"temp/{file.filename}", "wb") as f:
            shutil.copyfileobj(file.file, f)

        logging.info("File saved to temporary directory")

        # Chunk the file
        text_lines = langchain_pdf_loader(file_path=f"./temp/{file.filename}")
        logging.info("File chunked")

        # Create an embedding function
        

        # Push the documents and embeddings to collection
        


        # Remove the temporary file\
        os.remove(f"./temp/{file.filename}")
        
        logging.info("Temporary file removed")

        return {"message": "Document uploaded successfully!"}

    except Exception as e:
        raise AppException(e, sys)


@router.get("/collections")
def get_collections(request: Request):
    return {"collections": request.app.milvus_client.list_collections()}

@router.get('/reset_db')
def reset_chromadb(response: Request):
    response.app.chroma_client.reset()
    return {"status": "Deleted all data from ChromaDB"}