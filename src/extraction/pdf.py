import sys

from logging import logging
from exception import AppException

import PyPDF2
from langchain.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Create a PDF Loader using the file path
def langchain_pdf_loader(file_path, chunk_size=1000, overlap=200):
    try:
        loader = PyPDFLoader(file_path=file_path")
        docs = loader.load()

        logging.info("PDF loaded")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, 
                                                        chunk_overlap=overlap, 
                                                        # separators=["\n\n", "\n", " ", ""]
                                                        )
        splits_data = text_splitter.split_documents(docs)

        logging.info("PDF splitted into chunks")
    
    except Exception as e:
        raise AppException(e, sys)

    return splits_data


# Create a PDF Loader using the file path
def pypdf_loader(file_path, chunk_size=1000, overlap=200):
    try:
        pdf_file = open(file_path, 'rb')
        pdf_reader = PyPDF2.PdfReader(pdf_file)

        text = []
        for i in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[i]
            text.append(page.extract_text())

        logging.info("PDF loaded")

        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            if end > len(text):
                end = len(text)  # Handle the last chunk
            chunks.append(text[start:end])
            start += chunk_size - overlap  # Adjust starting position with overlap
        
        return chunks




