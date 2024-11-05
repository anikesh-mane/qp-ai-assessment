import sys
from logger import logger
from exception import AppException

from mistralai import Mistral
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_mistralai.embeddings import MistralAIEmbeddings
# from milvus_model.hybrid import BGEM3EmbeddingFunction
from langchain_milvus.utils.sparse import BM25SparseEmbedding

# Create a BGE-M3 embedding function
# def load_bge_embed_func(model_name='BAAI/bge-m3', device='cpu'):
#     '''
#     Create a BGE-M3 embedding function.

#     Args:
#     model_name: The name of the model to use.
#     device: The device to use.
#     use_fp16: Whether to use fp16. `False` for `device='cpu'`.

#     Returns:
#     BGE-M3 embedding function object 
#     '''
#     try:

#         bge_m3_ef = BGEM3EmbeddingFunction(model_name=model_name,
#                                              device=device,
#                                              use_fp16 = True
#                                             )
#         logger.info("CREATED BGE-M3 embedding function")

#         test_embedding = bge_m3_ef.encode_documents(["This is a test"])
#         embed_dim = len(test_embedding[0])
    
#     except Exception as e:
#         raise AppException(e, sys)
    
#     return bge_m3_ef, embed_dim

def load_hf_bge_embed_func(model_name='BAAI/bge-m3', device='cpu'):
    '''
    Create a BGE-M3 embedding function.

    Args:
    model_name: The name of the model to use.
    device: The device to use.
    use_fp16: Whether to use fp16. `False` for `device='cpu'`.

    Returns:
    BGE-M3 embedding function object 
    '''
    try:

        bge_m3_ef = HuggingFaceBgeEmbeddings(model_name=model_name,
                                             model_kwargs={'device':device},
                                            )
        logger.info("CREATED BGE-M3 embedding function")

        test_embedding = bge_m3_ef.embed_documents(["This is a test"])
        embed_dim = len(test_embedding[0])
    
    except Exception as e:
        raise AppException(e, sys)
    
    return bge_m3_ef, embed_dim

def load_mistral_embed_func(mistral_api_key):
    
    try:

        mistral_embed_func = MistralAIEmbeddings(api_key=mistral_api_key)
        logger.info("CREATED Mistral embedding function")

        test_embedding = mistral_embed_func.embed_documents(["This is a test"])
        embed_dim = len(test_embedding[0])
    
    except Exception as e:
        raise AppException(e, sys)

    return mistral_embed_func, embed_dim

# Create a sparse embedding function
def load_sparse_embedding_func(data):
    '''
    Create a sparse embedding function from a list of documents.

    Args:
    data: Langchain documents

    Returns:
    BM25SparseEmbedding object 
    '''
    import nltk
    nltk.download('punkt')

    text = []
    for doc in data:
        text.append(doc.page_contents)
    
    try:
        sparse_embedding_func = BM25SparseEmbedding(corpus=text)
        logger.info("CREATED sparse embedding function")
    
    except Exception as e:
        raise AppException(e, sys)

    return sparse_embedding_func