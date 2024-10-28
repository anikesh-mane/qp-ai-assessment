from logger import logger
from exception import AppException

from pymilvus import model
from langchain_milvus.utils.sparse import BM25SparseEmbedding

# Create a BGE-M3 embedding function
def load_bge_embed_func(model_name='BAAI/bge-m3', device='cpu', use_fp16=False):
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

        bge_m3_ef = model.hybrid.BGEM3EmbeddingFunction(
            model_name=model_name, # Specify t`he model name
            device=device, # Specify the device to use, e.g., 'cpu' or 'cuda:0'
            use_fp16=use_fp16 # Whether to use fp16. `False` for `device='cpu'`.
        )

        logger.info("CREATED BGE-M3 embedding function")

        test_embedding = bge_m3_ef.encode_documents(["This is a test"])
        embed_dim = test_embedding["dense"][0].shape
    
    except Exception as e:
        raise AppException(e, sys)
    
    return bge_m3_ef, embed_dim

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
        text.append(doc.page_content)
    
    try:
        sparse_embedding_func = BM25SparseEmbedding(corpus=text)
        logger.info("CREATED sparse embedding function")
    
    except Exception as e:
        raise AppException(e, sys)

    return sparse_embedding_func