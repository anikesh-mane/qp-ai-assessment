from logger import logging
from exception import AppException

from pymilvus import model

# Create a BGE-M3 embedding function
def load_bge_embed_func(model_name='BAAI/bge-m3', device='cpu', use_fp16=False):
    try:

        bge_m3_ef = model.hybrid.BGEM3EmbeddingFunction(
            model_name=model_name, # Specify t`he model name
            device=device, # Specify the device to use, e.g., 'cpu' or 'cuda:0'
            use_fp16=use_fp16 # Whether to use fp16. `False` for `device='cpu'`.
        )

        test_embedding = bge_m3_ef.encode_documents(["This is a test"])
        embed_dim = test_embedding["dense"][0].shape

        return bge_m3_ef, embed_dim
    
    except Exception as e:
        raise AppException(e, sys)

# Create a sparse embedding function
from langchain_milvus.utils.sparse import BM25SparseEmbedding

text = []
for doc in splits_data:
    text.append(doc.page_content)

sparse_embedding_func = BM25SparseEmbedding(corpus=text)