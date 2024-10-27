import os
import time

from logging import logging
from exception import AppException

from pymilvus import MilvusClient, connections, FieldSchema, CollectionSchema, DataType, Collection, utility

from ai_models.embedding import load_bge_embed_func, load_sparse_embedding_func


#Create vector store instance
def create_milvus():
    '''
    Creates a Milvus client.

    Returns:
        A MilvusClient object.
    '''
    client = MilvusClient(uri="./milvus_demo.db")

    logging.info("CREATED Milvus client")

    return client


# Create a collection and add documents to it
def create_or_load_collection(collection_name, client, embed_dim):
    '''
    Creates a new collection in Milvus if it doesn't exist, or loads an existing collection.

    Args:
        collection_name: The name of the collection.
        client: The MilvusClient object.
        embed_dim: The dimensionality of the embedding vectors.
    '''
    
    try:
        if client.has_collection(collection_name):
            client.load_collection(collection_name)

            logging.info(f"LOADED Collection with name {collection_name}")
        
        else:
            # Create a collection with a vector field
            collection_schema = CollectionSchema(
                description=collection_name,
                fields=[
                    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                    FieldSchema(name="dense_embed", dtype=DataType.FLOAT_VECTOR, dim=embed_dim),
                    FieldSchema(name="sparse_embed", dtype=DataType.SPARSE_FLOAT_VECTOR),
                    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65_535)
                ]
            )

            index_params = client.prepare_index_params()

            # Add an index on the vector field.
            index_params.add_index(
                field_name="dense_embed",
                metric_type="COSINE",
                index_type="FLAT",
                index_name="dense_index",
            )

            index_params.add_index(
                field_name="sparse_embed",
                metric_type="IP",
                index_type="SPARSE_INVERTED_INDEX",
                index_name="sparse_index",
            )

            client.create_collection(
                collection_name=collection_name,
                schema=collection_schema,
                index_params=index_params,
                consistency_level="Strong",
            )

            logging.info(f"CREATED Collection with name {collection_name}")

    except Exception as e:
        raise AppException(e, sys)
    
    return {"message": "Collection created successfully!",
            "collection_status": client.get_load_status(collection_name = collection_name)
            }


# Add documents to the collection
def add_documents_to_collection(collection_name, client, documents, embed_model, sparse_embed_model):
    '''
    Adds documents to the specified Milvus collection.

    This function iterates through a list of documents, generates embeddings for each document 
    using the provided embedding models, and inserts the data into the Milvus collection.

    Args:
        collection_name: The name of the Milvus collection to add documents to.
        client: The MilvusClient object.
        documents: A list of Document objects to be added.
        embed_model: The model used to generate dense embeddings.
        sparse_embed_model: The model used to generate sparse embeddings.

    '''
    try:
        start = time.time()
        data = []
        for doc in documents:
            data.append({
                "dense_embed": list(bge_m3_ef.encode_documents([doc.page_content])["dense"][0]),
                "sparse_embed": list(sparse_embedding_func.embed_documents([doc.page_content])[0]),
                "text": doc.page_content
            })
        end = time.time()


        insert_start_time = time.time() 
        client.insert(collection_name=collection_name, data=data)
        insert_end_time = time.time()

        logging.info(f"Added {len(documents)} documents to collection {collection_name}")

    except Exception as e:
        raise AppException(e, sys)
    
    return {"state": client.get_load_state(collection_name = collection_name)['state'],
            "collection_stats": client.get_collection_stats(collection_name = collection_name),
            "time_taken": {
                "to_create_vectors": end - start,
                "to_insert_in_collection": insert_end_time - insert_start_time
                }
            }


# Convert collection into retriever
def convert_collection_to_retriever(collection_name, client, embed_model, sparse_embed_model, k=3):
    '''
    Converts a Milvus collection into a retriever.

    Args:
        collection_name: The name of the Milvus collection.
        client: The MilvusClient object.
        embed_model: The model used to generate dense embeddings.
        sparse_embed_model: The model used to generate sparse embeddings.
        k: The number of documents to retrieve.

    Returns:
        A retriever object.
    '''
   
    sparse_search_params = {"metric_type": "IP"}
    dense_search_params = {"metric_type": "COSINE", "params": {}}
    
    try:

        retriever = MilvusCollectionHybridSearchRetriever(
            collection=Collection(collection_name),
            anns_fields=[dense_embed, sparse_embed],
            field_embeddings=[embed_model, sparse_embed_model],
            field_search_params=[dense_search_params, sparse_search_params],
            text_field=text,
            top_k=k,
        )

        logging.info(f"CONVERTED Collection with name {collection_name} into retriever")
    
    except Exception as e:
        raise AppException(e, sys)
    
    return retriever



