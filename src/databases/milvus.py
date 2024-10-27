import os

from logging import logging
from exception import AppException

from pymilvus import MilvusClient, connections, FieldSchema, CollectionSchema, DataType, Collection, utility

from ai_models.embedding import load_bge_embed_func


#Create vector store instance
def create_milvus():
    client = MilvusClient(uri="./milvus_demo.db")

    logging.info("CREATED Milvus client")

    return client


# Create a collection and add documents to it
def create_or_load_collection(collection_name, client, embed_dim):
    
    try:
        if client.has_collection(collection_name):
            client.load_collection(collection_name)

            logging.info(f"LOADED Collection with name {collection_name}")
        
        else:
            # Create a collection with a vector field
            collection_schema = CollectionSchema(
                description=collection_name,
                fields=[
                    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True)
                    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embed_dim)
                ]
            )

            index_params = client.prepare_index_params()

            # Add an index on the vector field.
            index_params.add_index(
                field_name="embedding",
                metric_type="COSINE",
                index_type="FLAT",
                index_name="vector_index",
                params={ "nlist": 128 }
            )

            client.create_collection(
                collection_name=collection_name,
                schema=collection_schema,
                index_params=index_params,
            )

            logging.info(f"CREATED Collection with name {collection_name}")

    except Exception as e:
        raise AppException(e, sys)
    
    return {"message": "Collection created successfully!",
            "collection_status": client.get_load_status(collection_name = collection_name)
            }


# Add documents to the collection
def add_documents_to_collection(collection_name, client, documents, embed_model):
    try:
        data = []
        for doc in documents:
            vector = embed_model.encode_documents([doc.page_content])["dense"][0].to_list()
            data.append({"embedding": vector})

        client.insert(collection_name=collection_name, data=data)

        logging.info(f"Added {len(documents)} documents to collection {collection_name}")

    except Exception as e:
        raise AppException(e, sys)
    
    return {"message": "Document uploaded successfully!",
            "collection_stats": client.get_collection_stats(collection_name = collection_name)
            }


