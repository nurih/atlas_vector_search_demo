
from pymongo import MongoClient
from config import COLLECTION, DB, INDEX_NAME, VECTOR_FIELD


def create_query(embedder, prompt, index_name=INDEX_NAME, vector_field=VECTOR_FIELD):
    vector = embedder.embed(prompt)

    pipeline = [
        {
            "$vectorSearch": {
                "index": index_name,
                "path": vector_field,
                "queryVector": vector,
                "numCandidates": 256,
                "limit": 4
                # "filter": {<filter-specification>}
            }
        },
        {
            "$project": {
                "score": {"$meta": "vectorSearchScore"},
                "title": 1,
                "name": 1,
                "year": 1,
                "plot": 1,
                "_id": 0,
            }
        },
    ]
    return pipeline
  
def run_query(query, connection_url, db_name = DB, collection_name= COLLECTION):
    collection =  MongoClient(connection_url).get_database(db_name).get_collection(collection_name)
    
    return [doc for doc in collection.aggregate(pipeline=query)]