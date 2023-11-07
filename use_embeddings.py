# %%
from config import CONNECTION_ENV_VARIABLE, DB, COLLECTION, INDEX_NAME, VECTOR_FIELD
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError
import os

from gpt4all import Embed4All
from pymongo import MongoClient
import typer
from rich import print_json

embedder = Embed4All()  # default model is 'all-MiniLM-L6-v2-f16.gguf'

# %%
# Test embedding to ensure it produces differing vectors given differing text
import numpy as np


def test_embedding():
    v1 = np.array(embedder.embed("french fries"))
    v2 = np.array(embedder.embed("milkshake"))

    diff = v1 == v2
    same = np.count_nonzero(diff)
    different = diff.size - same

    print(f"Vector elements match {same} times, {different} differ.")


# %%


def create_query(text):
    vector = embedder.embed(text)

    pipeline = [
        {
            "$vectorSearch": {
                "index": INDEX_NAME,
                "path": VECTOR_FIELD,
                "queryVector": vector,
                "numCandidates": 256,
                "limit": 9
                # "filter": {<filter-specification>}
            }
        },
        {
            "$project": {
                "score": {"$meta": "vectorSearchScore"},
                "title": 1,
                "year": 1,
                "fullplot": 1,
                "_id": 0,
            }
        },
    ]
    return pipeline


def run_query(query):
    collection = (
        MongoClient(os.environ.get(CONNECTION_ENV_VARIABLE))
        .get_database(DB)
        .get_collection(COLLECTION)
    )

    for doc in collection.aggregate(pipeline=query):
        print_json(data=doc)


def main():
    query_text = typer.prompt("What's the movie about?")
    typer.echo(f"K. Looking for a movie withe the concept: '{query_text}' ...")
    query = create_query(query_text)

    print_json(data=query)

    if typer.confirm("Run the query?"):
        typer.echo(f"Using {CONNECTION_ENV_VARIABLE} to connect...")
        try:
            run_query(query)
        except ServerSelectionTimeoutError:
            typer.confirm("Can't connect... Try Again?")

    if typer.confirm("Again?"):
        main()


if __name__ == "__main__":
    typer.run(main)
