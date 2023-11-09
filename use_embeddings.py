# %%
from config import CONNECTION_ENV_VARIABLE, DB, COLLECTION, INDEX_NAME, VECTOR_FIELD
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError
import os

from gpt4all import Embed4All
from pymongo import MongoClient
import typer
from rich import print_json
from atlas_vector_search import create_query

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

def run_query(query, collection):
    collection = (
        MongoClient(os.environ.get(CONNECTION_ENV_VARIABLE))
        .get_database(DB)
        .get_collection(collection)
    )

    for doc in collection.aggregate(pipeline=query):
        print_json(data=doc)


def main():
    prompt = typer.prompt("What's the movie about?")
    collection_name= typer.prompt("Collection name?", COLLECTION)
    typer.echo(f"K. Looking for a movie withe the concept: '{prompt}' ...")
    query = create_query(embedder=embedder,prompt=prompt)

    print_json(data=query)

    if typer.confirm("Run the query?"):
        typer.echo(f"Using {CONNECTION_ENV_VARIABLE} to connect...")
        try:
            run_query(query, collection=collection_name)
        except ServerSelectionTimeoutError:
            typer.confirm("Can't connect... Try Again?")

    if typer.confirm("Again?"):
        main()


if __name__ == "__main__":
    typer.run(main)
