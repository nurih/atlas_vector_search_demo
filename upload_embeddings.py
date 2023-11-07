# %%

from bson import json_util
from click import progressbar
from config import (
    CONNECTION_ENV_VARIABLE,
    DB,
    COLLECTION,
    INDEX_NAME,
    VECTOR_FIELD,
)
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError
from pymongo.collection import Collection
from pymongo.operations import SearchIndexModel
from rich.progress import track
from rich import print_json
import os
import typer

# %%


def batches(array, batch_size=100):
    return [
        array[offset : offset + batch_size]
        for offset in range(0, len(array), batch_size)
    ]


def get_collection():
    return (
        MongoClient(os.environ.get(CONNECTION_ENV_VARIABLE))
        .get_database(DB)
        .get_collection(COLLECTION)
    )


def upload(collection, docs):
    if typer.confirm("Delete existing docs?"):
        collection.delete_many({})

    for batch in track(batches(docs)):
        collection.insert_many(batch)


def get_docs(filename):
    with open(filename, "r") as f:
        for line in f:
            yield json_util.loads(line)


def create_vector_index(collection: Collection):
    index_definition = SearchIndexModel(
        definition={
            "mappings": {
                "dynamic": True,
                "fields": {
                    "plot_vector": {
                        "dimensions": 384,
                        "similarity": "cosine",
                        "type": "knnVector",
                    }
                },
            }
        },
        name=INDEX_NAME,
    )

    return collection.create_search_index(index_definition)


# %%


def main():
    typer.echo("Upload to Atlas:")

    typer.echo(f"Using {CONNECTION_ENV_VARIABLE} to connect...")

    try:
        collection = get_collection()

    except ServerSelectionTimeoutError:
        if typer.confirm("Can't connect... Try Again?"):
            main()

    source_file = typer.prompt("Enter the JSON file containing documents containing embedding:")
    
    if typer.confirm(
        f"Upload {source_file} to {DB}.{COLLECTION}? (embedding field {VECTOR_FIELD})"
    ):
        upload(collection, [d for d in get_docs(source_file)])

    if typer.confirm("Create index?"):
        print_json(create_vector_index(collection))

    typer.echo("Done.")


if __name__ == "__main__":
    typer.run(main)
