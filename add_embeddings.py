# %%

import rich
from config import (
    INPUT_FILE,
    PLOT_FIELD,
    VECTOR_FIELD,
    MAX_VECTOR_LENGTH,
)
from bson.json_util import dumps, loads
from gpt4all import Embed4All
from rich.progress import track

import typer

embedder = Embed4All()  # default model is 'all-MiniLM-L6-v2-f16.gguf'


# %%


def load_docs(source_file):
    with open(source_file, "r") as fh:
        for line in fh.readlines():
            yield loads(line)


# %%


def add_embeddings(original_docs, plot_field, vector_field_name):
    for doc in track(original_docs, "Adding embedding..."):
        text = doc.get(plot_field)

        if not text:
            rich.print(f"No plot for {doc.get('_id')} {doc.get('title')}")
            continue

        vector = embedder.embed(text)

        doc[vector_field_name] = vector[:MAX_VECTOR_LENGTH]

        if len(vector) > MAX_VECTOR_LENGTH:
            typer.Abort(
                f"{len(vector)} exceeds allowed vector length {MAX_VECTOR_LENGTH}"
            )
        yield doc


# %%
def main():
    source_file = typer.prompt("Original documents file?", INPUT_FILE)
    plot_field = typer.prompt("Original document text field", PLOT_FIELD)
    vector_field_name = typer.prompt("Field storing embedding vector", VECTOR_FIELD)

    target_file = source_file.replace(".json", ".gpt4all.json")

    original_docs = load_docs(source_file)

    with open(target_file, "w") as output_file:
        for doc in add_embeddings(original_docs, plot_field, vector_field_name):
            output_file.write(dumps(doc) + "\n")


if __name__ == "__main__":
    typer.run(main)
