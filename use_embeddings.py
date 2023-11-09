# %%
from config import CONNECTION_ENV_VARIABLE, DB, COLLECTION

from pymongo.errors import ServerSelectionTimeoutError
import os

from gpt4all import Embed4All
import typer
from rich import print_json
from atlas_vector_search import create_query, run_query

embedder = Embed4All()  # default model is 'all-MiniLM-L6-v2-f16.gguf'


# %%


def main():
    prompt = typer.prompt("What's the movie about?")
    collection_name = typer.prompt("Collection name?", COLLECTION)
    typer.echo(f"K. Looking for a movie withe the concept: '{prompt}' ...")
    query = create_query(embedder=embedder, prompt=prompt)

    print_json(data=query)

    if typer.confirm("Run the query?"):
        typer.echo(f"Using {CONNECTION_ENV_VARIABLE} to connect...")
        try:
            docs = run_query(
                query,
                os.environ.get(CONNECTION_ENV_VARIABLE),
                db_name=DB,
                collection_name=collection_name,
            )
            [print_json(data=doc) for doc in docs]
        except ServerSelectionTimeoutError:
            typer.confirm("Can't connect... Try Again?")

    if typer.confirm("Again?"):
        main()


if __name__ == "__main__":
    typer.run(main)
