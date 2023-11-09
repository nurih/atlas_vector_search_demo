# %%
from gpt4all import Embed4All, GPT4All
from rich import print
import typer
from rich.pretty import Pretty
from rich.panel import Panel
import pymongo
import os
import config
from atlas_vector_search import create_query, run_query


collection = (
    pymongo.MongoClient(os.environ[config.CONNECTION_ENV_VARIABLE])
    .get_database(config.DB)
    .get_collection("cmu_movies")
)
# %%


def main():
    embedder_model_name = typer.prompt("Embedding model?", "all-MiniLM-L6-v2-f16.gguf")
    embedder = Embed4All(model_name=embedder_model_name)

    gpt_model_name = typer.prompt("GPT model?", "gpt4all-falcon-q4_0.gguf")
    gpt_model = GPT4All(model_name=gpt_model_name)
    
    db_name = typer.prompt("DB name?", config.DB)
    collection_name = typer.prompt("Collection name?", config.COLLECTION)

    run(
        embedder=embedder,
        gpt_model=gpt_model,
        db_name=db_name,
        collection_name=collection_name,
    )


def run(embedder, gpt_model, db_name, collection_name):
    user_prompt = typer.prompt("Give me text:", "birria taco")

    typer.echo(f"Creating embedding for : '{user_prompt}' ...")    

    atlas_query = create_query(embedder=embedder, prompt=user_prompt)

    semantic_search_docs = run_query(
        atlas_query,
        os.environ[config.CONNECTION_ENV_VARIABLE],
        db_name=db_name,
        collection_name=collection_name,
    )

    
    plots = [d.get('plot') for d in semantic_search_docs]
    plots_combined = '.\n\n'.join(plots)
    
    
    typer.echo(f"working with plots of lengths {[len(p) for p in plots]} plots, from {[d.get('title') for d in semantic_search_docs]}.")
    
    engineered_prompt = "Assume the following plots:\n" + plots_combined + '.\n\nGiven these plots, tell me what is a common theme found in all of them, especially?'

    pretty = Pretty(locals())
    panel = Panel(pretty)
    print(panel)
    
    
    # with gpt_model.chat_session():
        
    #     for plot_context in plots:  
    #         x = GPT4All().generate
    #         typer.echo(gpt_model.generate(prompt=plot_context))
                        
        
    typer.echo("-----------------------------")
    typer.echo("-----HERE WE GO -------------")
    typer.echo("-----------------------------")
    
    typer.echo(gpt_model.generate(prompt=engineered_prompt))
    



    if typer.confirm("Again?"):
        run(embedder=embedder, gpt_model=gpt_model)


if __name__ == "__main__":
    typer.run(main)

# %%
