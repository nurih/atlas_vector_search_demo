# %%

import json
import pandas as pd

from bson.json_util import dumps

OUTPUT_FILE = "./cmu.movies.json"

# %%

# 1. Wikipedia movie ID
# 2. Freebase movie ID
# 3. Movie name
# 4. Movie release date
# 5. Movie box office revenue
# 6. Movie runtime
# 7. Movie languages (Freebase ID:name tuples)
# 8. Movie countries (Freebase ID:name tuples)
# 9. Movie genres (Freebase ID:name tuples)

meta_df = pd.read_csv(
    "./movie.metadata.tsv",
    sep="\t",
    header=None,
    names=[
        "id",
        "fid",
        "title",
        "released",
        "revenue",
        "runtime",
        "languages",
        "countries",
        "genres",
    ],
).set_index("id")

plots_df = pd.read_csv(
    "./plot_summaries.txt", sep="\t", header=None, names=["id", "plot"]
).set_index("id")

df = meta_df.join(plots_df).dropna(axis="index", subset=["plot"])

# not using the string freebase movie id
df.drop("fid", axis=1, inplace=True)

# %%
df.released = pd.to_datetime(df.released, errors="coerce", exact=False, utc=True)


# %%
# Cleanup foreign-keyed dicts to get list of values
def decode_values(stringified: str):
    if stringified and isinstance(stringified, str):
        return list(json.loads(stringified).values())
    return None


df.languages = df.languages.apply(lambda v: decode_values(v))
df.countries = df.countries.apply(lambda v: decode_values(v))
df.genres = df.genres.apply(lambda v: decode_values(v))

# %%
# prep for JSON dump
df["_id"] = df.index
# %%
# save to a json file
df.to_json(OUTPUT_FILE, orient="records", lines=True)

# %%
