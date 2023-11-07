# Add Embeddings

Adding embeddints to documents in a json file.


The script loads JSON from input file, creates a vector from a given field, and outputs 

JSON documents to the output file.


Using `Embed4All` library, the default model for embeddings is `all-MiniLM-L6-v2-f16.gguf` at time of writing.


Below is a config for an Atlas Search index to leverage the embedding.

It assumes

- Document field `plot_vector` contains the embedding
- Embedding vector length of `384`
- Using a `cosine` similarity function.


```json
{
  "mappings": {
    "name": "myAiEmbeddingIndex",
    "dynamic": true,
    "fields": {
      "plot_vector": {
        "type": "knnVector",
        "dimensions": 384,
        "similarity": "cosine"
      }
    }
  }
}
```
