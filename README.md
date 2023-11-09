# Atlas Vector Search Demo (AI)

The repo contains scripts that demonstrate using embeddings to perform semantic search of your own data in MongoDB Atlas.

The scripts include: 

1. Adding embeddings to a set of documents
2. Uploading a dataset to Atlas
3. Using the data to perform semantic search against your own data

Two sample datasets are already processed and include vector embeddings inside them. You can use MongoDB Compass or the upload script to upload all the parts. 

The embeddings are generated using `Gpt4All`'s `Embed4All` module. The vectors generated are 384 in length.

The demo app `use_embeddings.py` queries an **Atlas Search** index populated with the embedding data. It transforms your text prompt to a vector, then uses the vector in a $vectoSearch aggregation stage to find matches.

Usage: 
```
$: python use_embeddings.py
```

> Set the environment varibale `MONGO_URL` to connection string of your Atlas database.


   
