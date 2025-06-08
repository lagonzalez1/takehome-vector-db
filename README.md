> bugs and is super verbose and not really pythonic.
> 

## Objective

The goal of this project is to develop a REST API that allows users to **index** and **query** their documents within a Vector Database.

 A Vector Database specializes in storing and indexing vector embeddings, enabling fast retrieval and similarity searches. This capability is crucial for applications involving natural language processing, recommendation systems, and many more…

The REST API should be containerized in a Docker container.

### Definitions

To ensure a clear understanding, let's define some key concepts:

1. Chunk: A chunk is a piece of text with an associated embedding and metadata.
2. Document: A document is made out of multiple chunks, it also contains metadata.
3. Library: A library is made out of a list of documents and can also contain other metadata.

The API should:

1. Allow the users to create, read, update, and delete libraries.
2. Allow the users to create, read, update and delete chunks within a library.
3. Index the contents of a library.
4. Do **k-Nearest Neighbor vector search** over the selected library with a given embedding query.

### How to run
1. Git clone this repo
2. Open terminal and navigare /app/Dockerfile
3. Shell command (build the docker iamge): docker build -t vector-takehome .
4. Shell command (run the container): docker run -p 8000:8000 vector-takehome
5. Using postman the following endpoints are available ...

POST: /create_library
GET: /libraries
POST: /update_library
POST: /create_chunk
POST: /update_chunk
POST: /delete_chunk
POST: /index_hash
POST: /index_tree
POST: /search_hash
POST: /search_tree


### Solution

The indexing algorithms i choose are Hashmap (brute force), KD-Tree and finally VP Tree

