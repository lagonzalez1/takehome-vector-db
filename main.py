from fastapi import FastAPI, HTTPException
from Classes.Driver import Driver
from Classes.Library import Library, Document, Chunk, LibraryMetadata, DocumentMetadata, ChunkMetadata
from Classes.Requests import CreateChunkRequest, DeleteChunkRequest, UpdateChunkRequest, SearchHashRequest, SearchTreeRequest, IndexTreeRequest, UpdateLibraryRequest, IndexHashRequest
from uuid import UUID, uuid4
from typing import List, Optional
import numpy as np
# Create FastAPI app
app = FastAPI()

# Define Driver Class
driver = Driver()

chunk_texts = [
    "The sky is blue.",
    "I enjoy long walks in the park.",
    "Programming in Python is fun.",
    "I am feeling a bit sad",
    "I do not enjoy debugging",
    "Dictator",
    "Hitler",
    "Coffee tastes best when fresh.",
    "Reading a good book relaxes me."
]

meta_chunk = ChunkMetadata()
chunks = [Chunk(text=txt,              
                embedding=driver.generate_chunk_embedding(txt),   
                metadata=meta_chunk)
          for txt in chunk_texts]

doc = Document(
    id="6150990f-5e62-4996-82ee-84395b2e673b",
    chunks=chunks,
    metadata=DocumentMetadata()
)

library = Library(
    id="069b276d-2b12-4e37-9107-473daf96f54d",
    documents=[doc],
    metadata=LibraryMetadata(
        name="Example Library",
        description="One doc, five chunks"
    )
)

driver.create_library(library)


# Create a library with default id, documents and metadata
@app.post("/create_library")
def create_library(library: Library):
    try:
        driver.create_library(library)
        return {"message": f"Library created with ID: {library.id}"}
    except KeyError:
        raise HTTPException(status_code=404, detail="create_library error")


# Read all documents
@app.get("/libraries")
def get_libraries():
    try:
        return driver.read_libraries()
    except KeyError:
        raise HTTPException(status_code=404, detail="get_libraries error")


# Read entire library
@app.get("/libraries/{library_id}")
def get_library(library_id: UUID):
    try:
        library = driver.read_library(library_id)
        return library
    except KeyError:
        raise HTTPException(status_code=404, detail="libraries error")


# Update the library
@app.post("/update_library")
def update_library(request: UpdateLibraryRequest):
    try:
        update = driver.update_library(request.library_id, request.library)
        return {"message": f"update library request {update}"}
    except KeyError:
        raise HTTPException(status_code=404, detail="update_library error")
    return {"message": "Update library"}


# Add chunk to document
@app.post("/create_chunk")
def create_chunk(request: CreateChunkRequest):
    try:   
        create = driver.create_chunk(request.library_id,request.document_id, request.text, request.placement)
        return {"message": f"create chunk request: {create}"}
    except KeyError:
        raise HTTPException(status_code=404, detail="create_chunk error found") 

# Update chunk to document
@app.post("/update_chunk")
def update_chunk(request: UpdateChunkRequest):
    try:   
        update = driver.update_chunk(request.library_id,request.document_id, request.chunk_id, request.chunk)
        return {"message": f"update chunk request: {update}"}
    except KeyError:
        raise HTTPException(status_code=404, detail="update_chunk error found") 

# Delete chunk to document
@app.post("/delete_chunk")
def delete_chunk(request: DeleteChunkRequest):
    try:   
        delete = driver.delete_chunk(request.library_id, request.document_id, request.chunk_id)
        return {"message": f"delete chunk request: {delete}"}
    except KeyError:
        raise HTTPException(status_code=404, detail="delete_chunk error found") 


# Index documents using traditional hash
@app.post("/index_hash")
def index_hash(request: IndexHashRequest):
    try:   
        driver.index_library(request.library_id)
        return {"message": f"index hash"}
    except KeyError:
        raise HTTPException(status_code=404, detail="index_hash error found") 


# Index documents using tree hash
@app.post("/index_tree")
def index_tree(request: IndexTreeRequest):
    try:   
        driver.index_library_tree(request.library_id)
        return {"message": f"index result"}
    except KeyError:
        raise HTTPException(status_code=404, detail="index_tree error found") 


# Index documents using traditional hash
@app.post("/search_hash")
def search_hash(request: SearchHashRequest):
    try:   
        result = driver.search_hash(request.library_id, request.text, request.k)

        return {"message": f"index hash result: {result}"}
    except KeyError:
        raise HTTPException(status_code=404, detail="search_hash error found") 


# Index documents using traditional hash
@app.post("/search_tree")
def search_hash(request: SearchTreeRequest):
    try:   
        result = driver.search_tree(request.library_id, request.text, request.k)
        return {"message": f"search tree result {result}"}
    except KeyError:
        raise HTTPException(status_code=404, detail="search_tree error found") 


# Index documents using traditional hash
@app.post("/search_vptree")
def search_hash(request: SearchTreeRequest):
    try:   
        result = driver.search_vptree(request.library_id, request.text, request.k)
        return {"message": f"search vptree result {result}"}
    except KeyError:
        raise HTTPException(status_code=404, detail="search_tree error found") 


