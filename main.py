from fastapi import FastAPI, HTTPException
from Classes.Driver import Driver
from Classes.Library import Library, Document, Chunk
from Classes.Requests import CreateChunkRequest, DeleteChunkRequest, UpdateChunkRequest
from uuid import UUID, uuid4

# Create FastAPI app
app = FastAPI()

# Define Driver Class
driver = Driver()

# Define a simple GET endpoint
@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI!"}


# Create a library with default id, documents and metadata
@app.post("/create_library")
def create_library(library : Library):
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
def update_library(library_id: uuid4, library: Library):
    # Find the library using id then update the contents ?
    # Find the library using id then update the contents ?
    return {"message": "Update library"}


# Add chunk to document
@app.post("/create_chunk")
def create_chunk(request: CreateChunkRequest):
    try:   
        created = driver.create_chunk(request.library_id,request.document_id, request.text, request.placement)
        return {"message": f"create chunk result {created}"}
    except KeyError:
        raise HTTPException(status_code=404, detail="create_chunk error found") 

# Update chunk to document
@app.post("/update_chunk")
def update_chunk(request: UpdateChunkRequest):
    try:   
        update = driver.update_chunk(request.library_id,request.document_id, request.chunk_id, request.chunk)
        return {"message": f"create chunk result {update}"}
    except KeyError:
        raise HTTPException(status_code=404, detail="create_chunk error found") 



# Delete chunk to document
@app.post("/delete_chunk")
def delete_chunk(request: DeleteChunkRequest):
    try:   
        delete = driver.delete_chunk(request.library_id, request.document_id, request.chunk_id)
        return {"message": f"delete chunk result: {delete}"}
    except KeyError:
        raise HTTPException(status_code=404, detail="create_chunk error found") 


# Index documents using traditional hash
@app.get("/index_hash")
def index_hash(library_id: UUID):
    try:   
        driver.index_library(library_id)
        return {"message": f"index result"}
    except KeyError:
        raise HTTPException(status_code=404, detail="create_chunk error found") 

# Index documents using traditional hash
@app.post("/search_hash")
def index_hash(library_id: UUID, text: str):
    try:   
        driver.search_hash(library_id, text)
        return {"message": f"index result"}
    except KeyError:
        raise HTTPException(status_code=404, detail="create_chunk error found") 




# Index documents using traditional hash
@app.get("/index_tree")
def index_tree(library_id: UUID):
    try:   
        driver.index_library_tree(library_id)
        return {"message": f"index result"}
    except KeyError:
        raise HTTPException(status_code=404, detail="create_chunk error found") 



