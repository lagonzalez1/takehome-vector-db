from Classes.Library import Library, Document, Chunk, ChunkMetadata, DocumentMetadata
from Classes.Tree import TreeNode
from datetime import datetime
from typing import List, Any, Optional
from uuid import UUID, uuid4
from typing import Dict, List, Tuple
from numpy.linalg import norm
import os
import numpy as np
import cohere
from dotenv import load_dotenv

# Load env
load_dotenv()

# Cohere api for embedding 
KEY = os.getenv("COHERE_API")
co = cohere.ClientV2(KEY)

class Driver:
    def __init__(self):
        self.library_db: Dict[UUID, Library] = {}
        # {UID: {'embeddings:' 
        # List[List[Float], 'chunk_ids': List[UID], 'chunk_map': Dict[UID, Chunk] ]}}
        self.library_index: Dict[UUID, Dict[str,Any]] = {}
    
    # Params: id UUID
    def read_library(self, id: UUID) -> Library:
        return self.library_db[id]

    # Params: None
    def read_libraries(self) -> List[Library]:
        return list(self.library_db.values())
    
    # Params: library Class
    def create_library(self, library: Library) -> None:
        self.library_db[library.id] = library
    
    # Params: id UUID 
    def delete_library(self, id: UUID) -> bool:
        if id not in self.library_db:
            return False
        del self.library_db[id]
        return True

    # Params: id UUID
    # Params: library Class
    def update_library(self, id: UUID, library: Library) -> None:
        if id not in self.library_db:
            return
        self.library_db[id] = library
    

    def read_chunk(self, id: UUID, doc_id: UUID) -> List[Chunk]:
        if id not in self.library_db:
            return []
        library = self.library_db[id]
        for doc in library.documents:
            if doc.id == doc_id:
                return doc.chunks
            
        return []
    
    def generate_chunk_embedding(self, text: str) -> List[float]:
        try:
            response = co.embed(texts=[text], input_type="search_document", model="embed-english-v3.0", embedding_types=["float"])
            return response.embeddings.float[0]
        except cohere.CohereAPIError as e:
            print(f"An error occurred: {e}")
            return []
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return []

    
    def delete_chunk(self, id: UUID, doc_id: UUID, chunk_id: UUID) -> bool:
        if id not in self.library_db:
            return False
        library = self.library_db[id]
        for doc in library.documents:
            if doc.id == doc_id:
                for i, pre_chunk in enumerate(doc.chunks):
                    if pre_chunk.id == chunk_id:
                        del doc.chunks[i]
                        return True
        return False

    def create_chunk(self, id: UUID, doc_id: Optional[UUID], text: str, placement: str) -> bool:
        # No libary found
        if id not in self.library_db:
            return False
        library = self.library_db[id]
        # If empty documents, create a single and append
        if len(library.documents) == 0:
            embedding = self.generate_chunk_embedding(text)
            chunk = Chunk(text=text, embedding=embedding, metadata=ChunkMetadata())
            document = Document(chunks=[chunk], metadata=DocumentMetadata())
            library.documents.append(document)
            return True
        # Search and place chunk
        for doc in library.documents:
            if doc.id == doc_id:
                embedding = self.generate_chunk_embedding(text)
                chunk = Chunk(text=text, embedding=embedding, metadata=ChunkMetadata())
                if placement == "start":
                    doc.chunks.insert(0, chunk)
                    return True
                else:
                    doc.chunks.append(chunk)
                    return True
                
        return False
        # index_library(id)         

    def update_chunk(self, id: UUID, doc_id: UUID, chunk_id: UUID, chunk: Chunk) -> bool:
        if id not in self.library_db:
            return False
        library = self.library_db[id] 
        for doc in library.documents:
            if doc.id == doc_id:
                for i, pre_chunk in enumerate(doc.chunks):
                    if pre_chunk.id == chunk_id:
                        chunk.embedding = self.generate_chunk_embedding(chunk.text)
                        doc.chunks[i] = chunk
                        print(f"Chunk {chunk_id} updated in document {doc_id} of library {id}")
                        return True

        # index_library(id)
        return False


    def read_indexes(self, library_id: UUID) -> Dict[str,Any]:
        return list(self.library_index[library_id].values())

    # Calculate cosine similarity between two vectors
    def cos_sim(a: List[float], b: List[float]) -> float:
        anp, bnp = np.array(a), np.array(b)
        dot = np.dot(anp, bnp)
        magnitude_a,magnitude_b = norm(anp), norm(bnp)
        return dot / (magnitude_a * magnitude_b)

    def search_hash(self, library_id: UUID, text: str ,k: int) -> List[Chunk]:
        if library_id not in self.library_db:
            return []
        documents = self.library_db[library_id].documents
        if len(documents) == 0:
            return []
        results = []
        search_embedding = self.generate_chunk_embedding(text)
        for doc in documents:
            for chunk in doc.chunks:
                if chunk.embedding:
                    similarity = self.cos_sim(search_embedding, chunk.embedding)
                    results.append((chunk, similarity))
        
        # Sort on similarity reverse for values closest to 1
        results.sort(key=lambda x : x[1], reverse=True)
        return results[:k]
    
    
    def index_library(self, library_id: UUID) -> None:
        library = self.library_db[library_id]
        embeddings = []
        chunk_ids = []
        chunk_map = {}
        for doc in library.documents:
            for chunk in doc.chunks:
                if chunk.embedding is not None:
                    embeddings.append(chunk.embedding)
                    chunk_ids.append(chunk.id)
                    chunk_map[chunk.id] = chunk
        
        self.library_index[library_id] = {
            "embeddings": embeddings,
            "chunk_ids": chunk_ids,
            "chunk_map": chunk_map
        }
        print(self.library_index[library_id])    

    def index_library_tree(self, library_id: UUID) -> None:
        if library_id not in library_id:
            return
        library = self.library_db[library_id]
        # Build a tuple of (chunk_id, [0,0,0])
        chunk_points = [(chunk.id, chunk.embedding) for doc in library.documents for chunk in doc.chunks if chunk.embedding]
        print(chunk_points)
        # Build tree    
            

    def build_tree(self, points: List[Tuple[List[float], UUID]], depth: int = 0) -> Optional[TreeNode]:
        if not points:
            return None
        # Get the first of first array dimentions
        k = len(points[0][axis])
        axis = depth % k
        points.sort(key=lambda x: x[0][axis])
        median = len(points) // 2
        # Recursively create tree nodes
        return TreeNode (
            point=points[median][0],
            chunk_id=points[median][1],   
            left=self.build_tree(points=points[:median], depth= + 1),
            right=self.build_tree(points=points[median+1:], depth= + 1),
            axis=axis    
        )
