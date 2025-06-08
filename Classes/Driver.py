from Classes.Library import Library, Document, Chunk, ChunkMetadata, DocumentMetadata
from Classes.Tree import TreeNode
from datetime import datetime
from typing import List, Any, Optional
from uuid import UUID, uuid4
from typing import Dict, List, Tuple
from numpy.linalg import norm
import heapq
import os
import numpy as np
import cohere
from dotenv import load_dotenv
import random
import math
# Load env
load_dotenv()

# Cohere api for embedding 
KEY = os.getenv("COHERE_API")
co = cohere.ClientV2(KEY)

class Driver:
    def __init__(self):
        self.library_db: Dict[UUID, Library] = {}
        # Brute force approach
        self.library_index: Dict[UUID, Dict[str,Any]] = {}
        # KD - tree approach
        self.library_tree: Dict[UUID, TreeNode] = {}

    def read_library(self, id: UUID) -> Library:
        if id not in self.library_db[id]:
            return None
        return self.library_db[id]

    def read_libraries(self) -> List[Library]:
        return list(self.library_db.values())
    
    def create_library(self, library: Library) -> None:
        self.library_db[library.id] = library

    def delete_library(self, id: UUID) -> bool:
        if id not in self.library_db:
            return False
        del self.library_db[id]
        del self.library_tree[id]
        return True

    def update_library(self, id: UUID, library: Library) -> None:
        if id not in self.library_db:
            return
        self.library_db[id] = library
        self.index_library(id)
        self.index_library_tree(id)
    
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
        embedding = self.delete_chunk_hashmap(id, doc_id, chunk_id)
        if embedding is not None or embedding is not []:
            self.delete_chunk_library(id, doc_id, chunk_id)
            self.delete_chunk_tree(id, embedding)
            return True         
        return False

    def delete_chunk_library(self, id : UUID, doc_id:  UUID, chunk_id: UUID) -> None:
        library = self.library_db[id]
        for doc in library.documents:
            if doc.id == doc_id:
                for i, pre_chunk in enumerate(doc.chunks):
                    if pre_chunk.id == chunk_id:
                        del doc.chunks[i]
                        return

    def delete_chunk_hashmap(self,  id: UUID, doc_id: UUID, chunk_id: UUID) -> List[float]:
        if id not in self.library_index:
            return
        data = self.library_index[id]
        chunk_ids = data["chunk_data"]
        embeddings = data["embeddings"]
        chunk_map = data["chunk_map"]

        remove_idx = None
        embedding = List[float]
        for i, cid in enumerate(chunk_ids):
            if cid == chunk_id:
                remove_idx = i
                break
        if remove_idx is not None and embeddings[remove_idx]:
            embedding = embeddings[remove_idx]
        if remove_idx is not None:
            del embeddings[remove_idx]
            del chunk_id[remove_idx]
            del chunk_map[chunk_id]
        
        return embedding
        # Updating the library_db will cause indexing to receive the update
        # Improvements will be to directly modify the Data structure

    def delete_chunk_tree(self, id: UUID, embedding: List[float]) -> bool:
        if embedding is None:
            return 
        tree_node = self.library_tree[id]
        new_node = self.delete_node_tree(tree_node, embedding, depth=0)
        self.library_tree[id] = new_node

    def find_min(self, root: TreeNode, dim: int) -> Optional[TreeNode]:
        if root is None:
            return root
        axis = root.axis
        if axis == dim:
            # If left node is None then this must be the last value
            if root.left is None:
                return root
            return self.find_min(root.left, dim)
        
        left_min = self.find_min(root.left, dim)
        right_min = self.find_min(root.right, dim)

        c = [n for n in (root, left_min, right_min) if n is not None]
        return min(c, key=lambda x: x.point[dim])

    def delete_node_tree(self, root: TreeNode, embedding: List[float], depth: int = 0) -> Optional[TreeNode]:
        if root is None:
            return None
        axis = root.axis
        if root.point == embedding:
            if root.right is not None:
                min_node = self.find_min(root.right, axis)
                root.point = min_node.point
                root.chunk_id = min_node.chunk_id
                root.right = self.delete_node_tree(root.right, min_node.chunk_id, depth+1)
                return root
            if root.left is not None:
                min_node = self.find_min(root.left, axis)
                root.point = min_node.point
                root.chunk_id = min_node.chunk_id
                root.left = self.delete_node_tree(root.left, min_node.chunk_id, depth+1)
                return root   
        if embedding[axis] < root.point[axis]:
            root.left = self.delete_node_tree(root.left, embedding, depth+1)
        else:
            root.right = self.delete_node_tree(root.right, embedding, depth+1)
        return root             
        
    def create_chunk(self, id: UUID, doc_id: Optional[UUID], text: str, placement: str) -> bool:
        found = False
        if id not in self.library_db:
            return False
        library = self.library_db[id]
        # If empty documents, create a single and append
        if len(library.documents) == 0:
            embedding = self.generate_chunk_embedding(text)
            chunk = Chunk(text=text, embedding=embedding, metadata=ChunkMetadata())
            document = Document(chunks=[chunk], metadata=DocumentMetadata())
            library.documents.append(document)
            self.index_library(id)
            self.index_library_tree(id)
            found = True
        else:
            # Search and place chunk
            for doc in library.documents:
                if doc.id == doc_id:
                    embedding = self.generate_chunk_embedding(text)
                    chunk = Chunk(text=text, embedding=embedding, metadata=ChunkMetadata())
                    if placement == "start":
                        doc.chunks.insert(0, chunk)
                        found = True
                        break
                    else:
                        doc.chunks.append(chunk)
                        found = True
                        break

        # When a chunk is created update indexs for index structures
        self.index_library(id)
        self.index_library_tree(id)
        return found

    def update_chunk(self, id: UUID, doc_id: UUID, chunk_id: UUID, chunk: Chunk) -> bool:
        if id not in self.library_db:
            return False
        found = False
        library = self.library_db[id] 
        for doc in library.documents:
            if doc.id == doc_id:
                for i, pre_chunk in enumerate(doc.chunks):
                    if pre_chunk.id == chunk_id:
                        chunk.embedding = self.generate_chunk_embedding(chunk.text)
                        doc.chunks[i] = chunk
                        found = True
                        print(f"Chunk {chunk_id} updated in document {doc_id} of library {id}")

        self.index_library(id)
        self.index_library_tree(id)
        return found
 
    def cos_sim(self,a: List[float], b: List[float]) -> float:
        anp, bnp = np.array(a), np.array(b)
        dot = np.dot(anp, bnp)
        magnitude_a, magnitude_b = norm(anp), norm(bnp)
        return dot / (magnitude_a * magnitude_b)
    
    # Index library in a map data structure 
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

    # Index library in B-tree data structure
    def index_library_tree(self, library_id: UUID) -> None:
        if library_id not in self.library_db:
            return
        library = self.library_db[library_id]
        # Build a tuple of (chunk_id, list[float])
        chunk_points = [(chunk.embedding, chunk.id) for doc in library.documents for chunk in doc.chunks if chunk.embedding]
        # Build tree map and set
        self.library_tree[library_id] = self.build_tree(chunk_points)


    def build_tree(self, points: List[Tuple[List[float], UUID] ], depth: int = 0) -> Optional[TreeNode]:
        if not points:
            return None
        # Get the first of first array dimentions
        dimensions = len(points[0][0])
        axis = depth % dimensions
        # Sort based on the vectors 
        points.sort(key=lambda x: x[0][axis])
        # Get the median value for Tree split
        median = len(points) // 2
        # Recursively create tree nodes
        return TreeNode (
            point=points[median][0],
            chunk_id=points[median][1],   
            left=self.build_tree(points=points[:median], depth= depth+1),
            right=self.build_tree(points=points[median+1:], depth= depth+1),
            axis=axis    
        )

    # Search KD tree
    def search_tree(self, library_id: UUID, text: str, k: int) -> List[UUID]:
        if library_id not in self.library_tree:
            return []
        # Cost (E)
        search_embedding = self.generate_chunk_embedding(text)
        max_heap: List[Tuple[float, UUID]] = []
        def traverse_tree(tree: TreeNode, depth=0):
            if tree is None:
                return
            similarity = self.cos_sim(search_embedding, tree.point)
            if len(max_heap) < k:
                # Cost (log K) push pop from heap
                heapq.heappush(max_heap, (-similarity, tree.chunk_id))
            else:
                # K items, compare worst to top.
                if similarity > max_heap[0][0]:
                    heapq.heappushpop(max_heap, (similarity, tree.chunk_id))
            axis = tree.axis
            # To diviate left or right calculate the difference
            diff = np.array(search_embedding)[axis] - np.array(tree.point)[axis]
            close, away = (tree.left, tree.right) if diff < 0 else (tree.right, tree.left)

            traverse_tree(close, depth=depth+1)

            worse_sim = max_heap[0][0] if len(max_heap) == k else -1.0
            # Worse case traverse the other side of tree 
            # This will cause O(N)
            if abs(diff) < (1.0 -  worse_sim):
                traverse_tree(away, depth=depth+1)
        
        root = self.library_tree[library_id]
        traverse_tree(root, depth=0)

        top_k = sorted([(score, chunk_id) for score, chunk_id in max_heap], reverse=True)
        # Since the chunk_ids are stored in a map just directly map the result text
        index = self.library_index[library_id]
        chunk_map = index["chunk_map"]
        return [chunk_map[cid].text for _, cid in top_k]

    # Hash based indexing
    def search_hash(self, library_id: UUID, text: str, k: int) -> List[Chunk]:
        if library_id not in self.library_db:
            return []
        # Index the library documents
        ## self.index_library(library_id=library_id)
        # Mappings
        index = self.library_index[library_id]
        embeddings = index["embeddings"]
        chunk_ids = index["chunk_ids"]
        chunk_map = index["chunk_map"]

        documents = self.library_db[library_id].documents
        if len(documents) == 0:
            return []
        results = []
        # Cost O(E)
        search_embedding = self.generate_chunk_embedding(text)
        # Cost O(N) Runs n times
        for emb, chunk_id in zip(embeddings, chunk_ids):
            # Cost O(D)
            similarity = self.cos_sim(search_embedding, emb)
            # Cost O(log K)
            results.append((chunk_id, similarity))
        # Sort the result and reverse
        results.sort(key=lambda x : x[1], reverse=True)
	    # Combined per-chunk cost: O(D + log k)
        return [chunk_map[chunk_id].text for chunk_id, _ in results[:k]]
