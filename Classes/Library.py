from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Optional
from uuid import UUID, uuid4



class ChunkMetadata(BaseModel):
    created_at: datetime = Field(default_factory=datetime.utcnow)

class Chunk(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    text: str
    embedding: Optional[List[float]] = None
    metadata: ChunkMetadata

class DocumentMetadata(BaseModel):
    created_at: datetime = Field(default_factory=datetime.utcnow)

class Document(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    chunks: List[Chunk]
    metadata: DocumentMetadata

class LibraryMetadata(BaseModel):
    name: str
    description: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

class Library(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    documents: List[Document]
    metadata: LibraryMetadata
