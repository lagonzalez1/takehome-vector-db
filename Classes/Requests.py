from Classes.Library import Library, Document, Chunk, ChunkMetadata, LibraryMetadata
from uuid import UUID, uuid4
from typing import List, Optional
from pydantic import BaseModel, Field


class CreateChunkRequest(BaseModel):
    library_id: Optional[UUID] = None
    document_id: Optional[UUID] = None
    text: str
    placement: str

class DeleteChunkRequest(BaseModel):
    library_id: Optional[UUID] = None
    document_id: Optional[UUID] = None
    chunk_id: Optional[UUID] = None


class UpdateChunkRequest(BaseModel):
    library_id: Optional[UUID] = None
    document_id: Optional[UUID] = None
    chunk_id: Optional[UUID] = None
    chunk: Optional[Chunk] = None
    placement: str


