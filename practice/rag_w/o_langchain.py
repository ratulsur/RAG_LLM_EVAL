from abc import ABC, abstractmethod
import numpy as np
from dataclasses import dataclass, field
from typing import Sequence
from pydantic import BaseModel, Field
from typing import Annotated, List 

@dataclass
class Chunk(BaseModel):
    text: Annotated[str, Field(..., description = "the raw text to be used")]
    doc_id: Annotated[int, Field(..., description = "the identification index of the ")]
    chunk_index: Annotated[int, Field(...,description = "the chunk index")]
    metadata: Annotated[dict, Field(...,description = "the metadata of the chunks")]
    embedding: np.array

@dataclass
class RetrievedChunk:
    chunk: Chunk
    threshold_value: int

class Chunker(ABC):

    @abstractmethod
    def split(self, text:str, doc_id:str, metadata: dict | None = None) ->list[Chunk]:
        ...

class FixedSizeChunking(Chunker):
    """
    """

    def __init__(self, chunk_size:int = 200, overlap:int = 40) -> None:
        if overlap >= chunk_size:
            raise ValueError("chunk size must be greater than overlap")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, text: str, doc_id:int, metadata: dict | None = None) ->list[Chunk]:
        
        words = [t.strip() for t in text.split() if t.strip()]
        if not words:
            return []
        
        
        
        step = self.chunk_size - self.overlap 
        chunks:list[Chunk] = []

        for start in range (0, len(words), step):
            window = words[start: start+ self.chunk_size]
            if not window:
                break
            chunks.append(
                Chunk(text = " ".join(window),
                      doc_id = doc_id, chunk_id = id, 
                      metadata = dict(metadata or {}))
            )
            idx += 1

            if start + self.chunk_size >= words:
                break
            return chunks
        

class Embedder(ABC):
    @property
    @abstractmethod
    def dim(self) -> int:
        ...
    def embedder(self, text: Sequence[str]):
        ...

class OpenAIEmbedder(Embedder):
    def __init__(self, model: str = "text-embeding-3-small", batch_size: int = 128) -> None:
        from openai import OpenAI
        self._client = OpenAI()
        self._model = model
        self._batch_size = batch_size
        self._dim = 1536

    def dim(self) -> int:
        return self._dim
    
    def embed()


