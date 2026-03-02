import pydantic
class RAGChunkAndSrc(pydantic.BaseModel):
    """A chunk of text and its source information."""
    chunks: list[str]
    source_id: str = None

class RAGUpsertResult(pydantic.BaseModel):
    """Result of an upsert operation, including the number of items upserted."""
    ingested_count: int
    
class RAGSerchResult(pydantic.BaseModel):
    """Result of a search operation, including the retrieved chunks and their source information."""
    contexts: list[str]
    sources: list[str]
    
class RAGQueryResult(pydantic.BaseModel):
    """Result of a query operation, including the answer and the sources used to generate it."""
    answer: str
    sources: list[str]
    context_count: int