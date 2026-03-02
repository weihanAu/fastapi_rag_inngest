from typing import Annotated
from fastapi import Depends
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams

class QdrantStorage:
    def __init__(self,url="http://localhost:6333",collection="docs",dim=1024,):
            self.client = QdrantClient(url=url,timeout=30)
            self.collection = collection
            if not self.client.collection_exists( self.collection):
                self.client.recreate_collection(
                    collection_name= self.collection,
                    vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
                )
                
    def upsert(self, ids, vectors, payloads) -> None:
        points =[
                PointStruct(
                    id=ids[i], 
                    vector=vectors[i], 
                    payload=payloads[i]
                ) 
                for i in range(len(ids))
        ]
        self.client.upsert(collection_name=self.collection, points=points)
        
    def search(self, query_vector, top_k:int=5) -> dict[str, list]:
        search_result = self.client.query_points(
            collection_name=self.collection,
            with_payload=True,
            query=query_vector,
            limit=top_k
        )
        contexts = []
        sources = set()
        for res in search_result.points:
            payload = getattr(res, 'payload', {})
            text = payload.get('text', '')
            source = payload.get('source', '')
            if text:
                contexts.append(text)
                sources.add(source)
        return {"contexts": contexts, "sources": list(sources)}
    

def get_QdrantStorage() -> QdrantStorage:
    return QdrantStorage()

get_QdrantStorage_dependency = Annotated[QdrantStorage, Depends(get_QdrantStorage)]