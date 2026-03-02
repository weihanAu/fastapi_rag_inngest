import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api
from inngest.experimental.ai import openai
from dotenv import load_dotenv
import uuid
import os
import datetime

from openai import BaseModel
from data_loader import load_and_chunk_pdf, embed_texts
from custom_types.custom_types import RAGChunkAndSrc, RAGUpsertResult, RAGSerchResult
from vector_db import QdrantStorage

"# load dotenv"
load_dotenv()

inngest_client = inngest.Inngest(
    app_id="rag-app",
    logger=logging.getLogger("inngest"),
    is_production=False,
    serializer=inngest.PydanticSerializer()
)

@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf")
)
async def rag_ingest_pdf(ctx:inngest.Context):
    def _load(ctx:inngest.Context) -> RAGChunkAndSrc:
     pdf_path = ctx.event.data.get("pdf_path")
     source_id = ctx.event.data.get("source_id", pdf_path)  # Use pdf_path as source_id if not provided
     chunks = load_and_chunk_pdf(pdf_path)
     return RAGChunkAndSrc(chunks=chunks, source_id=source_id)
 
    def _upsert(chunks_and_src:RAGChunkAndSrc) -> RAGUpsertResult:
     chunks = chunks_and_src.chunks
     source_id = chunks_and_src.source_id
     vectors = embed_texts(chunks) 
     ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}-{i}")) for i in range(len(chunks))]
     payloads = [{"source": source_id,"text":chunks[i]} for i in range(len(chunks))]
     QdrantStorage().upsert(ids, vectors, payloads)
     return RAGUpsertResult(ingested_count=len(chunks))
     
     
    chunks_and_src = await ctx.step.run("load-and-chunk", lambda: _load(ctx),output_type=RAGChunkAndSrc)
    ingested = await ctx.step.run("embed-and-upsert", lambda: _upsert(chunks_and_src), output_type=RAGUpsertResult)
    return ingested.model_dump()
  
@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai")
)

async def rag_query_pdf_ai(ctx:inngest.Context):
    def _search(question:str,tok_k:str = 5)-> RAGSerchResult:
        query_vector = embed_texts([question])[0]
        store = QdrantStorage()
        results = store.search(query_vector, top_k=tok_k)
        return RAGSerchResult(contexts=results["contexts"], sources=results["sources"])
    
    question = ctx.event.data.get("question")
    top_k = int(ctx.event.data.get("top_k", 5))
    
    found = await ctx.step.run("embed-and-search", lambda: _search(question, top_k), output_type=RAGSerchResult)
    
    context_block = "\n\n".join( f" -{c}" for c in found.contexts)
    user_content = (
        "You are an assistant for answering questions based on the following retrieved contexts:\n\n"
        f"\n\n{context_block}\n"
        "answer concisely using context above"
    )
    deep_seek_client = openai.Adapter(
        base_url="https://api.deepseek.com",
        auth_key=os.getenv("DEEPSEEK_API_KEY"),
        model="deepseek-chat"
    )    
    res = await ctx.step.ai.infer(
        "llm-answer",
        adapter=deep_seek_client,
        body={
            "max_tokens": 1024,
            "temperature": 0.2,
            "messages": [
                {
                    "role": "system",
                    "content": "you answer the question only based on the context provided, if you don't know the answer based on the context, say you don't know"
                },
                {
                    "role": "user",
                    "content": user_content
                }
            ]
        }
    )
    answer = res.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
    return {"answer": answer, "sources": found.sources}

app = FastAPI()

class Message(BaseModel):
    message: str
    
@app.post("/query")
async def health(message:Message):
   
    await inngest_client.send(
        inngest.Event(
         name="rag/query_pdf_ai",
         data={"question": message.message, "top_k": 5}
        )
    )
    return {"status": "received"}

inngest.fast_api.serve(app, inngest_client,[rag_ingest_pdf, rag_query_pdf_ai])