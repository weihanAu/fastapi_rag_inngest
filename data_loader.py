from openai import OpenAI
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv
from llama_index.core.schema import Document
import voyageai


load_dotenv()
# client = OpenAI(
#     base_url="https://api.deepseek.com",
#     api_key=os.getenv("DEEPSEEK_API_KEY")
# )
EMBED_MODEL = "voyage-4-lite"
EMBED_DIM = 512

client = voyageai.Client()

splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)

def load_and_chunk_pdf(path:str)->list[str]:
    docs:list[Document] = PDFReader().load_data(file=path)
    texts:list[str] = [d.text for d in docs if getattr(d, 'text', None)]
    chunks:list[str] = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    return chunks

def embed_texts(texts:list[str])->list[list[float]]: 
    response = client.embed(texts,model=EMBED_MODEL,input_type="document")
    # embeddings = [e.embeddings for e in response]
    return response.embeddings