from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

db = Chroma(
    persist_directory="./db/chroma_db",
    embedding_function=embeddings
)

query = "What is PostgreSQL?"

docs = db.similarity_search(
    query,
    k=1
)

for i, doc in enumerate(docs, start=1):
    print(f"\nResult {i}")
    print(doc.page_content)