from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

db = Chroma(
    persist_directory="./db/chroma_db",
    embedding_function=embeddings
)

query = input("Ask a question: ")

docs = db.similarity_search(
    query,
    k=1
)

context = "\n\n".join(
    [doc.page_content for doc in docs]
)

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash"
)

prompt = f"""
Answer the question using ONLY the provided context.

Context:
{context}

Question:
{query}
"""

print("Retrieved Context:")
print(context)
print("-" * 50)

response = llm.invoke(prompt)

print(response.content)