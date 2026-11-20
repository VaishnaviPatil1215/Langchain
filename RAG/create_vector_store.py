import os

from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain_chroma import Chroma

load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))

documents_dir = os.path.join(
    current_dir,
    "documents"
)

persist_directory = os.path.join(
    current_dir,
    "db",
    "chroma_db"
)

all_docs = []

# Load all text files
for file_name in os.listdir(documents_dir):

    if file_name.endswith(".txt"):

        file_path = os.path.join(
            documents_dir,
            file_name
        )

        loader = TextLoader(file_path)

        docs = loader.load()

        all_docs.extend(docs)

print(f"Loaded {len(all_docs)} documents")

# Split documents
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(all_docs)

print(f"Created {len(chunks)} chunks")

# Gemini Embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

# Create Chroma DB
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=persist_directory
)

print("Vector Store Created Successfully")