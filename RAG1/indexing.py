from dotenv import load_dotenv
from .file_reader import extract_paragraphs
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

# Updated to the active, supported free-tier model name
embedding = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
vectore_store = InMemoryVectorStore(embedding)

chunks = extract_paragraphs("data/pdfs/Penguins_ACL.pdf")
vectore_store.add_texts(texts=chunks)

print(f"indexed {len(chunks)} chunks")
