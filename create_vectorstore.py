

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import pickle


with open('chunks.pkl', 'rb') as f:
    chunks = pickle.load(f)

print(f"Loaded {len(chunks)} chunks files")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

langchain_docs = [
    Document(page_content=chunk['text'], metadata=chunk['metadata'])
    for chunk in chunks
] 

print("This will take 5-10 minutes on CPU")

vectorstore = FAISS.from_documents(langchain_docs, embeddings)

vectorstore.save_local("medical_faiss_index")
print("Vector store created")