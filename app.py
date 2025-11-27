import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv


load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

st.set_page_config(page_title="Medical QA Assistant", )

@st.cache_resource
def load_qa_chain():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    vectorstore = FAISS.load_local(
        "medical_faiss_index", 
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",  
    google_api_key=api_key,
    temperature=0.3
)
    
    template = """You are a medical assistant. Answer based on the context provided.
    
Context: {context}

Question: {question}

Answer:"""
    
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    
    return qa_chain

# UI
st.title("🏥 Medical QA Assistant")
st.write("Ask questions about medical conditions, treatments, and procedures")

qa_chain = load_qa_chain()

query = st.text_input("Enter your medical question:")

if st.button("Get Answer") and query:
    with st.spinner("Searching medical records..."):
        result = qa_chain({"query": query})
        
        st.subheader("Answer:")
        st.write(result['result'])
        
        with st.expander("View Sources"):
            for i, doc in enumerate(result['source_documents']):
                st.write(f"**Source {i+1}:**")
                st.write(doc.page_content[:300] + "...")
                st.write(f"Specialty: {doc.metadata.get('specialty', 'N/A')}")
                st.divider()
