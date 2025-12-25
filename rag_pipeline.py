#rag_pipeline.py


#importing all necessary libraries

!pip install langchain-google-genai langchain langchain-community faiss-cpu pypdf
!pip install langchain-classic
!pip install -U langchain-huggingface
!pip install google-generativeai 
!pip install langchain-community sentence-transformers
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
import getpass


#loading pdf
from langchain_community.document_loaders import PyPDFLoader
pdf_path = r"C:\Users\LENOVO\Downloads\HealthCareSectorinindia-AnOverview.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()

#adding API key
import os
os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your API key:")
print("API key set succesfully")

#split into chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 150
)
chunks = text_splitter.split_documents(documents)

#creating embeddings
!pip install hf_xet
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

#create vector store
from langchain_community.vectorstores import FAISS
vectorstore = FAISS.from_documents(chunks,embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k":3})

#build RAG chain
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.chains.conversational_retrieval.base import ConversationalRetrievalChain

llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    temperature=0.2
)
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages = True
)

#building RAG Chain
qa= ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory
)

#main execution
while True:
    query = input("Ask:")
    if query.lower() == "exit":
        break
    result = qa.invoke({"question":query})
    print(result["answer"])

