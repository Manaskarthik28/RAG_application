from flask import Flask, request, jsonify, render_template 
import os 
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

# Initialize Flask app
app = Flask(__name__)

# Load environment variables 
load_dotenv()

def initialize_rag():
    # 1. Load Document
    file_path = 'data/NASA_blog.pdf'
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # 2. Split Text into Chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )
    texts = text_splitter.split_documents(documents)
    
    # 3. Create Embeddings and Vector Store
    embeddings = HuggingFaceEmbeddings(model_name = 'All-MiniLM-L6-v2')
    vectorstore = Chroma.from_documents(texts, embeddings)
    
    # 4. Initialize the LLM (Gemini)
    llm = ChatGoogleGenerativeAI(
        model = 'gemini-2.5-flash',
        temperature = 0 
    )
    
    # 5. Create the RetrievalQA Chain
    rag_chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = 'stuff', 
        retriever = vectorstore.as_retriever() 
    )
    return rag_chain
# Initialize the RAG chain globally
rag_chain = initialize_rag()

@app.route('/', methods=["POST",'GET'])
def home():
    question = None
    answer = None
    if request.method == 'POST':
        question = request.form.get('question')
        rag_result = rag_chain.invoke(question)
        answer  = rag_result['result']
    return render_template('index.html', user_question = question, rag_answer = answer)
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)

