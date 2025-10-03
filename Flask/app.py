# import .....
from flask import Flask, request, jsonify
import os 
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

# Initialize the flask App
app = Flask(__name__)

# load the API key
load_dotenv()
os.getenv("GOOGLE_API_KEY")

# Initialise the RAG
def initialize_rag():
    # load the pdf
    file_path = 'data/NASA_blog.pdf'
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    #split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )
    texts = text_splitter.split_documents(documents)
    # create embeddings and vector store
    embeddings = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(texts, embeddings)
    # create a RAG chain
    llm = ChatGoogleGenerativeAI(
        model = "gemini-2.5-flash",
        temperature = 0
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = "stuff",
        retriever = vectorstore.as_retriever()
    )
    return qa_chain
# call the RAG function
qa_chain = initialize_rag()

# define routes for prediction
@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    question = data['question']
    answer = qa_chain.invoke(question)
    print(answer)
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
    