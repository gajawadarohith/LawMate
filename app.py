import os
import io
import pytesseract
from PIL import Image
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import google.generativeai as genai
import streamlit as st

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Functions to process PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to process image files
def get_image_text(image_files):
    text = ""
    for image_file in image_files:
        image = Image.open(image_file)
        text += pytesseract.image_to_string(image)
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=50000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def create_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

def ingest_data(dataset_folder=None):
    if dataset_folder:
        raw_text = ""
        for root, dirs, files in os.walk(dataset_folder):
            for file in files:
                file_path = os.path.join(root, file)
                if file.endswith('.pdf'):
                    pdf_text = get_pdf_text([open(file_path, 'rb')])
                    raw_text += pdf_text
                elif file.endswith('.png') or file.endswith('.jpg') or file.endswith('.jpeg'):
                    image_text = get_image_text([file_path])
                    raw_text += image_text
        
        text_chunks = get_text_chunks(raw_text)
        vector_store = create_vector_store(text_chunks)
        st.session_state.vector_store = vector_store
        st.success("Files processed successfully!")
    else:
        st.warning("Please provide a valid dataset folder path.")

def get_conversational_chain():
    prompt_template = """
    You are LawMate, a highly experienced attorney providing legal advice based on Indian laws. 
    You will respond to the user's queries by leveraging your legal expertise and the provided information.
    Provide the Section Number for every legal advice.
    Provide Sequential Proceedings for Legal Procedures if to be provided.
    Remember you are an Attorney, so don't provide any other answers that are not related to Law or Legality.
    Context: {context}
    Chat History: {chat_history}
    Question: {question}
    Answer:
    """
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        temperature=0.3,
        system_instruction="You are LawMate, a highly experienced attorney providing legal advice based on Indian laws. You will respond to the user's queries by leveraging your legal expertise and the Context Provided.")
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "chat_history", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, chat_history):
    if "vector_store" not in st.session_state:
        st.warning("Please upload and process files first.")
        return None

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = st.session_state.vector_store
    docs = vector_store.similarity_search(user_question)
    qa_chain = get_conversational_chain()
    response = qa_chain({"input_documents": docs, "chat_history": chat_history, "question": user_question}, return_only_outputs=True)["output_text"]
    return response

def main():
    st.set_page_config("LawMate", page_icon=":scales:")
    st.header("LawMate :scales:")

    st.sidebar.header("Process Dataset")
    dataset_folder = st.sidebar.text_input("Enter dataset folder path", "")

    if st.sidebar.button("Process Dataset"):
        ingest_data(dataset_folder)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi, I'm LawMate, an AI Legal Advisor."}]

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Get user input
    user_question = st.chat_input("Type your legal question here:")

    if user_question:
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.write(user_question)

        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    chat_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])
                    response = user_input(user_question, chat_history)
                    if response is not None:
                        st.write(response)

            if response is not None:
                message = {"role": "assistant", "content": response}
                st.session_state.messages.append(message)

if __name__ == "__main__":
    main()
