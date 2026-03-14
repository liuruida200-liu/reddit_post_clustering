import os
import shutil
import sqlite3

import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

from htmlTemplates import css, bot_template, user_template


DB_PATH = "chatbot_lab9.db"
VECTORSTORE_DIR = "faiss_index"


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is not available. This app is configured to require GPU and will not fall back to CPU."
        )


def init_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS pdf_documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT NOT NULL,
            full_text TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS text_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL,
            chunk_index INTEGER NOT NULL,
            chunk_text TEXT NOT NULL,
            FOREIGN KEY (document_id) REFERENCES pdf_documents(id)
        )
    """)

    conn.commit()
    conn.close()


def reset_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM text_chunks")
    cursor.execute("DELETE FROM pdf_documents")
    conn.commit()
    conn.close()


def insert_document(file_name, full_text):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO pdf_documents (file_name, full_text) VALUES (?, ?)",
        (file_name, full_text)
    )
    document_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return document_id


def insert_chunks(document_id, chunks):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    for idx, chunk in enumerate(chunks):
        cursor.execute(
            "INSERT INTO text_chunks (document_id, chunk_index, chunk_text) VALUES (?, ?, ?)",
            (document_id, idx, chunk)
        )

    conn.commit()
    conn.close()


def extract_text_from_pdf(pdf_file):
    text = ""
    pdf_file.seek(0)
    pdf_reader = PdfReader(pdf_file)

    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text


def get_pdf_text(pdf_docs):
    text = ""

    if not pdf_docs:
        return text

    for pdf in pdf_docs:
        pdf_text = extract_text_from_pdf(pdf)
        if pdf_text:
            text += pdf_text + "\n"

    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def process_pdfs_and_store(pdf_docs):
    all_chunks = []
    all_metadatas = []

    reset_database()

    for pdf in pdf_docs:
        full_text = extract_text_from_pdf(pdf)

        if full_text.strip():
            document_id = insert_document(pdf.name, full_text)
            chunks = get_text_chunks(full_text)
            insert_chunks(document_id, chunks)

            for idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadatas.append({
                    "document_id": document_id,
                    "file_name": pdf.name,
                    "chunk_index": idx
                })

    return all_chunks, all_metadatas


def get_vectorstore(text_chunks, metadatas):
    require_cuda()

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": False}
    )

    if os.path.exists(VECTORSTORE_DIR):
        shutil.rmtree(VECTORSTORE_DIR)

    vectorstore = FAISS.from_texts(
        texts=text_chunks,
        embedding=embeddings,
        metadatas=metadatas
    )

    vectorstore.save_local(VECTORSTORE_DIR)
    return vectorstore


def load_vectorstore():
    require_cuda()

    if not os.path.exists(VECTORSTORE_DIR):
        return None

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": False}
    )

    vectorstore = FAISS.load_local(
        VECTORSTORE_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vectorstore


@st.cache_resource
def load_local_llm():
    require_cuda()

    model_id = "Qwen/Qwen2.5-7B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    first_param_device = str(next(model.parameters()).device)

    if not first_param_device.startswith("cuda"):
        raise RuntimeError(
            f"Model was not loaded onto GPU. Current device: {first_param_device}"
        )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=64,
        do_sample=False,
        repetition_penalty=1.05,
        return_full_text=False,
        pad_token_id=tokenizer.eos_token_id
    )

    llm = HuggingFacePipeline(pipeline=pipe)
    return llm


def get_conversation_chain(vectorstore):
    llm = load_local_llm()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    custom_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
Use only the context below to answer the question.
If the answer is not directly stated in the context, say: Answer not found in uploaded document.

Context:
{context}

Question:
{question}

Answer:
"""
    )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        ),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": custom_prompt},
        return_source_documents=False,
        output_key="answer"
    )

    return conversation_chain


def clean_response(answer):
    if not answer:
        return "Answer not found in uploaded document."

    answer = answer.strip().replace("\n", " ")

    weird_patterns = [
        "Tonia",
        "partes de",
        "año",
        "Restarting computer",
        "part-of-speonia",
        "Define the year of the year",
        "licpart",
        "part 1:",
        "part 2:",
        "part 3:",
        "port 2:"
    ]

    for pattern in weird_patterns:
        if pattern.lower() in answer.lower():
            return "Answer not found in uploaded document."

    if len(answer) > 400:
        answer = answer[:400].strip()

    if not answer:
        return "Answer not found in uploaded document."

    return answer


def handle_userinput(user_question):
    if st.session_state.conversation is None:
        st.warning("Please upload and process your PDF first!")
        return

    response = st.session_state.conversation({"question": user_question})

    if "answer" in response:
        response["answer"] = clean_response(response["answer"])

    st.session_state.chat_history = response["chat_history"]

    if st.session_state.chat_history:
        st.session_state.chat_history[-1].content = response["answer"]

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(
                user_template.replace("{{MSG}}", message.content),
                unsafe_allow_html=True
            )
        else:
            st.write(
                bot_template.replace("{{MSG}}", message.content),
                unsafe_allow_html=True
            )


def main():
    load_dotenv()
    init_database()

    st.set_page_config(page_title="Chat with PDFs", page_icon="🤖")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with PDFs 🤖")

    user_question = st.text_input("Ask questions about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")

        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'",
            accept_multiple_files=True,
            type=["pdf"]
        )

        if st.button("Clear Chat"):
            st.session_state.chat_history = None
            if os.path.exists(VECTORSTORE_DIR):
                try:
                    vectorstore = load_vectorstore()
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                except Exception:
                    st.session_state.conversation = None
            else:
                st.session_state.conversation = None

        if st.button("Process"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF file.")
            else:
                with st.spinner("Processing"):
                    try:
                        text_chunks, metadatas = process_pdfs_and_store(pdf_docs)

                        if not text_chunks:
                            st.warning("No readable text was found in the uploaded PDFs.")
                        else:
                            vectorstore = get_vectorstore(text_chunks, metadatas)
                            st.session_state.conversation = get_conversation_chain(vectorstore)
                            st.session_state.chat_history = None
                            st.success("Processing complete.")
                    except Exception as e:
                        st.error(f"Error while processing documents: {e}")


if __name__ == "__main__":
    main()