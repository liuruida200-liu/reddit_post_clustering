import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

from htmlTemplates import css, bot_template, user_template


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is not available. This app is configured to require GPU and will not fall back to CPU."
        )


def get_pdf_text(pdf_docs):
    text = ""

    if not pdf_docs:
        return text

    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=600,
        chunk_overlap=120,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    require_cuda()

    st.write("Embedding device: cuda")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": False}
    )

    vectorstore = FAISS.from_texts(
        texts=text_chunks,
        embedding=embeddings
    )

    return vectorstore


@st.cache_resource
def load_local_llm():
    require_cuda()

    model_id = "lmsys/vicuna-7b-v1.3"

    st.write("CUDA available:", torch.cuda.is_available())
    st.write("CUDA version:", torch.version.cuda)
    st.write("GPU count:", torch.cuda.device_count())
    st.write("GPU name:", torch.cuda.get_device_name(0))

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=False,
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map=0,
        trust_remote_code=True
    )

    first_param_device = str(next(model.parameters()).device)
    hf_device_map = getattr(model, "hf_device_map", None)

    st.write("HF device map:", hf_device_map)
    st.write("First parameter device:", first_param_device)

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
        temperature=0.0,
        repetition_penalty=1.1,
        return_full_text=False,
        pad_token_id=tokenizer.eos_token_id
    )

    llm = HuggingFacePipeline(pipeline=pipe)
    return llm


def build_prompt(context, question):
    prompt = f"""
USER: You are a document question answering assistant.

Use ONLY the provided context to answer the question.

Strict rules:
- Answer only from the context.
- Do not use outside knowledge.
- Do not guess.
- Do not translate into another language.
- Do not complete broken text fragments.
- If the exact answer is not directly stated in the context, reply exactly:
Answer not found in uploaded document.

Answer in 1 to 3 short sentences or bullet points only.

Context:
{context}

Question:
{question}

ASSISTANT:
"""
    return prompt


def clean_response(answer):
    if not answer:
        return "Answer not found in uploaded document."

    answer = answer.strip()

    weird_patterns = [
        "Tonia",
        "partes de",
        "año",
        "Restarting computer",
        "part-of-speonia",
        "Define the year of the year",
        "licpart"
    ]

    for pattern in weird_patterns:
        if pattern.lower() in answer.lower():
            return "Answer not found in uploaded document."

    if len(answer) > 300:
        answer = answer[:300].strip()

    return answer


def handle_userinput(user_question):
    if st.session_state.vectorstore is None:
        st.warning("Please upload and process your PDF first!")
        return

    docs = st.session_state.vectorstore.similarity_search(user_question, k=3)

    context_parts = []
    for doc in docs:
        if doc.page_content:
            context_parts.append(doc.page_content)

    context = "\n\n".join(context_parts).strip()

    if not context:
        answer = "Answer not found in uploaded document."
    else:
        llm = load_local_llm()
        prompt = build_prompt(context, user_question)
        raw_answer = llm.invoke(prompt)
        answer = clean_response(raw_answer)

    st.session_state.chat_history.append(("user", user_question))
    st.session_state.chat_history.append(("bot", answer))

    for role, message in st.session_state.chat_history:
        if role == "user":
            st.write(
                user_template.replace("{{MSG}}", message),
                unsafe_allow_html=True
            )
        else:
            st.write(
                bot_template.replace("{{MSG}}", message),
                unsafe_allow_html=True
            )

    with st.expander("Retrieved Chunks"):
        for i, doc in enumerate(docs):
            st.write(f"Chunk {i + 1}:")
            st.write(doc.page_content)


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with PDFs", page_icon="🤖")
    st.write(css, unsafe_allow_html=True)

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("Chat with PDFs 🤖")

    col1, col2 = st.columns([4, 1])

    with col1:
        user_question = st.text_input("Ask questions about your documents:")

    with col2:
        if st.button("Clear Chat"):
            st.session_state.chat_history = []

    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")

        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'",
            accept_multiple_files=True,
            type=["pdf"]
        )

        if st.button("Process"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF file.")
            else:
                with st.spinner("Processing"):
                    raw_text = get_pdf_text(pdf_docs)

                    if not raw_text.strip():
                        st.warning("No readable text was found in the uploaded PDFs.")
                    else:
                        try:
                            text_chunks = get_text_chunks(raw_text)
                            vectorstore = get_vectorstore(text_chunks)
                            st.session_state.vectorstore = vectorstore
                            st.session_state.chat_history = []
                            st.success("Processing complete.")
                        except Exception as e:
                            st.error(f"Error while processing documents: {e}")


if __name__ == "__main__":
    main()