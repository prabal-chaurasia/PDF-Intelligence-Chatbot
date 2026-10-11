from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st

load_dotenv()

st.set_page_config(page_title="📃 PDF Intelligence Chatbot", layout="wide")
st.header("📃 PDF Intelligence Chatbot")

with st.sidebar:
    uploaded_file = st.file_uploader(
        "Upload your PDF",
        type=["pdf"]
    )

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()


if "messages" not in st.session_state:
    st.session_state.messages = []

if uploaded_file:
    # Save uploaded PDF temporarily
    import tempfile

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as temp_file:
        temp_file.write(uploaded_file.getvalue())
        pdf_path = temp_file.name

    # Load PDF using its file path
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    print(docs)
else:
    st.info("Please upload a PDF to continue.")


splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=0)
embedding_engine = OpenAIEmbeddings()
chunks = splitter.split_documents(docs)

vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_engine,
    persist_directory="./db"
)

retriever = vectordb.as_retriever(search_kwargs={"k": 4})

llm = ChatOpenAI(model="gpt-4o-mini")

with st.form(key="question_form", clear_on_submit=True):
    question = st.text_area("Ask Your Question")
    submit = st.form_submit_button("Search")

if question and submit:
    retrieved_docs = retriever.invoke(question)

    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    prompt = ChatPromptTemplate.from_template(
        """
            Answer the questions only on the context provided below:
            {context}

            Question: {question}
        """
    )
    formatted_prompt = prompt.format(
        context= context,
        question= question
    )

    response = llm.invoke(formatted_prompt)
    st.session_state.messages.append({
        "question": question,
        "answer": response.content
    })

for message in st.session_state.messages:
    st.write("### 👤 You")
    st.write(message["question"])

    st.write("### 🤖 GPT")
    st.write(message["answer"])