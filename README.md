# PDF Intelligence Chatbot

A RAG (Retrieval-Augmented Generation) app that lets you upload a PDF and ask questions about its content. Built with Streamlit, LangChain, OpenAI and ChromaDB.

## Features
- Upload any PDF and chat with it
- Answers are based on the document's content
- Uses vector search to find the most relevant parts of the PDF
- Simple web interface built with Streamlit

## How It Works
1. The PDF is loaded and split into chunks
2. Each chunk is converted into embeddings using OpenAI
3. The embeddings are stored in a Chroma vector database
4. When you ask a question, the most relevant chunks are retrieved
5. The retrieved context is sent to the LLM (gpt-4o-mini) to generate the answer

## Tech Stack
Python, Streamlit, LangChain, OpenAI API (gpt-4o-mini and OpenAI embeddings), ChromaDB, PyPDFLoader

## Setup
1. Clone the repo
   git clone https://github.com/prabal-chaurasia/PDF-Intelligence-Chatbot.git
   cd PDF-Intelligence-Chatbot
2. Install dependencies
   pip install -r requirements.txt
3. Create a `.env` file and add your key
   OPENAI_API_KEY=your_key_here
4. Run the app
   streamlit run App_V1.py
