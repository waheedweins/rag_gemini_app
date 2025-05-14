import os
import streamlit as st
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import AIMessage, HumanMessage

# Set Google API Key and Qdrant credentials
os.environ['GOOGLE_API_KEY'] = st.secrets["GOOGLE_API_KEY"]
QDRANT_URL = st.secrets["QDRANT_URL"]
QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
COLLECTION_NAME = "my_documents"

# Init LLM and Embedding
llm = GoogleGenerativeAI(model='gemini-1.5-flash', temperature=0.7)
embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')

# Load retriever
from qdrant_client import QdrantClient

# Create a Qdrant client instance
qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

# Create QdrantVectorStore
qdrant = QdrantVectorStore(
    client=qdrant_client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
)
retriever = qdrant.as_retriever(search_kwargs={"k": 3})

# Setup RAG chain
qa_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever)

# Streamlit UI
st.title("🧠 Gemini RAG Chatbot with Qdrant")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_query = st.chat_input("Ask your question...")
if user_query:
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    response = qa_chain.invoke({
        "question": user_query,
        "chat_history": st.session_state.chat_history
    })
    ai_message = AIMessage(content=response["answer"])
    st.session_state.chat_history.append(ai_message)

# Display conversation
for msg in st.session_state.chat_history:
    role = "🧑 You" if isinstance(msg, HumanMessage) else "🤖 AI"
    st.markdown(f"**{role}:** {msg.content}")
