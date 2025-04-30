__import__("pysqlite3")
import sys

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
import streamlit as st
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.messages import ModelMessage
from pydantic_ai import Agent, ModelRetry, RunContext
from rich import print
import os
from dotenv import load_dotenv
import asyncio
import nest_asyncio
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dataclasses import dataclass
import random
import string
import openai
import pysqlite3 as sqlite3
import json
from datetime import datetime
import requests
import logging
from pathlib import Path

# Set up logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    filename=log_dir / "visitors.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_visitor_info():
    """Get visitor information including IP, user agent, and other metadata."""
    try:
        # Get IP address
        ip = requests.get("https://api.ipify.org").text

        # Get user agent
        user_agent = st.session_state.get("user_agent", "Unknown")

        # Get current time
        visit_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Get session ID
        session_id = st.session_state.get("session_id", "Unknown")

        return {
            "ip": ip,
            "user_agent": user_agent,
            "visit_time": visit_time,
            "session_id": session_id,
        }
    except Exception as e:
        logging.error(f"Error getting visitor info: {e}")
        return None


def log_visitor():
    """Log visitor information."""
    visitor_info = get_visitor_info()
    if visitor_info:
        logging.info(json.dumps(visitor_info))


# Initialize session state for visitor tracking
if "session_id" not in st.session_state:
    st.session_state.session_id = "".join(
        random.choices(string.ascii_letters + string.digits, k=16)
    )
if "user_agent" not in st.session_state:
    st.session_state.user_agent = st.experimental_get_query_params().get(
        "user_agent", ["Unknown"]
    )[0]

# Log visitor on each page load
log_visitor()

# Apply nest_asyncio first to handle event loops
nest_asyncio.apply()

# At the top of your Streamlit app
st.set_page_config(page_title="Wine Agent", page_icon="🍷", layout="centered")


# Initialize RAG components
@st.cache_resource
def initialize_rag():
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    # Create directories with proper permissions
    base_dir = os.path.expanduser("~/Documents/data/projects/ai/my-projects/agent-wine")
    vector_store_dir = os.path.join(base_dir, "data/vector_store")
    pdfs_dir = os.path.join(base_dir, "data/pdfs")
    texts_dir = os.path.join(base_dir, "data/texts")

    # Create directories if they don't exist
    for dir_path in [vector_store_dir, pdfs_dir, texts_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, mode=0o777, exist_ok=True)

    # Ensure Chroma database file has proper permissions
    db_path = os.path.join(vector_store_dir, "chroma.sqlite3")
    if os.path.exists(db_path):
        try:
            os.chmod(db_path, 0o666)  # rw-rw-rw-
        except Exception as e:
            print(f"Warning: Could not set database permissions: {e}")

    return embeddings, text_splitter


def process_pdfs():
    embeddings, text_splitter = initialize_rag()
    pdf_folder = "data/pdfs"

    documents = []
    for pdf_file in os.listdir(pdf_folder):
        if pdf_file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(pdf_folder, pdf_file))
            pages = loader.load_and_split(text_splitter)
            documents.extend(pages)

    if documents:
        Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory="data/vector_store",
        )
    return bool(documents)


def process_text_files():
    embeddings, text_splitter = initialize_rag()
    text_folder = "data/texts"

    documents = []
    for text_file in os.listdir(text_folder):
        if text_file.endswith(".txt"):
            loader = TextLoader(os.path.join(text_folder, text_file))
            pages = loader.load_and_split(text_splitter)
            documents.extend(pages)

    if documents:
        Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory="data/vector_store",
        )
    return bool(documents)


def process_website(url: str):
    embeddings, text_splitter = initialize_rag()

    # Load website content
    loader = WebBaseLoader(url)
    documents = loader.load_and_split(text_splitter)

    if documents:
        Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory="data/vector_store",
        )
    return bool(documents)


def reset_vector_store():
    """Reset the vector store by properly cleaning up the Chroma instance and its files."""
    import shutil

    base_dir = os.path.expanduser("~/Documents/data/projects/ai/my-projects/agent-wine")
    vector_store_dir = os.path.join(base_dir, "data/vector_store")

    # First, try to properly delete the Chroma collection
    try:
        embeddings, _ = initialize_rag()
        # Create a new Chroma instance with a temporary name
        db = Chroma(
            persist_directory=vector_store_dir,
            embedding_function=embeddings,
            collection_name="temp_reset",
        )
        # Delete the temporary collection
        db._client.delete_collection("temp_reset")
    except Exception as e:
        print(f"Error during Chroma cleanup: {e}")

    # Then remove the directory and recreate it
    if os.path.exists(vector_store_dir):
        shutil.rmtree(vector_store_dir)
        os.makedirs(vector_store_dir, mode=0o777)

        # Create an empty Chroma instance to initialize the database
        try:
            embeddings, _ = initialize_rag()
            db = Chroma(
                persist_directory=vector_store_dir,
                embedding_function=embeddings,
                collection_name="default",
            )
            # Ensure the database file has proper permissions
            db_path = os.path.join(vector_store_dir, "chroma.sqlite3")
            if os.path.exists(db_path):
                os.chmod(db_path, 0o666)
        except Exception as e:
            print(f"Error initializing new Chroma instance: {e}")

    st.session_state.rag_initialized = False
    return True


# Initialize RAG system on app start
if "rag_initialized" not in st.session_state:
    st.session_state.rag_initialized = process_pdfs() or process_text_files()

# Add this after page config
st.markdown(
    """
<style>
    .stButton>button {
        background-color: #f0f2f6;
        color: #2c3e50;
        border: 1px solid #ced4da;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #e2e6ea;
        border-color: #adb5bd;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Model selection and new chat button
col1, col2 = st.columns([3, 1])
with col1:
    if st.button(
        "🔄 New Chat", help="Start a new conversation", use_container_width=True
    ):
        st.session_state.messages = []
        st.session_state.message_history = []
        reset_vector_store()
        st.rerun()

# Add PDF, text file, and website upload section to sidebar
with st.sidebar:
    st.header("Provide your domain knowledge to enhance the agent")

    # PDF Upload
    uploaded_pdfs = st.file_uploader(
        "Upload PDF documents", type=["pdf"], accept_multiple_files=True
    )

    if uploaded_pdfs:
        for uploaded_file in uploaded_pdfs:
            file_path = os.path.join("data/pdfs", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.session_state.rag_initialized = process_pdfs()
        st.success(f"Processed {len(uploaded_pdfs)} new PDF documents!")

    # Text File Upload
    uploaded_texts = st.file_uploader(
        "Upload Text documents", type=["txt"], accept_multiple_files=True
    )

    if uploaded_texts:
        for uploaded_file in uploaded_texts:
            file_path = os.path.join("data/texts", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.session_state.rag_initialized = process_text_files()
        st.success(f"Processed {len(uploaded_texts)} new text documents!")

    # Website URL Input
    website_url = st.text_input(
        "Add Website URL", placeholder="https://www.majestic.co.uk/wine"
    )
    if website_url:
        if st.button("Process Website"):
            with st.spinner("Processing website content..."):
                success = process_website(website_url)
                if success:
                    st.success("Website content processed successfully!")
                else:
                    st.error("Failed to process website content.")
    st.caption(
        "You can enhance the chatbot with up-to-date info, like https://www.majestic.co.uk/wine"
    )
    st.markdown("---")

    # Add visitor logs section
    st.header("Visitor Logs")
    if st.button("View Visitor Logs", help="Show recent visitor information"):
        log_file = log_dir / "visitors.log"
        if log_file.exists():
            with open(log_file, "r") as f:
                logs = f.readlines()[-10:]  # Show last 10 entries
                for log in reversed(logs):
                    try:
                        log_data = json.loads(log.split(" - ")[1])
                        st.json(log_data)
                    except:
                        st.text(log)
        else:
            st.info("No visitor logs available yet.")


# Modify the Agent creation to include RAG context
def get_rag_context(query: str, k: int = 3) -> str:
    if not st.session_state.rag_initialized:
        return ""

    embeddings, _ = initialize_rag()
    db = Chroma(persist_directory="data/vector_store", embedding_function=embeddings)

    docs = db.similarity_search(query, k=k)
    return "\n\n".join([d.page_content for d in docs])


# Load environment variables
load_dotenv()

MODEL_CHOICE = os.getenv("MODEL_CHOICE")
# Validate API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY is not set in the environment or .env file.")
    st.stop()

openai.api_key = OPENAI_API_KEY
model = OpenAIModel("gpt-4o-mini")


# Response structure
class AIResponse(BaseModel):
    content: str
    category: str = "general"


# Initialize session states
if "message_history" not in st.session_state:
    st.session_state.message_history = []

if "messages" not in st.session_state:
    st.session_state.messages = []


# Create agent
@dataclass
class WineDeps:
    wine_name: str | None
    wine_region: str | None
    wine_variety: str | None
    wine_price: float | None
    wine_vintage: int | None
    wine_rating: float | None
    wine_body: str | None  # Light, Medium, Full
    wine_acidity: str | None  # Low, Medium, High
    wine_tannin: str | None  # Soft, Medium, Firm
    wine_alcohol: float | None  # Alcohol percentage
    wine_sweetness: str | None  # Dry, Off-dry, Sweet
    wine_producer: str | None
    wine_notes: list[str] | None  # Tasting notes/flavors


system_prompt = """
~~ CONTEXT: ~~

You are an AI agent named `AgentWine` designed to assist users in discovering and pairing wines with food. 
You have access to extensive wine-related documentation and the ability to fetch detailed content 
from a database of wine and food pairing resources.

You can also retrieve information about specific wines, their characteristics, and recommend pairings based 
on the user's preferences, food type, or occasion. Your main function is to guide the user through discovering 
new wines, understanding their profiles, and offering the best food pairings.

~~ GOAL: ~~

Your job is to help users discover wines and suggest appropriate food pairings based on their queries.
You should use the documentation available and, when necessary, fetch information from Supabase to answer the user's questions.

When the user asks about a specific wine or wine category, you will first check for relevant documentation, 
then provide detailed information about that wine, including its tasting notes, profile, and suggested pairings.

You are also capable of retrieving detailed documentation and suggesting wines based on parameters like flavor, 
region, or occasion.

~~ STRUCTURE: ~~

When you help a user discover a wine or wine pairing, return the information in an organized manner. 
This may include:
- A brief overview of the wine's profile (taste, aroma, body, region, etc.)
- Recommended food pairings
- Any relevant wine articles, guides, or documentation

The system prompts and tools related to wine discovery and pairing can be found in the relevant files.

Please ensure that the information is helpful and presented clearly to the user, allowing them to make an informed decision 
about the wine they should try or pair with their meal.

~~ INSTRUCTIONS: ~~

- Always respond with relevant, clear, and actionable information.
- Fetch details from Supabase or other tools when needed. When doing so, explain the query and the result if necessary.
- If no wine information is found, guide the user to refine their query or suggest other ways they might ask for a wine recommendation.
- The user may ask for specific wines, pairings, or general wine-related information, and you should be able to assist them with that.
- In case of a complex query, break down the answer into clear, structured steps. 
- Never forget that your role is to educate and guide the user, providing them with the right knowledge to make their own wine choices.
"""

agent = Agent(
    model=model,
    result_type=AIResponse,
    system_prompt=system_prompt,
    deps_type=WineDeps,
    retries=3,
)


@agent.tool
async def pick_wine(ctx: RunContext[WineDeps], query: str) -> float:
    """Suggest the wine based on a search query."""
    observation_relevant_chunks = get_rag_context(query, 3)
    observation_text = f"Observation: {observation_relevant_chunks}"
    conversation = []
    conversation.append({"role": "user", "content": f"{query}"})
    conversation.append({"role": "Agent Wine", "content": observation_text})
    return conversation


# Model-specific avatars
MODEL_AVATARS = {
    "OpenAI": "🍷",  # Robot emoji
}

st.image("images/agent-wine.png", width=128)
st.title("Agent Wine - Suggest the perfect wine for your taste")

st.caption(f"Currently using: {MODEL_CHOICE} {MODEL_AVATARS[MODEL_CHOICE]}")

# Update the chat history display
for message in st.session_state.messages:
    avatar = (
        MODEL_AVATARS[message.get("model")] if message["role"] == "assistant" else None
    )
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Handle user input
# Example chat input placeholder
if not st.session_state.messages:
    st.chat_message("assistant", avatar=MODEL_AVATARS[MODEL_CHOICE]).markdown(
         "👋 Hi! This demo shows how the data that you provide via RAG which can enhance the reply by LLM.\n\n"
         "You can ask me questions like:\n\n"
         '"Please pick the top 1 wine in 2024 for me"\n\n'
         "I'll help you find the perfect wine recommendation!"
    )

if prompt := st.chat_input("How can I help you today?"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        with st.spinner("Generating response..."):
            # Event loop management
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Execute async operation
            result = loop.run_until_complete(
                agent.run(
                    f"{prompt}",
                    message_history=st.session_state.message_history,
                )
            )

            # Update message history
            new_messages = result.new_messages()
            if MODEL_CHOICE == "OpenAI":
                for msg in new_messages:
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tool_call in msg.tool_calls:
                            if len(tool_call.id) > 40:
                                tool_call.id = tool_call.id[:40]

            st.session_state.message_history.extend(new_messages)

            print("\n[bold]Message History:[/bold]")
            for i, msg in enumerate(st.session_state["message_history"]):
                print(f"\n[yellow]--- Message {i+1} ---[/yellow]")
                print(msg)

        # Display assistant response
        with st.chat_message("assistant", avatar=MODEL_AVATARS[MODEL_CHOICE]):
            st.markdown(result.data.content)

        # Add to chat history
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result.data.content,
                "model": MODEL_CHOICE,  # Store model info with the message
            }
        )
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        st.session_state.messages.append(
            {"role": "assistant", "content": f"Error: {str(e)}"}
        )
