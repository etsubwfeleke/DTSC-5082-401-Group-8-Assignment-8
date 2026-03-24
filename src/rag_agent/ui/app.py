"""
app.py
======
Streamlit user interface for the Deep Learning RAG Interview Prep Agent.

Three-panel layout:
  - Left sidebar: Document ingestion and corpus browser
  - Centre: Document viewer
  - Right: Chat interface

API contract with the backend (agree this with Pipeline Engineer
before building anything):

  ingest(file_paths: list[Path]) -> IngestionResult
  list_documents() -> list[dict]
  get_document_chunks(source: str) -> list[DocumentChunk]
  chat(query: str, history: list[dict], filters: dict) -> AgentResponse

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

from pathlib import Path
import tempfile

import streamlit as st

from rag_agent.agent.graph import get_compiled_graph
from rag_agent.agent.state import AgentResponse
from rag_agent.config import get_settings
from rag_agent.corpus.chunker import DocumentChunker
from rag_agent.vectorstore.store import VectorStoreManager


# ---------------------------------------------------------------------------
# Cached Resources
# ---------------------------------------------------------------------------
# Use st.cache_resource for objects that should persist across reruns
# and be shared across all user sessions. This prevents re-initialising
# ChromaDB and reloading the embedding model on every button click.


@st.cache_resource
def get_vector_store() -> VectorStoreManager:
    """
    Return the singleton VectorStoreManager.

    Cached so ChromaDB connection is initialised once per application
    session, not on every Streamlit rerun.
    """
    return VectorStoreManager()


@st.cache_resource
def get_chunker() -> DocumentChunker:
    """Return the singleton DocumentChunker."""
    return DocumentChunker()


@st.cache_resource
def get_graph():
    """Return the compiled LangGraph agent."""
    return get_compiled_graph()


# ---------------------------------------------------------------------------
# Session State Initialisation
# ---------------------------------------------------------------------------


def initialise_session_state() -> None:
    """
    Initialise all st.session_state keys on first run.

    Must be called at the top of main() before any UI is rendered.
    Without this, state keys referenced in callbacks will raise KeyError.

    Interview talking point: Streamlit reruns the entire script on every
    user interaction. session_state is the mechanism for persisting data
    (chat history, ingestion results) across reruns.
    """
    defaults = {
        "chat_history": [],           # list of {"role": "user"|"assistant", "content": str}
        "ingested_documents": [],     # list of dicts from list_documents()
        "selected_document": None,    # source filename currently in viewer
        "last_ingestion_result": None,
        "thread_id": "default-session",  # LangGraph conversation thread
        "topic_filter": None,
        "difficulty_filter": None,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


# ---------------------------------------------------------------------------
# Ingestion Panel (Sidebar)
# ---------------------------------------------------------------------------
def render_ingestion_panel(
    store: VectorStoreManager, 
    chunker: DocumentChunker
    ) -> None:
    """
    Render the document ingestion panel in the sidebar.

    Allows multi-file upload of PDF and Markdown files. Displays
    ingestion results (chunks added, duplicates skipped, errors).
    Updates the ingested documents list after successful ingestion.

    Parameters
    ----------
    store : VectorStoreManager
    chunker : DocumentChunker
    """
    st.sidebar.header("📂 Corpus Ingestion")

    # 1. File Uploader
    uploaded_files = st.sidebar.file_uploader(
        "Upload study materials",
        type=["pdf", "md"],
        accept_multiple_files=True
    )

    # 2 & 3. Ingest Logic
    if st.sidebar.button("🚀 Ingest Documents", disabled=not uploaded_files):
        all_chunks = []
        with st.sidebar.status("Processing...", expanded=True) as status:
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = Path(tmp_file.name)
                chunks = chunker.chunk_file(
                    tmp_path,
                    metadata_overrides={"source": uploaded_file.name},
                )
                all_chunks.extend(chunks)

            if all_chunks:
                result = store.ingest(all_chunks)
                st.sidebar.success(f"Added {result.ingested} chunks!")
                st.session_state.ingested_documents = store.get_collection_stats()["sources"]
                status.update(label="Done!", state="complete")

    # 4. Document List
    st.sidebar.markdown("---")
    stats = store.get_collection_stats()
    for source in stats["sources"]:
        c1, c2 = st.sidebar.columns([4, 1])
        c1.caption(source)
        if c2.button("🗑️", key=source):
            store.delete_document(source)
            st.rerun()

    st.sidebar.info("Upload .pdf or .md files to populate the corpus.")

def render_corpus_stats(store: VectorStoreManager) -> None:
    """
    Render a compact corpus health summary in the sidebar.
    Shows total chunks, topics covered, and whether bonus topics
    are present. Used during Hour 3 to demonstrate corpus completeness.

    Parameters
    ----------
    store : VectorStoreManager
    """
    st.sidebar.markdown("---") # Adds a visual divider line
    st.sidebar.subheader("📊 Corpus Health")

    # 1. Get stats from the backend
    stats = store.get_collection_stats()

    # 2. Display the chunk count
    st.sidebar.metric("Total Chunks", stats["total_chunks"])

    # 3. Display the topics covered
    topics_list = stats.get("topics", [])
    if topics_list:
        st.sidebar.write(f"**Topics:** {', '.join(topics_list)}")
    else:
        st.sidebar.write("**Topics:** None ingested")

    # 4. Bonus topic indicator
    if stats.get("bonus_topics_present"):
        st.sidebar.success("✅ Bonus topics available")
    else:
        st.sidebar.info("💡 Add GAN or SOM for bonus points")

# ---------------------------------------------------------------------------
# Document Viewer Panel (Centre)
# ---------------------------------------------------------------------------

def render_document_viewer(store: VectorStoreManager) -> None:
    """
    Render the document viewer in the main centre column.

    Displays a selectable list of ingested documents. When a document
    is selected, renders its chunk content in a scrollable pane.

    Parameters
    ----------
    store : VectorStoreManager
    """
    st.subheader("📄 Document Viewer")

    # 1. Fetch current sources
    stats = store.get_collection_stats()
    docs = stats.get("sources", [])

    if not docs:
        st.info("Ingest documents using the sidebar to view content here.")
        return

    # 2. Selection Menu
    selected_source = st.selectbox("Select document", options=docs)

    # 3 & 4. Display Chunks in a scrollable container
    with st.container(height=600, border=True):
        # Retrieve chunks for this specific document
        results = store._collection.get(
            where={"source": selected_source},
            include=["documents", "metadatas"]
        )
        
        if results["documents"]:
            for i in range(len(results["documents"])):
                topic = results["metadatas"][i].get("topic", "N/A")
                diff = results["metadatas"][i].get("difficulty", "N/A")
                
                # We use a header style for each chunk
                st.markdown(f"**Chunk {i+1}** :blue[[{topic} | {diff}]]")
                st.text_area(
                    label=f"Content_{i}", 
                    value=results["documents"][i], 
                    height=150, 
                    disabled=True,
                    label_visibility="collapsed"
                )
                st.divider()
        else:
            st.warning("No chunks found for this document.")

    # 5. Coverage Summary
    st.caption(f"Viewing {len(results['documents'])} chunks from {selected_source}")

# ---------------------------------------------------------------------------
# Chat Interface Panel (Right)
# ---------------------------------------------------------------------------

def render_chat_interface(graph) -> None:
    """
    Render the chat interface in the right column.

    Supports multi-turn conversation with the LangGraph agent using streaming.
    Displays source citations and hallucination guards.
    """
    from langchain_core.messages import HumanMessage
    import streamlit as st

    st.subheader("💬 Interview Prep Chat")

    # 1. Topic & Difficulty Filters
    # These are passed to the graph to narrow the vector search scope
    col_topic, col_diff = st.columns(2)
    with col_topic:
        topic = st.selectbox(
            "Focus Topic", 
            ["All", "ANN", "CNN", "RNN", "LSTM", "Seq2Seq", "Autoencoder", "GAN", "SOM"],
            help="Filter retrieval to a specific deep learning architecture."
        )
    with col_diff:
        difficulty = st.selectbox(
            "Difficulty Level", 
            ["All", "beginner", "intermediate", "advanced"],
            help="Match the question complexity to your current level."
        )

    # 2. Chat history display
    # We render this inside a scrollable container
    chat_container = st.container(height=500)
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Render Citations if they exist
                if message.get("sources"):
                    with st.expander("📎 View Source Citations"):
                        for source in message["sources"]:
                            st.caption(source)
                
                # Render Hallucination Guard Warning
                if message.get("no_context_found"):
                    st.warning("⚠️ This response was generated without direct corpus context.")

    # 3. Chat Input & Agent Logic
    if query := st.chat_input("Ask about a deep learning topic..."):
        
        # a. Append user message to chat_history
        st.session_state.chat_history.append({"role": "user", "content": query})
        
        # b. Display user message immediately 
        # (This happens automatically on the next rerun, but we proceed to generate the AI response now)

        # c. Build LangGraph input
        inputs = {
            "messages": [HumanMessage(content=query)],
            "topic_filter": None if topic == "All" else topic,
            "difficulty_filter": None if difficulty == "All" else difficulty
        }
        
        # d. Configuration for conversation memory (thread_id)
        config = {"configurable": {"thread_id": st.session_state.thread_id}}

        # --- STRETCH GOAL: STREAMING AI RESPONSE ---
        with st.chat_message("assistant"):
            response_placeholder = st.empty()  # Container for the "typing" effect
            full_response_text = ""
            final_response_obj = None

            # e. Execute graph.stream to get the "Wow Factor"
            with st.spinner("Searching corpus...:)"):
                for event in graph.stream(inputs, config=config, stream_mode="values"):
                    # 2f. Extract the final_response from the graph state
                    if "final_response" in event and event["final_response"]:
                        final_response_obj = event["final_response"]
                        full_response_text = final_response_obj.answer
                        
                        # Update the UI in real-time with a cursor
                        response_placeholder.markdown(full_response_text + "▌")
            
            # Finalize the markdown without the cursor
            response_placeholder.markdown(full_response_text)

            # g. Append assistant message with metadata to session history
            if final_response_obj:
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": full_response_text,
                    "sources": final_response_obj.sources,
                    "no_context_found": final_response_obj.no_context_found
                })
        
        # Force a rerun to sync the chat container with the new history
        st.rerun()

# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------


def main() -> None:
    """
    Application entry point.

    Sets page config, initialises session state, instantiates shared
    resources, and renders all UI panels.

    Run with: uv run streamlit run src/rag_agent/ui/app.py
    """
    settings = get_settings()

    st.set_page_config(
        page_title=settings.app_title,
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title(f"🧠 {settings.app_title}")
    st.caption(
        "RAG-powered interview preparation — built with LangChain, LangGraph, and ChromaDB"
    )

    initialise_session_state()

    # Instantiate shared backend resources
    store = get_vector_store()
    chunker = get_chunker()
    graph = get_graph()

    # Sidebar
    render_ingestion_panel(store, chunker)
    render_corpus_stats(store)

    # Main content area — two columns
    viewer_col, chat_col = st.columns([1, 1], gap="large")

    with viewer_col:
        render_document_viewer(store)

    with chat_col:
        render_chat_interface(graph)


if __name__ == "__main__":
    main()
