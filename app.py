"""
Multi-Modal RAG System - Streamlit Application
Main application file for document processing and Q&A
"""

import streamlit as st
import tempfile
import os
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Page configuration
st.set_page_config(
    page_title="Multi-Modal RAG System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .source-card {
        background-color: #e7f3ff;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'vector_store' not in st.session_state:
        st.session_state['vector_store'] = None
    if 'processed_file' not in st.session_state:
        st.session_state['processed_file'] = None
    if 'num_chunks' not in st.session_state:
        st.session_state['num_chunks'] = 0

def main():
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown('<div class="main-header">🤖 Multi-Modal RAG System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Advanced Document Understanding with Text, Tables, and Images</div>', unsafe_allow_html=True)
    
    # Sidebar for document upload
    with st.sidebar:
        st.header("📄 Document Upload")
        st.markdown("Upload a PDF document to get started")
        
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Upload a PDF document containing text, tables, or images"
        )
        
        if uploaded_file is not None:
            st.success(f"📎 File loaded: {uploaded_file.name}")
            st.info(f"Size: {uploaded_file.size / 1024:.1f} KB")
            
            if st.button("🚀 Process Document", type="primary", use_container_width=True):
                process_document(uploaded_file)
        
        st.markdown("---")
        
        # Settings
        st.header("⚙️ Settings")
        chunk_size = st.slider("Chunk Size", 256, 1024, 512, 128)
        top_k = st.slider("Retrieved Chunks", 3, 10, 5, 1)
        search_type = st.selectbox("Search Type", ["Hybrid Search", "Vector Only"])
        
        # Store in session state
        st.session_state['chunk_size'] = chunk_size
        st.session_state['top_k'] = top_k
        st.session_state['search_type'] = search_type
        
        st.markdown("---")
        
        # Stats
        if st.session_state['vector_store'] is not None:
            st.header("📊 Statistics")
            stats = st.session_state['vector_store'].get_stats()
            st.metric("Total Vectors", stats['total_vectors'])
            st.metric("Documents", stats['total_documents'])
            
            if st.button("🗑️ Clear Database", use_container_width=True):
                st.session_state['vector_store'].clear()
                st.session_state['vector_store'] = None
                st.session_state['processed_file'] = None
                st.session_state['num_chunks'] = 0
                st.rerun()
    
    # Main content area
    if st.session_state['vector_store'] is None:
        # Welcome screen
        show_welcome_screen()
    else:
        # Q&A Interface
        show_qa_interface()

def show_welcome_screen():
    """Display welcome screen with instructions"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📝 Text Processing")
        st.write("Extracts and indexes text content from PDFs with semantic understanding")
        
    with col2:
        st.markdown("### 📊 Table Extraction")
        st.write("Detects and processes tables, making data searchable and queryable")
        
    with col3:
        st.markdown("### 🖼️ Image OCR")
        st.write("Extracts text from images and charts using OCR technology")
    
    st.markdown("---")
    
    st.markdown("### 🎯 How to Use")
    st.markdown("""
    1. **Upload a PDF** using the sidebar
    2. **Process the document** to extract and index all content
    3. **Ask questions** about the document content
    4. **Get answers** with source citations and confidence scores
    """)
    
    st.markdown("---")
    
    st.markdown("### 💡 Sample Questions")
    st.code("""
• "What is the main topic of this document?"
• "Summarize the key findings"
• "What data is shown in the tables?"
• "What are the recommendations?"
    """)
    
    st.info("👈 Start by uploading a PDF document in the sidebar")

def process_document(uploaded_file):
    """Process uploaded document"""
    try:
        # Import required modules
        from ingestion.parser import MultiModalParser
        from ingestion.ocr import OCREngine
        from chunking.semantic_chunker import SemanticChunker
        from embedding.multimodal_vector_store import MultiModalVectorStore
        
    except ImportError as e:
        st.error(f"""
        ❌ Missing required modules. Please ensure all files are in place:
        
        - ingestion/parser.py
        - ingestion/ocr.py
        - chunking/semantic_chunker.py
        - retrieval/vector_store.py
        
        Error: {str(e)}
        """)
        return
    
    with st.spinner("🔄 Processing document... This may take a minute."):
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Parse document
            status_text.text("📖 Parsing document...")
            progress_bar.progress(20)
            parser = MultiModalParser()
            elements = parser.parse_document(tmp_path)
            
            # Step 2: OCR on images
            status_text.text("🔍 Running OCR on images...")
            progress_bar.progress(40)
            ocr_engine = OCREngine()
            for elem in elements:
                if elem.type == 'image':
                    elem.metadata['ocr_text'] = ocr_engine.extract_text_from_image(elem.content)
            
            # Step 3: Chunk content
            status_text.text("✂️ Chunking content...")
            progress_bar.progress(60)
            chunk_size = st.session_state.get('chunk_size', 512)
            chunker = SemanticChunker(chunk_size=chunk_size, overlap=50)
            
            chunks = []
            for elem in elements:
                if elem.type == 'text':
                    chunks.extend(chunker.chunk_text(elem.content, elem.page_num))
                elif elem.type == 'table':
                    chunks.append(chunker.chunk_table(elem.content, elem.page_num))
                elif elem.type == 'image' and 'ocr_text' in elem.metadata:
                    chunks.append({
                        'type': 'image',
                        'content': elem.metadata['ocr_text'],
                        'page': elem.page_num,
                        'ocr_text': elem.metadata['ocr_text']
                    })
            
            # Step 4: Index in vector store
            status_text.text("🗄️ Indexing in vector database...")
            progress_bar.progress(80)
            vector_store = MultiModalVectorStore(collection_name=uploaded_file.name.replace('.pdf', ''))
            vector_store.add_documents(chunks)
            
            progress_bar.progress(100)
            status_text.empty()
            
            # Store in session state
            st.session_state['vector_store'] = vector_store
            st.session_state['processed_file'] = uploaded_file.name
            st.session_state['num_chunks'] = len(chunks)
            
            # Clean up
            os.unlink(tmp_path)
            
            # Success message
            st.markdown(f"""
            <div class="success-box">
                <h3>✅ Document Processed Successfully!</h3>
                <p><strong>File:</strong> {uploaded_file.name}</p>
                <p><strong>Elements extracted:</strong> {len(elements)} (Text: {sum(1 for e in elements if e.type=='text')}, 
                Tables: {sum(1 for e in elements if e.type=='table')}, 
                Images: {sum(1 for e in elements if e.type=='image')})</p>
                <p><strong>Chunks created:</strong> {len(chunks)}</p>
                <p><strong>Vectors indexed:</strong> {vector_store.get_stats()['total_vectors']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.balloons()
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error processing document: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

def show_qa_interface():
    """Display Q&A interface"""
    st.markdown(f"### 💬 Ask Questions About: {st.session_state['processed_file']}")
    
    # Initialize query in session state if not exists
    if 'current_query' not in st.session_state:
        st.session_state['current_query'] = ''
    
    # Query input
    query = st.text_input(
        "Enter your question:",
        value=st.session_state.get('current_query', ''),
        placeholder="e.g., What are the main findings in this document?",
        key="query_input"
    )
    
    col1, col2 = st.columns([3, 1])
    with col1:
        search_button = st.button("🔍 Search", type="primary", use_container_width=True)
    with col2:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state['current_query'] = ''
            st.rerun()
    
    # Sample questions
    st.markdown("**💡 Try these sample questions:**")
    sample_cols = st.columns(3)
    sample_questions = [
        "What is the main topic?",
        "Summarize key findings",
        "What data is in tables?"
    ]
    
    for i, (col, question) in enumerate(zip(sample_cols, sample_questions)):
        if col.button(question, key=f"sample_{i}"):
            st.session_state['current_query'] = question
            st.rerun()
    
    # Process query
    if search_button and query:
        perform_search(query)

def perform_search(query):
    """Perform search and display results"""
    try:
        from generation.qa_engine import QAEngine
    except ImportError:
        # Fallback: basic search without QA engine
        st.warning("QA Engine not available. Showing basic search results.")
        show_basic_search(query)
        return
    
    with st.spinner("🔍 Searching..."):
        try:
            vector_store = st.session_state['vector_store']
            top_k = st.session_state.get('top_k', 5)
            search_type = st.session_state.get('search_type', 'Hybrid Search')
            
            # Perform search
            if search_type == "Hybrid Search":
                results = vector_store.hybrid_search(query, top_k=top_k)
            else:
                results = vector_store.retrieve(query, top_k=top_k)
            
            if not results:
                st.warning("No relevant content found. Try rephrasing your question.")
                return
            
            # Display answer (simplified version without LLM)
            st.markdown("### 📝 Answer")
            st.markdown(f"""
            Based on the retrieved context, here are the most relevant sections for your query: "{query}"
            
            The system found {len(results)} relevant sections across different pages of the document.
            """)
            
            # Display sources
            st.markdown("### 📚 Source Documents")
            
            for i, result in enumerate(results, 1):
                with st.expander(
                    f"Source {i} - Page {result['metadata']['page']} ({result['metadata']['type']}) "
                    f"- Relevance: {result.get('hybrid_score', 1-result.get('distance', 0)):.2%}"
                ):
                    st.markdown(f"**Content:**")
                    st.write(result['content'][:500] + "..." if len(result['content']) > 500 else result['content'])
                    
                    st.markdown(f"**Metadata:**")
                    st.json({
                        'page': result['metadata']['page'],
                        'type': result['metadata']['type'],
                        'source': result['metadata'].get('source', 'unknown'),
                        'relevance_score': f"{result.get('hybrid_score', 1-result.get('distance', 0)):.4f}"
                    })
            
        except Exception as e:
            st.error(f"❌ Error during search: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

def show_basic_search(query):
    """Show basic search results without QA engine"""
    vector_store = st.session_state['vector_store']
    top_k = st.session_state.get('top_k', 5)
    
    results = vector_store.retrieve(query, top_k=top_k)
    
    if not results:
        st.warning("No results found.")
        return
    
    st.markdown("### 🔍 Search Results")
    for i, result in enumerate(results, 1):
        st.markdown(f"""
        <div class="source-card">
            <strong>Result {i}</strong> - Page {result['metadata']['page']} ({result['metadata']['type']})<br>
            {result['content'][:300]}...
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
