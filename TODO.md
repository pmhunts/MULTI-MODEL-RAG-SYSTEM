# Critical-Path Testing for Multi-RAG System

## 1. Ingestion Testing

- Test `ingestion/parser.py` with a PDF containing text, tables, and images.
- Verify extraction of text, tables (as lists), and images (PIL objects).
- Confirm metadata like page numbers are correct.

## 2. Chunking Testing

- Test `chunking/semantic_chunker.py` text chunking with a sample large text.
- Validate chunk size, overlap, and sentence boundaries.
- Test table conversion to markdown-like text format.

## 3. Embedding Testing

- Test `embedding/multimodal_embedder.py` embeddings for:
  - Single text
  - Multiple texts in batch
  - Table texts (converted from tables)
  - Images (PIL.Image)
- Check output shapes and types (numpy arrays).

## 4. Vector Store Testing

- Add sample documents with text and tables to `embedding/multimodal_vector_store.py`.
- Perform vector similarity and hybrid search.
- Confirm retrieval of expected documents by metadata and distance.

## 5. QA Engine Testing

- Create a `generation/qa_engine.py` QAEngine instance with vector store.
- Test query generation and retrieval.
- Confirm answer format and source extraction.

## 6. Integration Testing

- Full pipeline test:
  - Parse sample PDF
  - Chunk parsed elements
  - Embed chunks
  - Add to vector store
  - Query QAEngine for answers
- Verify end-to-end correctness and performance.

---

After testing, report any bugs or issues found for fixes before final task completion.
