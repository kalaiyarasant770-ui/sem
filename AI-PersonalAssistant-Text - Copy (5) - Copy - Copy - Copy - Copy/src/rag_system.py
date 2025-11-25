
# rag_system.py
import pickle
import logging
import time
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from config import VECTOR_DB_PATH, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDINGS_MODEL

SCRIPT_DIR = Path(__file__).parent.resolve()

class RAGSystem:
    """Retrieval-Augmented Generation system for document search."""
    
    def __init__(self, vector_db_path: str = VECTOR_DB_PATH):
        t_rag_start = time.time()
        logging.info("🔄 Initializing RAG System...")
        
        self.vector_db_path = SCRIPT_DIR / vector_db_path
        self.vector_db_path.mkdir(exist_ok=True)
        logging.info(f"   Vector DB path: {self.vector_db_path}")
        
        self.embedding_model = None
        self.faiss_index = None
        self.chunks = []
        self.chunk_metadata = []
        
        logging.info(f"🔍 Checking for existing RAG data in '{self.vector_db_path}'...")
        self._load_or_create_vector_db()
        
        logging.info(f"✅ RAG System initialized in {time.time() - t_rag_start:.2f} seconds")

    def _load_or_create_vector_db(self):
        """Load existing vector DB or create new one."""
        index_path = self.vector_db_path / "faiss_index.bin"
        chunks_path = self.vector_db_path / "chunks.pkl"
        metadata_path = self.vector_db_path / "metadata.pkl"
        
        if index_path.exists() and chunks_path.exists() and metadata_path.exists():
            try:
                logging.info(f"📂 Found existing RAG files:")
                logging.info(f"   - {index_path.name}")
                logging.info(f"   - {chunks_path.name}")
                logging.info(f"   - {metadata_path.name}")
                
                t_load = time.time()
                self.faiss_index = faiss.read_index(str(index_path))
                with open(chunks_path, 'rb') as f:
                    self.chunks = pickle.load(f)
                with open(metadata_path, 'rb') as f:
                    self.chunk_metadata = pickle.load(f)
                
                unique_docs = len(set([m['document'] for m in self.chunk_metadata]))
                logging.info(f"✅ Successfully loaded existing RAG data in {time.time() - t_load:.2f} seconds")
                logging.info(f"   Total chunks: {len(self.chunks)}")
                logging.info(f"   Documents indexed: {unique_docs}")
                return
            except Exception as e:
                logging.error(f"❌ Failed to load RAG data: {e}")
                logging.info("   Creating new index...")
        else:
            logging.info(f"📦 No existing RAG data found")
            missing_files = []
            if not index_path.exists():
                missing_files.append(index_path.name)
            if not chunks_path.exists():
                missing_files.append(chunks_path.name)
            if not metadata_path.exists():
                missing_files.append(metadata_path.name)
            if missing_files:
                logging.info(f"   Missing files: {', '.join(missing_files)}")
        
        dimension = 384
        self.faiss_index = faiss.IndexFlatL2(dimension)
        self.chunks = []
        self.chunk_metadata = []
        logging.info(f"📦 Created new empty FAISS index (dimension: {dimension})")

    def _load_embedding_model(self):
        """Lazy load embedding model."""
        if self.embedding_model is None:
            t_model_start = time.time()
            logging.info(f"🔄 Loading SentenceTransformer model: {EMBEDDINGS_MODEL}")
            self.embedding_model = SentenceTransformer(EMBEDDINGS_MODEL)
            logging.info(f"✅ SentenceTransformer model loaded in {time.time() - t_model_start:.2f} seconds")

    def add_document(self, file_path: str):
        """Add a document to the RAG system."""
        file_path = Path(file_path)
        logging.info(f"📄 Processing document: {file_path.name}")
        
        t_extract = time.time()
        text_content = self._extract_text(file_path)
        if not text_content:
            logging.warning(f"⚠️  No text extracted from {file_path.name}")
            return False
        logging.info(f"   Extracted text in {time.time() - t_extract:.2f} seconds ({len(text_content)} chars)")
        
        t_chunk = time.time()
        chunks = self._split_text(text_content)
        if not chunks:
            logging.warning(f"⚠️  No chunks created from {file_path.name}")
            return False
        logging.info(f"   Created {len(chunks)} chunks in {time.time() - t_chunk:.2f} seconds")
        
        self._load_embedding_model()
        
        t_embed = time.time()
        logging.info(f"   Generating embeddings for {len(chunks)} chunks...")
        embeddings = self.embedding_model.encode(chunks, show_progress_bar=False)
        logging.info(f"   Generated embeddings in {time.time() - t_embed:.2f} seconds")
        
        self.faiss_index.add(embeddings.astype('float32'))
        
        doc_name = file_path.name
        for i, chunk in enumerate(chunks):
            self.chunks.append(chunk)
            self.chunk_metadata.append({
                'document': doc_name,
                'chunk_id': len(self.chunks),
                'text_preview': chunk[:100] + "..." if len(chunk) > 100 else chunk
            })
        
        self._save_vector_db()
        logging.info(f"✅ Successfully indexed {file_path.name}: {len(chunks)} chunks added (Total: {len(self.chunks)})")
        return True

    def _extract_text(self, file_path: Path) -> str:
        """Extract text from file."""
        if file_path.suffix.lower() == '.txt':
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    return file.read()
            except Exception as e:
                logging.error(f"❌ Failed to read {file_path.name}: {e}")
                return ""
        logging.warning(f"⚠️  Unsupported file type: {file_path.suffix}")
        return ""

    def _split_text(self, text: str) -> list:
        """Split text into chunks with overlap."""
        chunks = []
        words = text.split()
        
        for i in range(0, len(words), CHUNK_SIZE - CHUNK_OVERLAP):
            chunk_words = words[i:i + CHUNK_SIZE]
            chunk = " ".join(chunk_words)
            if chunk.strip():
                chunks.append(chunk.strip())
        
        return chunks

    def _save_vector_db(self):
        """Save vector database to disk."""
        t_save = time.time()
        index_path = self.vector_db_path / "faiss_index.bin"
        chunks_path = self.vector_db_path / "chunks.pkl"
        metadata_path = self.vector_db_path / "metadata.pkl"
        
        faiss.write_index(self.faiss_index, str(index_path))
        with open(chunks_path, 'wb') as f:
            pickle.dump(self.chunks, f)
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.chunk_metadata, f)
        
        logging.info(f"💾 Saved RAG data to {self.vector_db_path} in {time.time() - t_save:.2f} seconds")

    def search(self, query: str, top_k: int = 3) -> list:
        """Search for relevant chunks."""
        logging.debug(f"🔍 Searching RAG for: {query[:80]}...")
        self._load_embedding_model()
        
        if not self.embedding_model or not self.faiss_index or not self.chunks:
            logging.warning("⚠️  No embeddings or chunks available for search")
            return []
        
        query_embedding = self.embedding_model.encode([query])
        distances, indices = self.faiss_index.search(query_embedding.astype('float32'), top_k)
        
        results = [
            {
                'text': self.chunks[idx],
                'metadata': self.chunk_metadata[idx],
                'similarity_score': float(1 / (1 + distance)),
                'rank': i + 1
            }
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0]))
            if idx < len(self.chunks)
        ]
        
        logging.debug(f"   Found {len(results)} relevant chunks")
        return results


def auto_load_documents(rag_system: RAGSystem):
    """Load documents from maintenance_docs folder into RAG system."""
    logging.info(f"\n{'='*80}")
    logging.info(f"📚 AUTO-LOADING DOCUMENTS FROM maintenance_docs/")
    logging.info(f"{'='*80}")
    
    docs_folder = SCRIPT_DIR / "maintenance_docs"
    
    # Check if RAG data already exists
    if len(rag_system.chunks) > 0:
        logging.info(f"✅ RAG already loaded with {len(rag_system.chunks)} chunks from previous session")
        logging.info(f"   Skipping reload to save time")
        logging.info(f"{'='*80}\n")
        return
    
    # Check if maintenance_docs folder exists
    if not docs_folder.exists():
        logging.warning(f"⚠️  '{docs_folder}' folder not found")
        logging.info(f"💡 To use RAG: Create '{docs_folder}' folder and add .txt files")
        logging.info(f"📍 Expected location: {docs_folder}")
        logging.info(f"{'='*80}\n")
        return
    
    # Get all .txt files
    doc_files = list(docs_folder.glob("*.txt"))
    if not doc_files:
        logging.warning(f"⚠️  No .txt files found in '{docs_folder}'")
        logging.info(f"💡 Add chiller maintenance documentation (.txt files) to enable advisory features")
        logging.info(f"{'='*80}\n")
        return
    
    # Index all documents
    logging.info(f"📚 Found {len(doc_files)} documents in '{docs_folder}':")
    for doc_file in doc_files:
        logging.info(f"   - {doc_file.name}")
    
    logging.info(f"{'-'*80}")
    
    success_count = 0
    failed_count = 0
    t_total = time.time()
    
    for idx, doc_file in enumerate(doc_files, 1):
        try:
            logging.info(f"[{idx}/{len(doc_files)}] Processing: {doc_file.name}")
            if rag_system.add_document(str(doc_file)):
                success_count += 1
            else:
                failed_count += 1
                logging.warning(f"⚠️  Failed to index: {doc_file.name}")
        except Exception as e:
            failed_count += 1
            logging.error(f"❌ Error indexing {doc_file.name}: {e}")
    
    logging.info(f"{'-'*80}")
    
    if success_count > 0:
        logging.info(f"🎉 RAG SYSTEM READY!")
        logging.info(f"   ✅ Successfully indexed: {success_count}/{len(doc_files)} documents")
        logging.info(f"   📊 Total chunks: {len(rag_system.chunks)}")
        logging.info(f"   ⏱️  Total time: {time.time() - t_total:.2f} seconds")
        if failed_count > 0:
            logging.warning(f"   ⚠️  Failed: {failed_count} documents")
    else:
        logging.warning(f"⚠️  No documents were successfully indexed out of {len(doc_files)} files")
    
    logging.info(f"{'='*80}\n")