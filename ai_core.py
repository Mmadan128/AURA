import os
import time
import hashlib
import pickle
from typing import List, Dict, Any, Optional, Generator
from dataclasses import dataclass
from functools import lru_cache

# Core LangChain imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import UnstructuredPDFLoader, UnstructuredImageLoader
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document
from langchain.memory import ConversationBufferMemory
from langchain.retrievers import MergerRetriever, EnsembleRetriever
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from operator import itemgetter

# Additional imports
from langdetect import detect, LangDetectException
from gtts import gTTS
import pycountry
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class CacheEntry:
    """Cache entry for storing query results"""
    query_hash: str
    response: str
    retrieved_docs: List[Dict[str, Any]]
    timestamp: float
    hit_count: int = 0

class MetadataAwareEmbeddings:
    """Wrapper for embeddings that considers metadata"""
    
    def __init__(self, base_embeddings: HuggingFaceEmbeddings, metadata_weight: float = 0.2):
        self.base_embeddings = base_embeddings
        self.metadata_weight = metadata_weight
        
    def embed_documents(self, texts: List[str], metadatas: List[Dict] = None) -> List[List[float]]:
        """Embed documents with metadata awareness"""
        base_embeddings = self.base_embeddings.embed_documents(texts)
        
        if not metadatas:
            return base_embeddings
            
        enhanced_embeddings = []
        for i, (embedding, metadata) in enumerate(zip(base_embeddings, metadatas)):
            # Create metadata features
            metadata_features = self._extract_metadata_features(metadata)
            
            # Combine base embedding with metadata features
            enhanced_embedding = np.array(embedding)
            if metadata_features:
                metadata_array = np.array(metadata_features)
                # Pad or truncate to match embedding dimension
                if len(metadata_array) < len(enhanced_embedding):
                    metadata_array = np.pad(metadata_array, (0, len(enhanced_embedding) - len(metadata_array)))
                else:
                    metadata_array = metadata_array[:len(enhanced_embedding)]
                    
                # Weighted combination
                enhanced_embedding = (1 - self.metadata_weight) * enhanced_embedding + \
                                   self.metadata_weight * metadata_array
                                   
            enhanced_embeddings.append(enhanced_embedding.tolist())
            
        return enhanced_embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed query text"""
        return self.base_embeddings.embed_query(text)
    
    def _extract_metadata_features(self, metadata: Dict) -> List[float]:
        """Extract numerical features from metadata"""
        features = []
        
        # File type encoding
        source = metadata.get('source', '').lower()
        if '.pdf' in source:
            features.extend([1.0, 0.0, 0.0])
        elif any(ext in source for ext in ['.jpg', '.jpeg', '.png']):
            features.extend([0.0, 1.0, 0.0])
        else:
            features.extend([0.0, 0.0, 1.0])
            
        # Page number (normalized)
        page = metadata.get('page', 0)
        features.append(min(page / 100.0, 1.0))  # Normalize to 0-1
        
        # Document length (if available)
        length = metadata.get('length', 500)
        features.append(min(length / 10000.0, 1.0))  # Normalize to 0-1
        
        return features

class QueryRewriter:
    """Handles query rewriting and expansion"""
    
    def __init__(self, chat_model):
        self.chat_model = chat_model
        self.rewrite_prompt = PromptTemplate.from_template(
            """Given the user's question and chat history, rewrite the question to be more specific and searchable.
            Make it clear and focused while preserving the original intent.
            
            Chat History:
            {chat_history}
            
            Original Question: {question}
            
            Rewritten Question (be concise and specific):"""
        )
        
    def rewrite_query(self, question: str, chat_history: str = "") -> str:
        """Rewrite query for better retrieval"""
        try:
            chain = self.rewrite_prompt | self.chat_model | StrOutputParser()
            rewritten = chain.invoke({
                "question": question,
                "chat_history": chat_history
            })
            return rewritten.strip()
        except Exception as e:
            print(f"Query rewriting failed: {e}")
            return question
    
    def expand_query(self, question: str) -> List[str]:
        """Generate multiple query variations"""
        expansion_prompt = PromptTemplate.from_template(
            """Generate 3 different ways to ask the same question. Each should focus on different aspects:
            
            Original: {question}
            
            1. Specific technical version:
            2. Broader conceptual version:
            3. Alternative phrasing:
            
            Provide only the 3 questions, one per line."""
        )
        
        try:
            chain = expansion_prompt | self.chat_model | StrOutputParser()
            expansions = chain.invoke({"question": question})
            
            # Parse the response to extract questions
            lines = [line.strip() for line in expansions.split('\n') if line.strip()]
            expanded_queries = []
            
            for line in lines:
                # Remove numbering and prefixes
                cleaned = line.split(':', 1)[-1].strip()
                if cleaned and cleaned != question:
                    expanded_queries.append(cleaned)
                    
            return expanded_queries[:3]  # Return max 3 expansions
            
        except Exception as e:
            print(f"Query expansion failed: {e}")
            return []

class ResponseCache:
    """In-memory cache for responses with LRU eviction"""
    
    def __init__(self, max_size: int = 100, ttl: int = 3600):
        self.cache: Dict[str, CacheEntry] = {}
        self.max_size = max_size
        self.ttl = ttl  # Time to live in seconds
        
    def _generate_key(self, query: str, context_hash: str = "") -> str:
        """Generate cache key from query and context"""
        key_string = f"{query.lower().strip()}_{context_hash}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, query: str, context_hash: str = "") -> Optional[CacheEntry]:
        """Get cached response if valid"""
        key = self._generate_key(query, context_hash)
        
        if key in self.cache:
            entry = self.cache[key]
            
            # Check if expired
            if time.time() - entry.timestamp > self.ttl:
                del self.cache[key]
                return None
                
            entry.hit_count += 1
            return entry
            
        return None
    
    def put(self, query: str, response: str, retrieved_docs: List[Dict], context_hash: str = ""):
        """Cache response with LRU eviction"""
        key = self._generate_key(query, context_hash)
        
        # Evict least recently used if cache is full
        if len(self.cache) >= self.max_size:
            self._evict_lru()
            
        entry = CacheEntry(
            query_hash=key,
            response=response,
            retrieved_docs=retrieved_docs,
            timestamp=time.time()
        )
        
        self.cache[key] = entry
    
    def _evict_lru(self):
        """Evict least recently used entry"""
        if not self.cache:
            return
            
        # Find entry with lowest hit count and oldest timestamp
        lru_key = min(
            self.cache.keys(),
            key=lambda k: (self.cache[k].hit_count, self.cache[k].timestamp)
        )
        del self.cache[lru_key]
    
    def clear(self):
        """Clear all cached entries"""
        self.cache.clear()

class AURACore:
    """Enhanced Core AI functionality for Project AURA"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base_embeddings = None
        self.metadata_embeddings = None
        self.chat_model = None
        self.vectordb = None
        self.bm25_retriever = None
        self.retriever = None
        self.multivector_retriever = None
        self.memories = {}
        self.query_rewriter = None
        self.cache = ResponseCache(
            max_size=config.get('cache_size', 100),
            ttl=config.get('cache_ttl', 3600)
        )
        self.use_hybrid_retrieval = config.get('use_hybrid_retrieval', True)
        self.use_multivector = config.get('use_multivector', True)
        
    def initialize_models(self):
        """Initialize the AI models"""
        self.base_embeddings = HuggingFaceEmbeddings(
            model_name=self.config['hf_embed_model']
        )
        
        # Initialize metadata-aware embeddings
        self.metadata_embeddings = MetadataAwareEmbeddings(
            self.base_embeddings,
            metadata_weight=self.config.get('metadata_weight', 0.2)
        )
        
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        
        self.chat_model = ChatGoogleGenerativeAI(
            model=self.config['gemini_model'],
            google_api_key=self.config['gemini_api_key'],
            temperature=0.4,
            safety_settings=safety_settings
        )
        
        # Initialize query rewriter
        self.query_rewriter = QueryRewriter(self.chat_model)
        
    def build_or_load_vector_db(self, progress_callback=None) -> bool:
        """Build or load the vector database with multivector support"""
        # Check if index exists and is valid
        index_file = os.path.join(self.config['faiss_index_dir'], 'index.faiss')
        bm25_file = os.path.join(self.config['faiss_index_dir'], 'bm25_retriever.pkl')
        
        if os.path.exists(index_file):
            try:
                self.vectordb = FAISS.load_local(
                    self.config['faiss_index_dir'], 
                    self.base_embeddings, 
                    allow_dangerous_deserialization=True
                )
                
                # Load BM25 retriever if exists
                if os.path.exists(bm25_file) and self.use_hybrid_retrieval:
                    with open(bm25_file, 'rb') as f:
                        self.bm25_retriever = pickle.load(f)
                
                if progress_callback:
                    progress_callback(1, 1, "Loaded existing index")
                return True
            except Exception as e:
                print(f"Failed to load existing index: {e}")
                return self._build_new_index(progress_callback)
        else:
            return self._build_new_index(progress_callback)
    
    def _build_new_index(self, progress_callback=None) -> bool:
        """Build a new vector index with enhanced features"""
        print("Building new enhanced FAISS index...")
        
        files_to_process = []
        
        # Collect PDF files
        if os.path.exists(self.config['pdf_folder']):
            pdf_files = [
                (os.path.join(self.config['pdf_folder'], f), UnstructuredPDFLoader)
                for f in os.listdir(self.config['pdf_folder'])
                if f.lower().endswith(".pdf")
            ]
            files_to_process.extend(pdf_files)
            print(f"Found {len(pdf_files)} PDF files")
            
        # Collect image files
        if os.path.exists(self.config['image_folder']):
            image_files = [
                (os.path.join(self.config['image_folder'], f), UnstructuredImageLoader)
                for f in os.listdir(self.config['image_folder'])
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
            files_to_process.extend(image_files)
            print(f"Found {len(image_files)} image files")
            
        if not files_to_process:
            print("No documents found to process!")
            return False
            
        print(f"Processing {len(files_to_process)} total files...")
        
        # Enhanced text splitter with metadata preservation
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.get('chunk_size', 1000), 
            chunk_overlap=self.config.get('chunk_overlap', 200),
            add_start_index=True
        )
        
        all_documents = []
        all_texts = []
        vectordb = None
        total_files = len(files_to_process)
        
        for i, (fp, loader_cls) in enumerate(files_to_process):
            if progress_callback:
                progress_callback(i + 1, total_files, os.path.basename(fp))
            
            print(f"Processing {i+1}/{total_files}: {os.path.basename(fp)}")
            
            try:
                loader = (
                    loader_cls(fp, strategy="hi_res") 
                    if loader_cls == UnstructuredPDFLoader 
                    else loader_cls(fp)
                )
                docs = loader.load()
                
                if not docs:
                    print(f"Warning: No content extracted from {fp}")
                    continue
                
                # Split documents with enhanced metadata
                split_docs = text_splitter.split_documents(docs)
                
                # Enhance metadata for each chunk
                for j, doc in enumerate(split_docs):
                    doc.metadata.update({
                        'chunk_id': f"{os.path.basename(fp)}_{j}",
                        'file_type': 'pdf' if fp.endswith('.pdf') else 'image',
                        'chunk_index': j,
                        'total_chunks': len(split_docs),
                        'length': len(doc.page_content)
                    })
                
                all_documents.extend(split_docs)
                all_texts.extend([doc.page_content for doc in split_docs])
                print(f"Split into {len(split_docs)} chunks")
                    
            except Exception as e:
                print(f"Error processing {fp}: {e}")
                continue
        
        if not all_documents:
            print("Failed to process any documents!")
            return False
        
        # Create FAISS index with metadata-aware embeddings
        print("Creating FAISS vector store...")
        metadatas = [doc.metadata for doc in all_documents]
        enhanced_embeddings = self.metadata_embeddings.embed_documents(all_texts, metadatas)
        
        # Create FAISS index manually with enhanced embeddings
        import faiss
        dimension = len(enhanced_embeddings[0])
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(enhanced_embeddings).astype('float32'))
        
        self.vectordb = FAISS(
            embedding_function=self.base_embeddings.embed_query,
            index=index,
            docstore=InMemoryStore({str(i): doc for i, doc in enumerate(all_documents)}),
            index_to_docstore_id={i: str(i) for i in range(len(all_documents))}
        )
        
        # Create BM25 retriever for hybrid search
        if self.use_hybrid_retrieval:
            print("Creating BM25 retriever...")
            self.bm25_retriever = BM25Retriever.from_documents(all_documents)
            self.bm25_retriever.k = 5
        
        # Setup multivector retriever
        if self.use_multivector:
            print("Setting up multivector retriever...")
            self._setup_multivector_retriever(all_documents)
        
        # Save everything
        os.makedirs(self.config['faiss_index_dir'], exist_ok=True)
        
        try:
            self.vectordb.save_local(self.config['faiss_index_dir'])
            
            # Save BM25 retriever
            if self.bm25_retriever:
                bm25_file = os.path.join(self.config['faiss_index_dir'], 'bm25_retriever.pkl')
                with open(bm25_file, 'wb') as f:
                    pickle.dump(self.bm25_retriever, f)
            
            print(f"Saved enhanced index to {self.config['faiss_index_dir']}")
            return True
            
        except Exception as e:
            print(f"Error saving enhanced index: {e}")
            return False
    
    def _setup_multivector_retriever(self, documents: List[Document]):
        """Setup multivector retriever with summaries and sub-chunks"""
        try:
            # Create summaries for documents
            summary_prompt = PromptTemplate.from_template(
                "Summarize the following text in 2-3 sentences, focusing on key information:\n\n{text}"
            )
            
            summaries = []
            sub_chunks = []
            
            for doc in documents:
                # Create summary
                try:
                    summary_chain = summary_prompt | self.chat_model | StrOutputParser()
                    summary = summary_chain.invoke({"text": doc.page_content[:1000]})
                    
                    summary_doc = Document(
                        page_content=summary,
                        metadata={**doc.metadata, 'type': 'summary'}
                    )
                    summaries.append(summary_doc)
                    
                    # Create sub-chunks
                    sentences = doc.page_content.split('. ')
                    for i, sentence in enumerate(sentences):
                        if len(sentence.strip()) > 20:  # Filter out very short fragments
                            sub_doc = Document(
                                page_content=sentence.strip(),
                                metadata={**doc.metadata, 'type': 'sub_chunk', 'sub_index': i}
                            )
                            sub_chunks.append(sub_doc)
                            
                except Exception as e:
                    print(f"Error creating summary for document: {e}")
                    continue
            
            # Combine all document types
            all_multivector_docs = documents + summaries + sub_chunks
            
            # Create separate vectorstore for multivector
            if all_multivector_docs:
                multivector_db = FAISS.from_documents(all_multivector_docs, self.base_embeddings)
                self.multivector_retriever = multivector_db.as_retriever(
                    search_kwargs={"k": 8}
                )
                print(f"Created multivector retriever with {len(all_multivector_docs)} vectors")
                
        except Exception as e:
            print(f"Error setting up multivector retriever: {e}")
            self.multivector_retriever = None
        
    def get_retriever(self):
        """Get the appropriate retriever based on configuration"""
        if not self.vectordb:
            return None
        
        # Base FAISS retriever
        faiss_retriever = self.vectordb.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": 5}
        )
        
        # Return appropriate retriever based on configuration
        if self.use_hybrid_retrieval and self.bm25_retriever:
            # Ensemble retriever combining FAISS and BM25
            ensemble_retriever = EnsembleRetriever(
                retrievers=[faiss_retriever, self.bm25_retriever],
                weights=[0.7, 0.3]  # Favor semantic search slightly
            )
            
            if self.use_multivector and self.multivector_retriever:
                # Merge ensemble with multivector
                self.retriever = MergerRetriever(
                    retrievers=[ensemble_retriever, self.multivector_retriever]
                )
            else:
                self.retriever = ensemble_retriever
                
        elif self.use_multivector and self.multivector_retriever:
            # Use multivector with FAISS
            self.retriever = MergerRetriever(
                retrievers=[faiss_retriever, self.multivector_retriever]
            )
        else:
            # Use basic FAISS retriever
            self.retriever = faiss_retriever
            
        return self.retriever
        
    def get_or_create_memory(self, session_id: str) -> ConversationBufferMemory:
        """Get or create memory for a session"""
        if session_id not in self.memories:
            self.memories[session_id] = ConversationBufferMemory(
                memory_key="history", 
                return_messages=False
            )
        return self.memories[session_id]
        
    def clear_memory(self, session_id: str):
        """Clear memory for a session"""
        if session_id in self.memories:
            del self.memories[session_id]
            
    def detect_language(self, text: str) -> tuple[str, str]:
        """Detect language of input text"""
        try:
            lang_code = detect(text)
            language_obj = pycountry.languages.get(alpha_2=lang_code)
            language_name = language_obj.name if language_obj else "the user's language"
            return lang_code, language_name
        except (LangDetectException, AttributeError):
            return 'en', 'English'
    
    def create_rag_chain(self, memory: ConversationBufferMemory):
        """Create the enhanced RAG chain with caching and query rewriting"""
        template = """
        You are AURA, a helpful AI assistant. Use the following context and conversation history to answer the question.
        Answer faithfully and accurately based ONLY on the provided context and history.
        The user's question is in the language '{language}'. You MUST write your entire answer in that same language.
        If the user asks in Hinglish, you MUST reply in natural-sounding Hinglish.

        CHAT HISTORY:
        {chat_history}

        CONTEXT:
        {context}

        ORIGINAL QUESTION:
        {original_question}
        
        PROCESSED QUESTION:
        {question}

        ANSWER (in {language}):
        """
        prompt = PromptTemplate.from_template(template)
        
        def format_docs(docs):
            """Format retrieved documents with metadata"""
            formatted = []
            for doc in docs:
                doc_type = doc.metadata.get('type', 'chunk')
                source = doc.metadata.get('source', 'Unknown')
                
                content = f"[{doc_type.upper()}] {doc.page_content}"
                if doc_type != 'summary':
                    content += f" (Source: {os.path.basename(source)})"
                    
                formatted.append(content)
                
            return "\n\n".join(formatted)
        
        def rewrite_and_retrieve(inputs):
            """Rewrite query and retrieve with caching"""
            original_question = inputs["original_question"]
            chat_history = inputs.get("chat_history", "")
            
            # Check cache first
            context_hash = hashlib.md5(chat_history.encode()).hexdigest()[:8]
            cached = self.cache.get(original_question, context_hash)
            
            if cached:
                print("🎯 Cache hit!")
                return {
                    **inputs,
                    "question": original_question,  # Use original for cached response
                    "context": "\n\n".join([doc["content"] for doc in cached.retrieved_docs])
                }
            
            # Rewrite query for better retrieval
            rewritten_question = self.query_rewriter.rewrite_query(
                original_question, chat_history
            )
            
            print(f"🔄 Query rewritten: '{original_question}' → '{rewritten_question}'")
            
            # Retrieve documents
            docs = self.retriever.invoke(rewritten_question)
            
            # Cache the results (will be updated with response later)
            retrieved_docs = [
                {
                    "source": doc.metadata.get('source', 'N/A'),
                    "content": doc.page_content,
                    "type": doc.metadata.get('type', 'chunk')
                }
                for doc in docs
            ]
            
            return {
                **inputs,
                "question": rewritten_question,
                "context": format_docs(docs),
                "_retrieved_docs": retrieved_docs,
                "_context_hash": context_hash
            }

        rag_chain = (
            RunnablePassthrough.assign(
                chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("history")
            )
            | RunnableLambda(rewrite_and_retrieve)
            | prompt
            | self.chat_model
            | StrOutputParser()
        )
        
        return rag_chain
        
    def generate_response(self, question: str, session_id: str) -> Generator[str, None, None]:
        """Generate streaming response with enhanced features"""
        lang_code, language_name = self.detect_language(question)
        memory = self.get_or_create_memory(session_id)
        rag_chain = self.create_rag_chain(memory)
        
        # Stream the response
        response_chunks = []
        retrieved_docs = []
        context_hash = ""
        
        for chunk in rag_chain.stream({
            "original_question": question, 
            "language": language_name
        }):
            response_chunks.append(chunk)
            yield chunk
            
        # Save to memory and cache after complete response
        full_response = ''.join(response_chunks)
        memory.save_context({"question": question}, {"output": full_response})
        
        # Cache the response if it was newly generated
        chat_history = memory.load_memory_variables({}).get("history", "")
        context_hash = hashlib.md5(chat_history.encode()).hexdigest()[:8]
        
        if not self.cache.get(question, context_hash):  # Only cache if not already cached
            retrieved_docs = self.get_retrieved_documents(question)
            self.cache.put(question, full_response, retrieved_docs, context_hash)
        
    def get_retrieved_documents(self, question: str) -> List[Dict[str, Any]]:
        """Get retrieved documents for a question"""
        if not self.retriever:
            return []
            
        # Use rewritten query for consistency
        chat_history = ""  # Could get from active session if needed
        rewritten_question = self.query_rewriter.rewrite_query(question, chat_history)
        
        docs = self.retriever.invoke(rewritten_question)
        return [
            {
                "source": doc.metadata.get('source', 'N/A'),
                "content": doc.page_content,
                "type": doc.metadata.get('type', 'chunk'),
                "chunk_id": doc.metadata.get('chunk_id', 'N/A')
            }
            for doc in docs
        ]
        
    def generate_audio(self, text: str, lang_code: str = 'en') -> Optional[bytes]:
        """Generate audio from text"""
        try:
            tts = gTTS(text=text, lang=lang_code)
            audio_fp = f"response_{int(time.time())}.mp3"
            tts.save(audio_fp)
            
            with open(audio_fp, "rb") as f:
                audio_bytes = f.read()
                
            # Clean up temporary file
            if os.path.exists(audio_fp):
                os.remove(audio_fp)
                
            return audio_bytes
        except Exception as e:
            print(f"Audio generation error: {e}")
            return None
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_hits = sum(entry.hit_count for entry in self.cache.cache.values())
        return {
            "cache_size": len(self.cache.cache),
            "max_size": self.cache.max_size,
            "total_hits": total_hits,
            "hit_rate": total_hits / max(1, len(self.cache.cache))
        }
    
    def clear_cache(self):
        """Clear the response cache"""
        self.cache.clear()
        print("🗑️ Cache cleared")