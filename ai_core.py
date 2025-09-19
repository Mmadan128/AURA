import os
import time
from typing import List, Dict, Any, Optional, Generator
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import UnstructuredPDFLoader, UnstructuredImageLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.memory import ConversationBufferMemory
from operator import itemgetter
from langdetect import detect, LangDetectException
from gtts import gTTS
import pycountry
from google.generativeai.types import HarmCategory, HarmBlockThreshold


class AURACore:
    """Core AI functionality for Project AURA"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.embeddings = None
        self.chat_model = None
        self.vectordb = None
        self.retriever = None
        self.memories = {}  # Store memories per session
        
    def initialize_models(self):
        """Initialize the AI models"""
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.config['hf_embed_model']
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
        
    def build_or_load_vector_db(self, progress_callback=None) -> bool:
        """Build or load the vector database"""
        # Check if index exists and is valid
        index_file = os.path.join(self.config['faiss_index_dir'], 'index.faiss')
        if os.path.exists(index_file):
            try:
                self.vectordb = FAISS.load_local(
                    self.config['faiss_index_dir'], 
                    self.embeddings, 
                    allow_dangerous_deserialization=True
                )
                if progress_callback:
                    progress_callback(1, 1, "Loaded existing index")
                return True
            except Exception as e:
                print(f"Failed to load existing index: {e}")
                # If loading fails, rebuild the index
                return self._build_new_index(progress_callback)
        else:
            return self._build_new_index(progress_callback)
            
    def _build_new_index(self, progress_callback=None) -> bool:
        """Build a new vector index from documents"""
        print("Building new FAISS index...")
        
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
            print(f"PDF folder: {self.config['pdf_folder']} exists: {os.path.exists(self.config['pdf_folder'])}")
            print(f"Image folder: {self.config['image_folder']} exists: {os.path.exists(self.config['image_folder'])}")
            return False
            
        print(f"Processing {len(files_to_process)} total files...")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=100
        )
        
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
                    
                split_docs = text_splitter.split_documents(docs)
                print(f"Split into {len(split_docs)} chunks")
                
                if vectordb is None:
                    vectordb = FAISS.from_documents(split_docs, self.embeddings)
                    print("Created initial FAISS index")
                else:
                    vectordb.add_documents(split_docs)
                    print("Added documents to existing index")
                    
            except Exception as e:
                print(f"Error processing {fp}: {e}")
                continue
        
        if vectordb is None:
            print("Failed to create any vector index!")
            return False
            
        # Ensure directory exists
        os.makedirs(self.config['faiss_index_dir'], exist_ok=True)
        
        try:
            vectordb.save_local(self.config['faiss_index_dir'])
            print(f"Saved FAISS index to {self.config['faiss_index_dir']}")
            self.vectordb = vectordb
            return True
        except Exception as e:
            print(f"Error saving FAISS index: {e}")
            return False
        
    def get_retriever(self):
        """Get the document retriever"""
        if self.vectordb:
            self.retriever = self.vectordb.as_retriever(
                search_type="similarity", 
                search_kwargs={"k": 5}
            )
            return self.retriever
        return None
        
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
        """Create the RAG chain with memory"""
        template = """
        You are AURA, a helpful AI assistant. Use the following context and conversation history to answer the question.
        Answer faithfully and accurately based ONLY on the provided context and history.
        The user's question is in the language '{language}'. You MUST write your entire answer in that same language.
        If the user asks in Hinglish, you MUST reply in natural-sounding Hinglish.

        CHAT HISTORY:
        {chat_history}

        CONTEXT:
        {context}

        QUESTION:
        {question}

        ANSWER (in {language}):
        """
        prompt = PromptTemplate.from_template(template)
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            RunnablePassthrough.assign(
                chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("history")
            )
            | {
                "context": itemgetter("question") | self.retriever | format_docs,
                "question": itemgetter("question"),
                "language": itemgetter("language"),
                "chat_history": itemgetter("chat_history")
              }
            | prompt
            | self.chat_model
            | StrOutputParser()
        )
        return rag_chain
        
    def generate_response(self, question: str, session_id: str) -> Generator[str, None, None]:
        """Generate streaming response"""
        lang_code, language_name = self.detect_language(question)
        memory = self.get_or_create_memory(session_id)
        rag_chain = self.create_rag_chain(memory)
        
        # Stream the response
        response_chunks = []
        for chunk in rag_chain.stream({"question": question, "language": language_name}):
            response_chunks.append(chunk)
            yield chunk
            
        # Save to memory after complete response
        full_response = ''.join(response_chunks)
        memory.save_context({"question": question}, {"output": full_response})
        
    def get_retrieved_documents(self, question: str) -> List[Dict[str, Any]]:
        """Get retrieved documents for a question"""
        if not self.retriever:
            return []
            
        docs = self.retriever.invoke(question)
        return [
            {
                "source": doc.metadata.get('source', 'N/A'),
                "content": doc.page_content
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