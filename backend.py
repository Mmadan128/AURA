from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import os
import json
import uuid
import threading
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="AURA Backend API", version="1.0.0")

# Get frontend URL from environment variable
FRONTEND_URL = os.getenv("FRONTEND_URL", "*")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL] if FRONTEND_URL != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
CONFIG = {
    "gemini_api_key": os.getenv("GOOGLE_API_KEY"),
    "pdf_folder": "pdfs",
    "image_folder": "images",
    "faiss_index_dir": "faiss_index",
    "hf_embed_model": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "gemini_model": "gemini-2.0-flash",
}

# Global variables
aura_core = None
initialization_status = {"initialized": False, "error": None, "initializing": False}
build_progress = {"current": 0, "total": 0, "file": "", "completed": False}

# Pydantic models
class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    sources: List[Dict[str, Any]]
    language: str

class AudioRequest(BaseModel):
    text: str
    lang_code: str = 'en'

def update_build_progress(current: int, total: int, filename: str):
    """Callback to update build progress"""
    global build_progress
    build_progress.update({
        "current": current,
        "total": total,
        "file": filename,
        "completed": False
    })

def initialize_aura_background():
    """Initialize AURA Core in background thread"""
    global aura_core, initialization_status
    
    print("Starting background initialization...")
    initialization_status["initializing"] = True
    
    try:
        if not CONFIG["gemini_api_key"]:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        print("Importing AURACore...")
        from ai_core import AURACore
        
        print("Creating AURACore instance...")
        aura_core = AURACore(CONFIG, google_api_key=CONFIG["gemini_api_key"])
        
        print("Initializing models...")
        aura_core.initialize_models()
        
        print("Creating directories...")
        os.makedirs(CONFIG["pdf_folder"], exist_ok=True)
        os.makedirs(CONFIG["image_folder"], exist_ok=True)
        os.makedirs(CONFIG["faiss_index_dir"], exist_ok=True)
        
        print("Building/loading vector database...")
        success = aura_core.build_or_load_vector_db(progress_callback=update_build_progress)
        
        if success:
            print("Getting retriever...")
            aura_core.get_retriever()
            initialization_status = {"initialized": True, "error": None, "initializing": False}
            build_progress["completed"] = True
            print("✅ AURA Core initialized successfully!")
        else:
            error_msg = "No documents found to process. Please add documents to pdfs/ or images/ folder."
            initialization_status = {"initialized": False, "error": error_msg, "initializing": False}
            print(f"❌ {error_msg}")
            
    except Exception as e:
        error_msg = f"Initialization failed: {str(e)}"
        print(f"❌ {error_msg}")
        initialization_status = {"initialized": False, "error": error_msg, "initializing": False}

@app.on_event("startup")
async def startup_event():
    """Start background initialization without blocking"""
    print("🚀 FastAPI startup - starting background initialization...")
    
    # Start initialization in a separate thread
    init_thread = threading.Thread(target=initialize_aura_background)
    init_thread.daemon = True
    init_thread.start()
    
    print("✅ FastAPI startup completed - initialization running in background")

@app.get("/")
async def root():
    return {
        "message": "AURA Backend API is running",
        "version": "1.0.0",
        "status": "healthy"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "server_running": True,
        "api_key_configured": bool(CONFIG["gemini_api_key"]),
        "folders_exist": {
            "pdfs": os.path.exists(CONFIG["pdf_folder"]),
            "images": os.path.exists(CONFIG["image_folder"]),
            "faiss_index": os.path.exists(CONFIG["faiss_index_dir"])
        },
        "aura_status": {
            "initialized": initialization_status["initialized"],
            "initializing": initialization_status["initializing"],
            "error": initialization_status.get("error")
        },
        "build_progress": build_progress
    }

@app.get("/init-status")
async def get_init_status():
    """Get detailed initialization status"""
    return {
        "initialized": initialization_status["initialized"],
        "initializing": initialization_status["initializing"],
        "error": initialization_status["error"],
        "build_progress": build_progress
    }

@app.post("/initialize")
async def manual_initialize():
    """Manually trigger initialization if not started"""
    if not initialization_status["initialized"] and not initialization_status["initializing"]:
        init_thread = threading.Thread(target=initialize_aura_background)
        init_thread.daemon = True
        init_thread.start()
        return {"message": "Initialization started in background"}
    elif initialization_status["initialized"]:
        return {"message": "Already initialized"}
    else:
        return {"message": "Initialization already in progress"}

@app.post("/chat")
async def chat_endpoint(chat_message: ChatMessage):
    """Main chat endpoint"""
    if not initialization_status["initialized"]:
        if initialization_status["initializing"]:
            raise HTTPException(
                status_code=503,
                detail="System is still initializing. Please wait and try again in a few moments."
            )
        else:
            raise HTTPException(
                status_code=503,
                detail=f"System not initialized: {initialization_status.get('error', 'Unknown error')}"
            )
    
    if not aura_core:
        raise HTTPException(status_code=503, detail="AURA Core not available")
    
    try:
        # Generate session ID if not provided
        session_id = chat_message.session_id or str(uuid.uuid4())
        
        # Generate response
        response_chunks = []
        for chunk in aura_core.generate_response(chat_message.message, session_id):
            response_chunks.append(chunk)
        
        full_response = ''.join(response_chunks)
        
        # Get sources and detect language
        sources = aura_core.get_retrieved_documents(chat_message.message)
        lang_code, language_name = aura_core.detect_language(chat_message.message)
        
        return ChatResponse(
            response=full_response,
            session_id=session_id,
            sources=sources,
            language=language_name
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/stream")
async def chat_stream_endpoint(chat_message: ChatMessage):
    """Streaming chat endpoint"""
    if not initialization_status["initialized"] or not aura_core:
        raise HTTPException(
            status_code=503,
            detail="System not ready for streaming"
        )
    
    try:
        session_id = chat_message.session_id or str(uuid.uuid4())
        
        def generate_stream():
            for chunk in aura_core.generate_response(chat_message.message, session_id):
                yield f"data: {json.dumps({'chunk': chunk, 'session_id': session_id})}\n\n"
            yield f"data: {json.dumps({'done': True, 'session_id': session_id})}\n\n"
        
        return StreamingResponse(generate_stream(), media_type="text/plain")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-audio")
async def generate_audio_endpoint(audio_request: AudioRequest):
    """Generate audio from text"""
    if not initialization_status["initialized"] or not aura_core:
        raise HTTPException(status_code=503, detail="System not ready")
    
    try:
        audio_bytes = aura_core.generate_audio(
            audio_request.text, 
            audio_request.lang_code
        )
        
        if audio_bytes is None:
            raise HTTPException(status_code=500, detail="Failed to generate audio")
        
        return StreamingResponse(
            iter([audio_bytes]),
            media_type="audio/mpeg",
            headers={"Content-Disposition": "attachment; filename=response.mp3"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sources/{session_id}")
async def get_sources(session_id: str, query: str):
    """Get retrieved documents for a query"""
    if not initialization_status["initialized"] or not aura_core:
        raise HTTPException(status_code=503, detail="System not ready")
    
    try:
        sources = aura_core.get_retrieved_documents(query)
        return {"sources": sources}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/memory/{session_id}")
async def clear_session_memory(session_id: str):
    """Clear memory for a specific session"""
    try:
        if aura_core:
            aura_core.clear_memory(session_id)
        return {"message": f"Memory cleared for session {session_id}"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rebuild-index")
async def rebuild_index(background_tasks: BackgroundTasks):
    """Trigger index rebuild"""
    if not initialization_status["initialized"] or not aura_core:
        raise HTTPException(status_code=503, detail="System not ready")
    
    def rebuild_task():
        global build_progress, initialization_status
        build_progress = {"current": 0, "total": 0, "file": "", "completed": False}
        
        try:
            import shutil
            if os.path.exists(CONFIG["faiss_index_dir"]):
                shutil.rmtree(CONFIG["faiss_index_dir"])
            
            success = aura_core.build_or_load_vector_db(
                progress_callback=update_build_progress
            )
            
            if success:
                aura_core.get_retriever()
                build_progress["completed"] = True
                initialization_status["initialized"] = True
            else:
                initialization_status = {
                    "initialized": False,
                    "error": "No documents found to process"
                }
                
        except Exception as e:
            initialization_status = {"initialized": False, "error": str(e)}
    
    background_tasks.add_task(rebuild_task)
    return {"message": "Index rebuild started"}

@app.get("/build-progress")
async def get_build_progress():
    """Get current build progress"""
    return build_progress
