from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import os
import json
import uuid
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="AURA Backend API", version="1.0.0")

# Get frontend URL from environment variable (fallback to localhost for development)
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

# Global variables for tracking initialization
initialization_status = {"initialized": False, "error": None, "initializing": False}
build_progress = {"current": 0, "total": 0, "file": "", "completed": False}
aura_core = None

# Pydantic models for API
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

class IndexBuildStatus(BaseModel):
    status: str
    message: str
    progress: Optional[Dict[str, Any]] = None

def update_build_progress(current: int, total: int, filename: str):
    """Callback to update build progress"""
    global build_progress
    build_progress.update({
        "current": current,
        "total": total,
        "file": filename,
        "completed": False
    })

async def initialize_aura_core():
    """Initialize AURA Core asynchronously"""
    global initialization_status, aura_core
    
    if initialization_status["initializing"]:
        return
    
    initialization_status["initializing"] = True
    
    try:
        print("Starting AURA Core initialization...")
        
        if not CONFIG["gemini_api_key"]:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        # Import here to catch import errors
        from ai_core import AURACore
        aura_core = AURACore(CONFIG, google_api_key=CONFIG["gemini_api_key"])
        
        print("Initializing AI models...")
        aura_core.initialize_models()
        
        print("Building/loading vector database...")
        # Run this in a thread to avoid blocking
        success = await asyncio.get_event_loop().run_in_executor(
            None, 
            lambda: aura_core.build_or_load_vector_db(progress_callback=update_build_progress)
        )
        
        if not success:
            initialization_status = {
                "initialized": False, 
                "error": "No documents found to process",
                "initializing": False
            }
            return
            
        print("Getting retriever...")
        aura_core.get_retriever()
        
        initialization_status = {"initialized": True, "error": None, "initializing": False}
        build_progress["completed"] = True
        
        print("AURA Core initialization completed successfully!")
        
    except ImportError as e:
        error_msg = f"Failed to import ai_core: {str(e)}"
        print(f"Import error: {error_msg}")
        initialization_status = {"initialized": False, "error": error_msg, "initializing": False}
    except Exception as e:
        error_msg = f"Initialization failed: {str(e)}"
        print(f"Initialization error: {error_msg}")
        initialization_status = {"initialized": False, "error": error_msg, "initializing": False}

@app.on_event("startup")
async def startup_event():
    """Initialize the AI models on startup"""
    # Start initialization in background to not block startup
    asyncio.create_task(initialize_aura_core())

@app.get("/")
async def root():
    return {"message": "AURA Backend API is running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if initialization_status["initialized"] else "initializing",
        "initialized": initialization_status["initialized"],
        "initializing": initialization_status.get("initializing", False),
        "error": initialization_status["error"]
    }

@app.get("/init-status")
async def get_init_status():
    """Get initialization status"""
    return {
        "initialized": initialization_status["initialized"],
        "initializing": initialization_status.get("initializing", False),
        "error": initialization_status["error"],
        "build_progress": build_progress
    }

@app.post("/initialize")
async def manual_initialize():
    """Manually trigger initialization"""
    if not initialization_status["initialized"] and not initialization_status.get("initializing", False):
        asyncio.create_task(initialize_aura_core())
        return {"message": "Initialization started"}
    elif initialization_status["initialized"]:
        return {"message": "Already initialized"}
    else:
        return {"message": "Initialization in progress"}

@app.post("/chat")
async def chat_endpoint(chat_message: ChatMessage):
    """Main chat endpoint"""
    if not initialization_status["initialized"]:
        if initialization_status.get("initializing", False):
            raise HTTPException(
                status_code=503, 
                detail="System is still initializing. Please wait and try again."
            )
        else:
            raise HTTPException(
                status_code=503, 
                detail=f"System not initialized: {initialization_status['error']}"
            )
    
    try:
        # Generate session ID if not provided
        session_id = chat_message.session_id or str(uuid.uuid4())
        
        # Generate response (collect all chunks)
        response_chunks = []
        for chunk in aura_core.generate_response(chat_message.message, session_id):
            response_chunks.append(chunk)
        
        full_response = ''.join(response_chunks)
        
        # Get sources
        sources = aura_core.get_retrieved_documents(chat_message.message)
        
        # Detect language
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
    if not initialization_status["initialized"]:
        raise HTTPException(
            status_code=503, 
            detail=f"System not initialized: {initialization_status['error']}"
        )
    
    try:
        # Generate session ID if not provided
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
    if not initialization_status["initialized"]:
        raise HTTPException(
            status_code=503, 
            detail=f"System not initialized: {initialization_status['error']}"
        )
    
    try:
        audio_bytes = aura_core.generate_audio(
            audio_request.text, 
            audio_request.lang_code
        )
        
        if audio_bytes is None:
            raise HTTPException(
                status_code=500, 
                detail="Failed to generate audio"
            )
        
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
    if not initialization_status["initialized"]:
        raise HTTPException(
            status_code=503, 
            detail=f"System not initialized: {initialization_status['error']}"
        )
    
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
    if not initialization_status["initialized"]:
        raise HTTPException(
            status_code=503, 
            detail="System not initialized"
        )
    
    def rebuild_task():
        global build_progress, initialization_status
        build_progress = {"current": 0, "total": 0, "file": "", "completed": False}
        
        try:
            # Remove existing index
            import shutil
            if os.path.exists(CONFIG["faiss_index_dir"]):
                shutil.rmtree(CONFIG["faiss_index_dir"])
            
            # Rebuild
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
