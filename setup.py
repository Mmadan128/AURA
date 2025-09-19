#!/usr/bin/env python3
"""
Quick setup script for AURA - creates test documents and builds index
"""

import os
import shutil
from pathlib import Path

def create_test_documents():
    """Create some test documents for AURA to work with"""
    print("📄 Creating test documents...")
    
    # Create directories
    Path("pdfs").mkdir(exist_ok=True)
    Path("images").mkdir(exist_ok=True)
    
    # Create a simple text file that we can convert to PDF later
    # For now, let's create a simple text document that unstructured can process
    test_txt_path = Path("pdfs") / "test_document.txt"
    
    with open(test_txt_path, 'w', encoding='utf-8') as f:
        f.write("""
# AURA Test Document

## What is AURA?
AURA (Autonomous, Multimodal Campus Assistant) is an AI-powered document query system.

## Features
- Document processing for PDFs and images
- Conversational memory across sessions
- Multilingual support with automatic language detection
- Audio generation for responses
- Real-time streaming responses

## Technical Stack
- Backend: FastAPI with Python
- Frontend: Streamlit
- AI: LangChain + Google Gemini
- Vector Database: FAISS
- Embeddings: Sentence Transformers

## How to Use
1. Place your documents in the pdfs/ or images/ folder
2. Start the backend server
3. Launch the Streamlit frontend
4. Ask questions about your documents

## Machine Learning Concepts
Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed for every task.

### Types of Machine Learning:
1. **Supervised Learning**: Learning with labeled examples
2. **Unsupervised Learning**: Finding patterns in unlabeled data  
3. **Reinforcement Learning**: Learning through interaction and feedback

### Common Applications:
- Natural Language Processing
- Computer Vision
- Recommendation Systems
- Autonomous Vehicles
- Medical Diagnosis

This document serves as a test case for AURA's document processing capabilities.
        """)
    
    print(f"✅ Created test document: {test_txt_path}")
    
    # Create another test document
    test_txt_path2 = Path("pdfs") / "ai_basics.txt"
    
    with open(test_txt_path2, 'w', encoding='utf-8') as f:
        f.write("""
# Artificial Intelligence Basics

## What is Artificial Intelligence?
Artificial Intelligence (AI) is the simulation of human intelligence in machines that are programmed to think and learn like humans.

## History of AI
- 1950: Alan Turing proposes the Turing Test
- 1956: The term "Artificial Intelligence" is coined at Dartmouth Conference
- 1960s-1970s: Early AI programs and expert systems
- 1980s-1990s: Machine learning renaissance
- 2000s-2010s: Big data and deep learning revolution
- 2020s: Large language models and generative AI

## Types of AI
1. **Narrow AI**: Designed for specific tasks (like chess, image recognition)
2. **General AI**: Hypothetical AI with human-level intelligence across all domains
3. **Super AI**: AI that surpasses human intelligence

## Current AI Applications
- Virtual assistants (Siri, Alexa, Google Assistant)
- Image and speech recognition
- Autonomous vehicles
- Medical diagnosis
- Financial fraud detection
- Content recommendation
- Language translation

## AI Ethics and Considerations
- Bias in AI systems
- Privacy concerns
- Job displacement
- Transparency and explainability
- Safety and security

The field of AI continues to evolve rapidly, with new breakthroughs happening regularly.
        """)
    
    print(f"✅ Created test document: {test_txt_path2}")
    
    return True

def check_requirements():
    """Check if basic requirements are met"""
    print("🔍 Checking requirements...")
    
    # Check if .env exists and has API key
    env_file = Path('.env')
    if not env_file.exists():
        print("❌ .env file not found!")
        return False
    
    with open('.env', 'r') as f:
        content = f.read()
        if 'GOOGLE_API_KEY=' not in content or 'your_google_api_key_here' in content:
            print("❌ Please set your GOOGLE_API_KEY in the .env file")
            return False
    
    print("✅ Environment file looks good")
    return True

def clean_faiss_index():
    """Clean existing FAISS index if it's corrupted"""
    print("🧹 Cleaning existing FAISS index...")
    
    faiss_dir = Path("faiss_index")
    if faiss_dir.exists():
        shutil.rmtree(faiss_dir)
        print("✅ Removed old FAISS index")
    
    faiss_dir.mkdir(exist_ok=True)
    print("✅ Created fresh FAISS index directory")

def test_ai_core():
    """Test the AI core functionality"""
    print("🧠 Testing AI Core...")
    
    try:
        from ai_core import AURACore
        
        config = {
            "gemini_api_key": os.getenv("GOOGLE_API_KEY"),
            "pdf_folder": "pdfs",
            "image_folder": "images", 
            "faiss_index_dir": "faiss_index",
            "hf_embed_model": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            "gemini_model": "gemini-1.5-flash",
        }
        
        aura = AURACore(config)
        print("✅ AI Core initialized")
        
        aura.initialize_models()
        print("✅ Models loaded")
        
        success = aura.build_or_load_vector_db()
        if success:
            print("✅ Vector database built successfully")
            aura.get_retriever()
            print("✅ Retriever configured")
            return True
        else:
            print("❌ Failed to build vector database")
            return False
            
    except Exception as e:
        print(f"❌ AI Core test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🤖 AURA Quick Setup")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Setup requirements not met!")
        return False
    
    # Clean old index
    clean_faiss_index()
    
    # Create test documents
    if not create_test_documents():
        print("\n❌ Failed to create test documents!")
        return False
    
    # Load environment
    from dotenv import load_dotenv
    load_dotenv()
    
    # Test AI core
    if not test_ai_core():
        print("\n❌ AI Core test failed!")
        return False
    
    print("\n🎉 Setup completed successfully!")
    print("\nYou can now run:")
    print("1. python backend.py  # Start the backend")
    print("2. streamlit run frontend.py  # Start the frontend")
    print("\nOr use: python run_app.py")
    
    return True

if __name__ == "__main__":
    main()