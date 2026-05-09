import os
import json
import shutil
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
from ai_engine import AIEngine
from rag_vault import RAGVault

load_dotenv()

# Initialize RAG and AI Engine
rag_vault = RAGVault()
ai_engine = AIEngine(rag_vault=rag_vault)

app = FastAPI(title="ASPICE AI Requirement Agent API")

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerationRequest(BaseModel):
    user_input: str
    level: str  # "SYS.2" or "SWE.1"
    existing_context: Optional[str] = None

@app.get("/")
async def root():
    return {"message": "ASPICE AI Requirement Agent API is running"}

@app.post("/generate/sys2")
async def generate_sys2(request: GenerationRequest):
    try:
        requirements = await ai_engine.generate_sys2(request.user_input, request.existing_context)
        return {"status": "success", "requirements": requirements}
    except Exception as e:
        print(f"Error in SYS2 generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate/swe1")
async def generate_swe1(request: GenerationRequest):
    try:
        sys2_data = json.loads(request.user_input)
        requirements = await ai_engine.generate_swe1(sys2_data)
        return {"status": "success", "requirements": requirements}
    except Exception as e:
        print(f"Error in SWE1 generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest/file")
async def ingest_file(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        if file.filename.endswith('.pdf'):
            num_chunks = await rag_vault.ingest_pdf(temp_path)
        elif file.filename.endswith(('.xlsx', '.xls', '.csv')):
            num_chunks = await rag_vault.ingest_excel(temp_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
            
        return {"status": "success", "chunks_added": num_chunks}
    except Exception as e:
        print(f"Error in ingestion: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/stats")
async def get_stats():
    return rag_vault.get_stats()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
