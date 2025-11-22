from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn

# Import our custom modules
from ingestion import KnowledgeBase
from rag_agent import TestGenAgent
from selenium_agent import SeleniumAgent

app = FastAPI(title="QA Agent Backend")

# --- 1. Initialize Logic Components ---
kb = KnowledgeBase()

# Initialize Test Generation Agent (RAG)
try:
    test_gen_agent = TestGenAgent()
except Exception as e:
    print(f"Warning: TestGenAgent not initialized. {e}")
    test_gen_agent = None

# Initialize Selenium Script Agent
try:
    selenium_agent = SeleniumAgent()
except Exception as e:
    print(f"Warning: SeleniumAgent not initialized. {e}")
    selenium_agent = None


# --- 2. Data Models (Pydantic) ---
class TestRequest(BaseModel):
    query: str

class ScriptRequest(BaseModel):
    test_case: dict
    html_content: str


# --- 3. API Endpoints ---

@app.get("/")
def home():
    return {"message": "QA Agent Backend is Running"}

@app.post("/upload-docs")
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    Endpoint 1: Ingest Documents
    Takes PDF/MD/TXT files, chunks them, and stores in Vector DB.
    """
    try:
        status = kb.ingest_documents(files)
        return {"status": "success", "message": status}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-tests")
async def generate_test_cases(request: TestRequest):
    """
    Endpoint 2: RAG-based Test Case Generation
    Input: User query (e.g., "Generate positive tests for login")
    Output: JSON list of test cases
    """
    if not test_gen_agent:
        raise HTTPException(status_code=500, detail="Test Agent not initialized. Check server logs for API Key errors.")
        
    response = test_gen_agent.generate_tests(request.query)
    return response

@app.post("/generate-script")
async def generate_selenium_script(request: ScriptRequest):
    """
    Endpoint 3: Generate Selenium Script
    Input: A single test case JSON object + The full HTML string
    Output: Python code string
    """
    if not selenium_agent:
        raise HTTPException(status_code=500, detail="Selenium Agent not initialized. Check server logs for API Key errors.")
        
    response = selenium_agent.generate_script(request.test_case, request.html_content)
    return response


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)