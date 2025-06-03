from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Literal
from openai import OpenAI
from dotenv import load_dotenv
import os
import re
import logging
from enum import Enum

# Configure logging
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

# Load environment variables from .env file
load_dotenv()

# Create OpenAI client with basic configuration
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.openai.com/v1"
)

class InputType(str, Enum):
    PITCH = "pitch"
    CHAT = "chat"
    RANDOM = "random"

class TranscriptRequest(BaseModel):
    text: str

class AnalysisResponse(BaseModel):
    input_type: InputType
    key_strength: str = ""
    key_weakness: str = ""
    investor_impression: str = ""
    missed_opportunity: str = ""
    confidence_rating: str = ""
    final_summary: str = ""
    message: str = ""

def classify_input(text: str) -> InputType:
    """Classify the input text using GPT."""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": """You are a text classifier. Classify the input into one of these categories:
            - PITCH: Investor pitches, earnings calls, financial presentations
            - CHAT: General conversation, Q&A, informal discussions
            - RANDOM: Anything that doesn't fit the above
            
            Respond with ONLY the category name in caps."""},
            {"role": "user", "content": text[:1000]}  # Use first 1000 chars for classification
        ],
        temperature=0,
        max_tokens=10
    )
    
    result = response.choices[0].message.content.strip()
    if result == "PITCH":
        return InputType.PITCH
    elif result == "CHAT":
        return InputType.CHAT
    else:
        return InputType.RANDOM

def analyze_pitch(text: str) -> dict:
    """Analyze investor pitch/call transcript."""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": """You are an expert investor call analyst. Your task is to analyze transcripts and provide clear, direct insights.
            Always format your response exactly as requested, with each section clearly labeled and followed by your analysis."""},
            {"role": "user", "content": f"""Analyze this investor call transcript and provide the following insights:

            **Key Strength:**
            Identify the single most important positive point from the call.

            **Key Weakness:**
            Point out the most critical concern or weakness discussed.

            **Investor Impression:**
            Provide a one-line summary of how investors would likely perceive this call.

            **Missed Opportunity:**
            Identify one key opportunity that wasn't fully addressed or leveraged.

            **Confidence:**
            Rate your confidence in this analysis as Low/Medium/High based on the transcript quality.

            **Summary:**
            Provide a comprehensive 2-3 sentence summary that ties all insights together.

            Transcript:
            {text}"""}
        ],
        temperature=0.7,
    )
    
    return parse_response(response.choices[0].message.content)

def analyze_chat(text: str) -> dict:
    """Analyze general conversation."""
    return {
        "input_type": InputType.CHAT,
        "message": "This appears to be a general conversation or chat. Please provide an investor pitch or earnings call transcript for detailed analysis."
    }

def handle_random_input(text: str) -> dict:
    """Handle random/unclassified input."""
    return {
        "input_type": InputType.RANDOM,
        "message": "This doesn't appear to be a startup pitch or earnings call transcript. Please provide a relevant transcript for detailed analysis."
    }

app = FastAPI()

# Configure CORS
origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://127.0.0.1",
    "http://127.0.0.1:8000",
    "null",  # For local file access
    "*"      # Allow all origins for testing
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_root():
    return FileResponse("static/index.html")

def parse_response(content: str) -> dict:
    """Parse the response content into structured data."""
    # Remove markdown asterisks
    content = content.replace('**', '')
    
    # Initialize the result dictionary
    result = {
        'key_strength': '',
        'key_weakness': '',
        'investor_impression': '',
        'missed_opportunity': '',
        'confidence_rating': '',
        'final_summary': ''
    }
    
    # Split content into sections
    sections = content.split('\n\n')
    
    for section in sections:
        section = section.strip()
        if not section:
            continue
            
        # Extract section name and content
        if ':' in section:
            section_name, section_content = section.split(':', 1)
            section_name = section_name.strip().lower()
            section_content = section_content.strip()
            
            if 'key strength' in section_name:
                result['key_strength'] = section_content
            elif 'key weakness' in section_name:
                result['key_weakness'] = section_content
            elif 'investor impression' in section_name:
                result['investor_impression'] = section_content
            elif 'missed opportunity' in section_name:
                result['missed_opportunity'] = section_content
            elif 'confidence' in section_name:
                result['confidence_rating'] = section_content
            elif 'summary' in section_name:
                result['final_summary'] = section_content
    
    # Set default values for empty fields
    for key in result:
        if not result[key]:
            result[key] = f"No {key.replace('_', ' ')} provided"
    
    return result

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_transcript(request: TranscriptRequest):
    try:
        if not request.text:
            raise HTTPException(status_code=400, detail="Transcript text is required")

        # Classify input
        input_type = classify_input(request.text)
        
        # Route to appropriate analyzer
        if input_type == InputType.PITCH:
            result = analyze_pitch(request.text)
            result["input_type"] = input_type
            return AnalysisResponse(**result)
        
        elif input_type == InputType.CHAT:
            result = analyze_chat(request.text)
            return AnalysisResponse(**result)
        
        else:
            result = handle_random_input(request.text)
            return AnalysisResponse(**result)

    except Exception as e:
        print(f"Error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"detail": str(e)}
        )

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    print("Server starting at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000) 