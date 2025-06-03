from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
from openai import OpenAI
from dotenv import load_dotenv
import os
import re

# Load environment variables from .env file
load_dotenv()

# Create a single client instance
client = OpenAI()

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

class TranscriptRequest(BaseModel):
    text: str

class Insight(BaseModel):
    key_strength: str
    key_weakness: str
    investor_impression: str
    missed_opportunity: str
    confidence_rating: str
    final_summary: str

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

@app.post("/analyze", response_model=Insight)
async def analyze_transcript(request: TranscriptRequest):
    try:
        if not request.text:
            raise HTTPException(status_code=400, detail="Transcript text is required")

        system_prompt = """You are an expert investor call analyst. Your task is to analyze transcripts and provide clear, direct insights.
Always format your response exactly as requested, with each section clearly labeled and followed by your analysis.
Be specific and actionable in your insights."""

        user_prompt = f"""Analyze this investor call transcript and provide the following insights:

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
{request.text}"""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
        )

        # Get the response content
        content = response.choices[0].message.content
        print(f"Raw GPT Response:\n{content}")  # Debug print
        
        # Parse the response
        parsed_response = parse_response(content)
        print(f"Parsed Response:\n{parsed_response}")  # Debug print
        
        return Insight(**parsed_response)

    except Exception as e:
        print(f"Error: {str(e)}")  # Log the error
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