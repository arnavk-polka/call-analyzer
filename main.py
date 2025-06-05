from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Literal, Dict, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime
import os
import re
import logging
from enum import Enum
from modules.pitch_analyzer import PitchUnderstandingModule
from modules.investor_response_analyzer import InvestorResponseModule
from modules.communication_scorer import CommunicationScorerModule
from modules.fundraising_calibration import FundraisingCalibrationModule
from modules.auditor import AuditorModule

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
    type: Literal["pitch", "investor_response", "communication"] = "pitch"

class InvestorResponseData(BaseModel):
    objections: List[str] = []
    questions: List[str] = []
    interest_signals: List[str] = []
    areas_of_skepticism: List[str] = []

class CommunicationScores(BaseModel):
    scores: Dict[str, Dict[str, Any]]
    overall_assessment: str

class FundraisingAnalysis(BaseModel):
    funding_ask_clarity: Dict[str, Any]
    use_of_funds: Dict[str, Any]
    realism_confidence: Dict[str, Any]
    improvement_suggestions: List[str]
    overall_assessment: str

class AuditResults(BaseModel):
    scores: Dict[str, Any]
    flags: Dict[str, str]
    overall_quality: str
    sections_to_regenerate: List[str]
    audit_summary: str

class AnalysisResponse(BaseModel):
    input_type: InputType
    key_strength: str = ""
    key_weakness: str = ""
    investor_impression: str = ""
    missed_opportunity: str = ""
    confidence_rating: str = ""
    final_summary: str = ""
    message: str = ""
    pitch_understanding: Dict[str, str] = {}
    investor_response: Optional[InvestorResponseData] = None
    communication_scores: Optional[CommunicationScores] = None
    fundraising_analysis: Optional[FundraisingAnalysis] = None
    audit_results: Optional[AuditResults] = None

# Add debug storage
debug_storage = {
    'last_analysis': {
        'input_text': '',
        'input_type': '',
        'pitch_understanding_module': {},
        'investor_response_module': None,
        'communication_scorer_module': None,
        'fundraising_calibration_module': None,
        'auditor_module': None,
        'final_analysis': {},
        'timestamp': None,
        'token_usage': {}
    }
}

def track_token_usage(module_name: str, response) -> int:
    """Track token usage for a module and store in debug storage."""
    try:
        if hasattr(response, 'usage') and response.usage:
            tokens = response.usage.total_tokens
            debug_storage['last_analysis']['token_usage'][module_name] = tokens
            return tokens
        return 0
    except Exception as e:
        print(f"Error tracking tokens for {module_name}: {e}")
        return 0

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
    # Store input for debugging
    debug_storage['last_analysis']['input_text'] = text
    debug_storage['last_analysis']['input_type'] = 'PITCH'
    debug_storage['last_analysis']['timestamp'] = datetime.now().isoformat()
    debug_storage['last_analysis']['token_usage'] = {}
    
    try:
        # First, get the pitch understanding using CoT
        pitch_understanding_module = PitchUnderstandingModule(client)
        pitch_understanding, pitch_tokens = pitch_understanding_module.analyze(text)
        debug_storage['last_analysis']['token_usage']['pitch_understanding'] = pitch_tokens
        
        # Store pitch understanding for debugging
        debug_storage['last_analysis']['pitch_understanding_module'] = pitch_understanding
        
        # Then analyze communication style
        communication_scorer = CommunicationScorerModule(client)
        comm_analysis, comm_tokens = communication_scorer.analyze(text)
        debug_storage['last_analysis']['token_usage']['communication_scorer'] = comm_tokens
        
        # Store communication analysis for debugging
        debug_storage['last_analysis']['communication_scorer_module'] = {
            'scores': comm_analysis['scores'],
            'overall_assessment': comm_analysis['overall_assessment']
        }
        
        # Analyze fundraising ask using the "ask" section from pitch understanding
        fundraising_module = FundraisingCalibrationModule(client)
        ask_section = pitch_understanding.get('ask', '')
        fundraising_analysis, fundraising_tokens = fundraising_module.analyze(ask_section)
        if fundraising_analysis:  # Only track tokens if analysis was performed
            debug_storage['last_analysis']['token_usage']['fundraising_calibration'] = fundraising_tokens
        
        # Store fundraising analysis for debugging
        debug_storage['last_analysis']['fundraising_calibration_module'] = fundraising_analysis
        
        # Then get the detailed analysis
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
        
        # Track tokens for main analysis
        track_token_usage('main_analysis', response)
        
        result = parse_response(response.choices[0].message.content)
        result['pitch_understanding'] = pitch_understanding
        result['communication_scores'] = {
            'scores': comm_analysis['scores'],
            'overall_assessment': comm_analysis['overall_assessment']
        }
        
        # Add fundraising analysis if available
        if fundraising_analysis:
            result['fundraising_analysis'] = fundraising_analysis
        
        # Run audit on all analysis results
        auditor = AuditorModule(client)
        audit_results, audit_tokens = auditor.audit_analysis(result)
        debug_storage['last_analysis']['token_usage']['auditor'] = audit_tokens
        
        # Store audit results for debugging
        debug_storage['last_analysis']['auditor_module'] = audit_results
        
        # Add audit results to final response
        result['audit_results'] = audit_results
        
        # Check if any sections need regeneration
        if audit_results['sections_to_regenerate']:
            print(f"Audit flagged sections for regeneration: {audit_results['sections_to_regenerate']}")
            # TODO: Implement selective regeneration logic here
        
        debug_storage['last_analysis']['final_analysis'] = result
        return result
        
    except Exception as e:
        print(f"Error in pitch analysis: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise

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

def analyze_investor_response(text: str) -> dict:
    """Analyze investor response."""
    # Store input for debugging
    debug_storage['last_analysis']['input_text'] = text
    debug_storage['last_analysis']['input_type'] = 'INVESTOR_RESPONSE'
    debug_storage['last_analysis']['timestamp'] = datetime.now().isoformat()
    
    try:
        # Analyze investor response
        investor_response_module = InvestorResponseModule(client)
        response_analysis = investor_response_module.analyze(text)
        
        # Store analysis for debugging
        debug_storage['last_analysis']['investor_response_module'] = response_analysis
        
        # Format the response to match AnalysisResponse model
        result = {
            "input_type": InputType.CHAT,
            "message": "Investor response analysis completed",
            "investor_response": {
                'objections': response_analysis['objections'],
                'questions': response_analysis['questions'],
                'interest_signals': response_analysis['interest_signals'],
                'areas_of_skepticism': response_analysis['areas_of_skepticism']
            }
        }
        
        # Run audit on investor response analysis
        auditor = AuditorModule(client)
        audit_results = auditor.audit_analysis({'investor_response': result['investor_response']})
        
        # Store audit results for debugging
        debug_storage['last_analysis']['auditor_module'] = audit_results
        
        # Add audit results to final response
        result['audit_results'] = audit_results
        
        debug_storage['last_analysis']['final_analysis'] = result
        return result
        
    except Exception as e:
        print(f"Error in investor response analysis: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise

def analyze_communication(text: str) -> dict:
    """Analyze communication style."""
    # Store input for debugging
    debug_storage['last_analysis']['input_text'] = text
    debug_storage['last_analysis']['input_type'] = 'COMMUNICATION'
    debug_storage['last_analysis']['timestamp'] = datetime.now().isoformat()
    
    try:
        # Analyze communication
        communication_scorer = CommunicationScorerModule(client)
        comm_analysis = communication_scorer.analyze(text)
        
        # Store analysis for debugging
        debug_storage['last_analysis']['communication_scorer_module'] = {
            'scores': comm_analysis['scores'],
            'overall_assessment': comm_analysis['overall_assessment']
        }
        
        # Format the response to match AnalysisResponse model
        result = {
            "input_type": InputType.CHAT,
            "message": "Communication style analysis completed",
            "communication_scores": {
                'scores': comm_analysis['scores'],
                'overall_assessment': comm_analysis['overall_assessment']
            }
        }
        
        debug_storage['last_analysis']['final_analysis'] = result
        return result
    except Exception as e:
        print(f"Error in communication analysis: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise

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

        # Reset debug storage for new request
        debug_storage['last_analysis'] = {
            'input_text': request.text,
            'input_type': request.type,
            'pitch_understanding_module': {},
            'investor_response_module': None,
            'communication_scorer_module': None,
            'fundraising_calibration_module': None,
            'auditor_module': None,
            'final_analysis': {},
            'timestamp': datetime.now().isoformat()
        }

        print(f"Processing request type: {request.type}")

        if request.type == "communication":
            print("Processing as communication analysis")
            result = analyze_communication(request.text)
            print(f"Communication analysis result: {result}")
            return AnalysisResponse(**result)
        elif request.type == "investor_response":
            print("Processing as investor response")
            result = analyze_investor_response(request.text)
            return AnalysisResponse(**result)
        else:
            print("Processing as pitch")
            input_type = classify_input(request.text)
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
        print(f"Error in analyze_transcript: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"detail": str(e)}
        )

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/debug")
async def debug_panel():
    """Debug panel to show intermediate values from each module."""
    return debug_storage['last_analysis']

@app.get("/debug-ui")
async def debug_ui():
    """Serve the debug UI HTML interface."""
    return FileResponse("static/debug.html")

if __name__ == "__main__":
    import uvicorn
    print("Server starting at http://localhost:8000")
    print("Debug panel available at http://localhost:8000/debug")
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000,
        reload=True,  # Enable auto-reload
        reload_dirs=["./"],  # Watch current directory
        reload_includes=["*.py"],  # Watch Python files
        reload_excludes=["__pycache__/*"]  # Exclude cache files
    ) 