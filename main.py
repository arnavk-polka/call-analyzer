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
from modules.conversation_preprocessor import ConversationPreprocessor
from modules.memory_module import memory_system

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
    FOUNDER_RESPONSE = "founder_response"
    INVESTOR_RESPONSE = "investor_response"
    CHAT = "chat"
    RANDOM = "random"

class TranscriptRequest(BaseModel):
    text: str
    type: Literal["pitch", "founder_response", "investor_response", "communication"] = "pitch"

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

class MemoryInsights(BaseModel):
    insights: List[str] = []
    comparisons: List[str] = []
    recommendations: List[str] = []

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
    memory_insights: Optional[MemoryInsights] = None

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
    print(f"=== CLASSIFY_INPUT DEBUG ===")
    print(f"Classifying first 1000 chars: {text[:1000]}...")
    
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
    print(f"OpenAI classification result: '{result}'")
    
    if result == "PITCH":
        print("Classified as PITCH")
        return InputType.PITCH
    elif result == "CHAT":
        print("Classified as CHAT")
        return InputType.CHAT
    else:
        print(f"Classified as RANDOM (original result: '{result}')")
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
        
        # Get historical context from mem0 before analysis
        print("🔍 Searching mem0 for similar pitch analyses...")
        historical_context = ""
        try:
            # Extract key info for context search
            text_preview = text[:500]  # First 500 chars for context search
            context_memories = memory_system.search_context(f"Similar pitch analysis: {text_preview}", limit=3)
            
            if context_memories:
                historical_context = f"\n\n**Historical Context from Similar Analyses:**\n"
                for i, memory in enumerate(context_memories[:2], 1):  # Use top 2 results
                    historical_context += f"{i}. {memory}\n"
                print(f"🧠 Found {len(context_memories)} similar pitch analyses in memory")
            else:
                print(f"🧠 No similar pitch analyses found in memory")
        except Exception as e:
            print(f"⚠️ Could not retrieve pitch analysis context: {e}")

        # Then get the detailed analysis with context
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": """You are an expert investor call analyst with access to historical analysis data. Your task is to analyze transcripts and provide clear, direct insights.
                Use any provided historical context to inform your analysis, but focus primarily on the current transcript.
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
                {historical_context}

                **Current Transcript to Analyze:**
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
    debug_storage['last_analysis']['token_usage'] = {}
    
    try:
        # Get historical context from mem0 before analysis
        print("🔍 Searching mem0 for investor response patterns...")
        try:
            text_preview = text[:300]
            context_memories = memory_system.search_context(f"Investor feedback and questions: {text_preview}", limit=3)
            
            if context_memories:
                print(f"🧠 Found {len(context_memories)} similar investor responses in memory")
            else:
                print(f"🧠 No similar investor responses found in memory")
        except Exception as e:
            print(f"⚠️ Could not retrieve investor response context: {e}")
        
        # Analyze investor response
        investor_response_module = InvestorResponseModule(client, memory_system)
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
        audit_results, audit_tokens = auditor.audit_analysis({'investor_response': result['investor_response']})
        debug_storage['last_analysis']['token_usage']['auditor'] = audit_tokens
        
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

def analyze_founder_response(text: str) -> dict:
    """Analyze founder response to investor questions/concerns."""
    # Store input for debugging
    debug_storage['last_analysis']['input_text'] = text
    debug_storage['last_analysis']['input_type'] = 'FOUNDER_RESPONSE'
    debug_storage['last_analysis']['timestamp'] = datetime.now().isoformat()
    debug_storage['last_analysis']['token_usage'] = {}
    
    try:
        # Get historical context for founder responses
        historical_context = ""
        try:
            context_memories = memory_system.search_context(f"Founder response patterns: {text[:300]}", limit=2)
            if context_memories:
                historical_context = f"\n\n**Historical Context from Similar Founder Responses:**\n"
                for i, memory in enumerate(context_memories, 1):
                    historical_context += f"{i}. {memory}\n"
                print(f"🧠 Using context from {len(context_memories)} similar founder responses")
        except Exception as e:
            print(f"⚠️ Could not retrieve founder response context: {e}")
        
        # Analyze founder response using specialized prompting with context
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": """You are an expert at analyzing founder responses to investor inquiries with access to historical patterns. 
                Use any provided historical context to understand common founder communication patterns.
                Evaluate how well the founder addresses concerns, communicates value, and builds investor confidence."""},
                {"role": "user", "content": f"""Analyze this founder response and provide insights:

                **Key Strength:**
                What did the founder do well in this response?

                **Key Weakness:**
                What could the founder have addressed better?

                **Investor Impression:**
                How would investors likely perceive this response?

                **Missed Opportunity:**
                What opportunity did the founder miss to strengthen their case?

                **Confidence:**
                Rate confidence in this analysis as Low/Medium/High.

                **Summary:**
                Provide a 2-3 sentence summary of the founder's communication effectiveness.
                {historical_context}

                **Current Founder Response to Analyze:**
                {text}"""}
            ],
            temperature=0.7,
        )
        
        # Track tokens
        track_token_usage('founder_response_analysis', response)
        
        result = parse_response(response.choices[0].message.content)
        result['input_type'] = InputType.FOUNDER_RESPONSE
        
        # Run audit on founder response analysis
        auditor = AuditorModule(client)
        audit_results, audit_tokens = auditor.audit_analysis(result)
        debug_storage['last_analysis']['token_usage']['auditor'] = audit_tokens
        
        # Store audit results for debugging
        debug_storage['last_analysis']['auditor_module'] = audit_results
        
        # Add audit results to final response
        result['audit_results'] = audit_results
        
        debug_storage['last_analysis']['final_analysis'] = result
        return result
        
    except Exception as e:
        print(f"Error in founder response analysis: {str(e)}")
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

        print(f"=== ANALYZE ENDPOINT DEBUG ===")
        print(f"Original request text length: {len(request.text)}")
        print(f"Request type: {request.type}")
        print(f"Text preview: {request.text[:200]}...")

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
        elif request.type == "founder_response":
            print("Processing as founder response")
            result = analyze_founder_response(request.text)
            return AnalysisResponse(**result)
        else:
            print("Processing with advanced preprocessing")
            print(f"Text length before preprocessing: {len(request.text)}")
            
            try:
                # Use advanced preprocessing to detect conversation structure
                preprocessor = ConversationPreprocessor(client)
                preprocessing_result = preprocessor.preprocess_conversation(request.text)
                
                print(f"Preprocessing successful. Speakers: {preprocessing_result['conversation_summary']['speakers_present']}")
                print(f"Phases: {preprocessing_result['conversation_summary']['phases_present']}")
                print(f"Number of blocks: {len(preprocessing_result['thematic_blocks'])}")
                
                # Debug: Print each block's text length
                for i, block in enumerate(preprocessing_result['thematic_blocks']):
                    print(f"Block {i+1} text length: {len(block['text'])}, preview: {block['text'][:100]}...")
                
                speakers = preprocessing_result["conversation_summary"]["speakers_present"]
                phases = preprocessing_result["conversation_summary"]["phases_present"]
                
                # Determine analysis type based on sophisticated preprocessing
                if len(preprocessing_result["thematic_blocks"]) > 1:
                    # Multi-block conversation - use conversation analysis
                    print("Detected multi-block conversation, using conversation analysis")
                    analyzed_blocks = []
                    
                    for i, block in enumerate(preprocessing_result["thematic_blocks"]):
                        print(f"Analyzing block {i+1}: speaker={block['speaker']}, phase={block['phase']}")
                        print(f"Block text preview: {block['text'][:100]}...")
                        
                        try:
                            block_analysis = await analyze_conversation_block(block)
                            analyzed_blocks.append(block_analysis)
                            print(f"Block {i+1} analysis successful")
                        except Exception as block_error:
                            print(f"Error analyzing block {i+1}: {block_error}")
                            import traceback
                            traceback.print_exc()
                            # Add a fallback analysis for this block
                            analyzed_blocks.append({
                                "block_info": block,
                                "analysis_results": {"type": "error", "message": str(block_error)}
                            })
                    
                    print(f"Completed analysis of {len(analyzed_blocks)} blocks")
                    
                    # Extract the most relevant analysis for the response
                    try:
                        primary_analysis = extract_primary_analysis(analyzed_blocks)
                        primary_analysis["input_type"] = InputType.CHAT
                        print(f"Primary analysis extracted successfully")
                        print(f"Primary analysis keys: {list(primary_analysis.keys())}")
                        
                        # Debug the AnalysisResponse creation
                        print(f"=== CREATING ANALYSIS RESPONSE ===")
                        print(f"input_type: {primary_analysis.get('input_type')}")
                        print(f"investor_response type: {type(primary_analysis.get('investor_response'))}")
                        print(f"investor_response value: {primary_analysis.get('investor_response')}")
                        
                        # Validate and convert investor_response if needed
                        if primary_analysis.get('investor_response'):
                            print(f"Converting investor_response to InvestorResponseData")
                            try:
                                investor_data = primary_analysis['investor_response']
                                primary_analysis['investor_response'] = InvestorResponseData(**investor_data)
                                print(f"Successfully converted investor_response")
                            except Exception as conversion_error:
                                print(f"Error converting investor_response: {conversion_error}")
                                # Set to None if conversion fails
                                primary_analysis['investor_response'] = None
                        
                        try:
                            response_obj = AnalysisResponse(**primary_analysis)
                            print(f"AnalysisResponse created successfully")
                            print(f"Response investor_response: {response_obj.investor_response}")
                            print(f"=== ABOUT TO RETURN RESPONSE ===")
                            print(f"Response type: {type(response_obj)}")
                            print(f"Response input_type: {response_obj.input_type}")
                            print(f"Response message: {response_obj.message}")
                            print(f"=== SUCCESSFULLY RETURNING RESPONSE ===")
                            return response_obj
                        except Exception as response_error:
                            print(f"!!! ERROR CREATING ANALYSISRESPONSE: {response_error}")
                            print(f"Primary analysis keys: {list(primary_analysis.keys())}")
                            print(f"Primary analysis investor_response type: {type(primary_analysis.get('investor_response'))}")
                            import traceback
                            traceback.print_exc()
                            
                            # Don't raise - create a basic fallback response instead
                            print("Creating fallback response due to AnalysisResponse error")
                            fallback_response = {
                                "input_type": InputType.CHAT,
                                "message": f"Analysis completed but response formatting failed: {str(response_error)}",
                                "key_strength": primary_analysis.get("key_strength", "Analysis completed"),
                                "key_weakness": primary_analysis.get("key_weakness", "Technical formatting issue"),
                                "investor_impression": primary_analysis.get("investor_impression", "Mixed results"),
                                "missed_opportunity": primary_analysis.get("missed_opportunity", "Response formatting limitation"),
                                "confidence_rating": "Low",
                                "final_summary": "Analysis completed with technical issues"
                            }
                            return AnalysisResponse(**fallback_response)
                            
                    except Exception as extract_error:
                        print(f"Error in extract_primary_analysis: {extract_error}")
                        import traceback
                        traceback.print_exc()
                        
                        # Manual fallback - create a basic response
                        fallback_response = {
                            "input_type": InputType.CHAT,
                            "message": "Multi-block conversation analysis completed with partial results",
                            "key_strength": "Multi-speaker conversation detected",
                            "key_weakness": "Analysis encountered technical issues",
                            "investor_impression": "Mixed conversation analysis",
                            "missed_opportunity": "Technical analysis limitations",
                            "confidence_rating": "Low",
                            "final_summary": f"Analyzed {len(analyzed_blocks)} conversation blocks"
                        }
                        return AnalysisResponse(**fallback_response)
                    
                else:
                    # Single block - route to appropriate analyzer
                    single_block = preprocessing_result["thematic_blocks"][0]
                    speaker = single_block["speaker"]
                    phase = single_block["phase"]
                    
                    print(f"Single block detected: {speaker} in {phase} phase")
                    
                    if speaker == "founder" and phase == "pitch":
                        print("Routing to pitch analyzer")
                        result = analyze_pitch(request.text)
                        result["input_type"] = InputType.PITCH
                        return AnalysisResponse(**result)
                        
                    elif speaker == "investor":
                        print("Routing to investor response analyzer")
                        result = analyze_investor_response(request.text)
                        return AnalysisResponse(**result)
                        
                    elif speaker == "founder" and phase in ["qa", "objections"]:
                        print("Routing to founder response analyzer")
                        result = analyze_founder_response(request.text)
                        return AnalysisResponse(**result)
                        
                    elif speaker == "founder":
                        print("Routing to pitch analyzer (founder content)")
                        result = analyze_pitch(request.text)
                        result["input_type"] = InputType.PITCH
                        return AnalysisResponse(**result)
                        
                    else:
                        # Fallback to old classification method
                        print(f"No specific routing for speaker={speaker}, phase={phase}. Using fallback classification")
                        input_type = classify_input(request.text)
                        if input_type == InputType.PITCH:
                            result = analyze_pitch(request.text)
                            result["input_type"] = input_type
                            return AnalysisResponse(**result)
                        elif input_type == InputType.FOUNDER_RESPONSE:
                            result = analyze_founder_response(request.text)
                            return AnalysisResponse(**result)
                        elif input_type == InputType.INVESTOR_RESPONSE:
                            result = analyze_investor_response(request.text)
                            return AnalysisResponse(**result)
                        elif input_type == InputType.CHAT:
                            result = analyze_chat(request.text)
                            return AnalysisResponse(**result)
                        else:
                            result = handle_random_input(request.text)
                            return AnalysisResponse(**result)
                            
            except Exception as e:
                print(f"Error in advanced preprocessing: {e}")
                import traceback
                traceback.print_exc()
                
                # Fallback to old classification method
                print("Using fallback classification due to preprocessing error")
                input_type = classify_input(request.text)
                if input_type == InputType.PITCH:
                    result = analyze_pitch(request.text)
                    result["input_type"] = input_type
                    return AnalysisResponse(**result)
                elif input_type == InputType.FOUNDER_RESPONSE:
                    result = analyze_founder_response(request.text)
                    return AnalysisResponse(**result)
                elif input_type == InputType.INVESTOR_RESPONSE:
                    result = analyze_investor_response(request.text)
                    return AnalysisResponse(**result)
                elif input_type == InputType.CHAT:
                    result = analyze_chat(request.text)
                    return AnalysisResponse(**result)
                else:
                    result = handle_random_input(request.text)
                    return AnalysisResponse(**result)

    except Exception as e:
        print(f"!!! OUTER EXCEPTION IN ANALYZE_TRANSCRIPT: {str(e)}")
        print(f"Exception type: {type(e)}")
        import traceback
        print(traceback.format_exc())
        
        # Log the request details for debugging
        print(f"Request text length: {len(request.text) if request.text else 0}")
        print(f"Request type: {request.type}")
        
        return JSONResponse(
            status_code=500,
            content={"detail": f"Server error: {str(e)}"}
        )

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/debug")
async def debug_ui():
    """Serve the debug UI HTML interface."""
    return FileResponse("static/debug.html")

@app.get("/debug/data")
async def debug_data():
    """Get debug data for the debug UI."""
    return debug_storage['last_analysis']

@app.get("/memory/stats")
async def memory_stats():
    """Get memory system statistics."""
    return await memory_system.get_memory_stats()

@app.get("/memory/insights/{query}")
async def get_memory_insights(query: str):
    """Get insights from memory based on query."""
    try:
        insights = memory_system.get_contextual_insights({"key_strength": query})
        return insights
    except Exception as e:
        return {"error": str(e), "insights": [], "comparisons": [], "recommendations": []}

@app.get("/memory/search/{query}")
async def search_memory(query: str, limit: int = 5):
    """Search mem0 for relevant past analyses"""
    try:
        if not memory_system.enabled:
            return {"error": "Memory system is not enabled", "memories": []}
        
        memories = memory_system.search_context(query, limit=limit)
        return {
            "query": query,
            "limit": limit,
            "found": len(memories),
            "memories": memories
        }
    except Exception as e:
        return {"error": f"Failed to search memory: {str(e)}", "memories": []}

def preprocess_and_suggest_type(text: str) -> dict:
    """Preprocess text using the advanced conversation preprocessor."""
    
    try:
        # Use the new conversation preprocessor
        preprocessor = ConversationPreprocessor(client)
        result = preprocessor.preprocess_conversation(text)
        
        # Add backward compatibility for the old format
        # Determine primary suggested type based on conversation analysis
        speakers = result["conversation_summary"]["speakers_present"]
        phases = result["conversation_summary"]["phases_present"]
        
        if "founder" in speakers and "investor" in speakers:
            # Multi-speaker conversation
            if "pitch" in phases:
                suggested_type = InputType.PITCH
            elif "qa" in phases or "objections" in phases:
                suggested_type = InputType.CHAT  # Mixed conversation
            else:
                suggested_type = InputType.CHAT
        elif "founder" in speakers:
            if "pitch" in phases:
                suggested_type = InputType.PITCH
            else:
                suggested_type = InputType.FOUNDER_RESPONSE
        elif "investor" in speakers:
            suggested_type = InputType.INVESTOR_RESPONSE
        else:
            # Fall back to old classification method
            suggested_type = classify_input(text)
        
        # Add the new preprocessing results
        result["suggested_type"] = suggested_type
        result["legacy_pattern_suggestion"] = suggested_type  # For backward compatibility
        result["text_preview"] = text[:200] + "..." if len(text) > 200 else text
        
        return result
        
    except Exception as e:
        print(f"Error in advanced preprocessing: {e}")
        # Fallback to simple preprocessing
        return simple_preprocess_and_suggest_type(text)

def simple_preprocess_and_suggest_type(text: str) -> dict:
    """Simple preprocessing function as fallback."""
    
    # Quick pattern matching for common indicators
    founder_indicators = [
        "our business model", "our revenue", "we've grown", "our customers",
        "let me explain", "to address your concern", "our metrics show",
        "we're seeing", "our strategy", "we believe", "our team has"
    ]
    
    investor_indicators = [
        "what about", "how do you", "i'm concerned about", "can you explain",
        "what's your", "how will you", "what if", "i'd like to understand",
        "my question is", "i'm wondering", "what are your thoughts on",
        "a couple of questions", "what kind of", "any early", "how long is",
        "what key milestones", "thanks for", "that was compelling",
        "i really like", "couple of questions", "what milestones",
        "projected runway", "traction have you seen", "partnerships",
        "compelling pitch", "questions:", "pilots or"
    ]
    
    pitch_indicators = [
        "today i'll be presenting", "our company", "market opportunity",
        "our solution", "business model", "financial projections",
        "funding ask", "use of funds", "competitive advantage"
    ]
    
    text_lower = text.lower()
    
    founder_score = sum(1 for indicator in founder_indicators if indicator in text_lower)
    investor_score = sum(1 for indicator in investor_indicators if indicator in text_lower)
    pitch_score = sum(1 for indicator in pitch_indicators if indicator in text_lower)
    
    # Use GPT classification as primary method
    suggested_type = classify_input(text)
    
    # Pattern matching as secondary validation
    pattern_suggestion = None
    if founder_score > investor_score and founder_score > pitch_score:
        pattern_suggestion = InputType.FOUNDER_RESPONSE
    elif investor_score > founder_score and investor_score > pitch_score:
        pattern_suggestion = InputType.INVESTOR_RESPONSE
    elif pitch_score > 0:
        pattern_suggestion = InputType.PITCH
    
    return {
        "suggested_type": suggested_type,
        "pattern_suggestion": pattern_suggestion,
        "confidence_scores": {
            "founder_indicators": founder_score,
            "investor_indicators": investor_score,
            "pitch_indicators": pitch_score
        },
        "text_preview": text[:200] + "..." if len(text) > 200 else text
    }

@app.post("/analyze-conversation")
async def analyze_conversation(request: TranscriptRequest):
    """Advanced conversation analysis with role separation and thematic chunking."""
    try:
        if not request.text:
            raise HTTPException(status_code=400, detail="Conversation text is required")
        
        # Use the advanced conversation preprocessor
        preprocessor = ConversationPreprocessor(client)
        conversation_analysis = preprocessor.preprocess_conversation(request.text)
        
        # Analyze each thematic block
        analyzed_blocks = []
        for block in conversation_analysis["thematic_blocks"]:
            block_analysis = await analyze_conversation_block(block)
            analyzed_blocks.append(block_analysis)
        
        return {
            "conversation_structure": conversation_analysis,
            "analyzed_blocks": analyzed_blocks,
            "overall_insights": generate_conversation_insights(analyzed_blocks)
        }
        
    except Exception as e:
        print(f"Error in analyze_conversation: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"detail": str(e)}
        )

async def analyze_conversation_block(block: dict) -> dict:
    """Analyze a single conversation block based on its phase and speaker."""
    
    phase = block["phase"]
    speaker = block["speaker"] 
    text = block["text"]
    
    print(f"=== ANALYZE_CONVERSATION_BLOCK DEBUG ===")
    print(f"Block phase: {phase}, speaker: {speaker}")
    print(f"Block text length: {len(text)}")
    print(f"Block text preview: {text[:150]}...")
    
    analysis = {
        "block_info": block,
        "analysis_results": {}
    }
    
    try:
        if phase == "pitch" and speaker == "founder":
            # Analyze as pitch
            print(f"Calling analyze_pitch with text length: {len(text)}")
            result = analyze_pitch(text)
            analysis["analysis_results"] = result
        elif speaker == "investor":
            # Analyze as investor response
            print(f"Calling InvestorResponseModule with text length: {len(text)}")
            investor_response_module = InvestorResponseModule(client, memory_system)
            result = investor_response_module.analyze(text)
            
            # Store in debug_storage for debugging visibility
            debug_storage['last_analysis']['investor_response_module'] = result
            
            analysis["analysis_results"] = {
                "type": "investor_response",
                "data": result
            }
        elif speaker == "founder" and phase in ["qa", "objections"]:
            # Analyze as founder response
            print(f"Calling analyze_founder_response with text length: {len(text)}")
            result = analyze_founder_response(text)
            analysis["analysis_results"] = result
        else:
            # General analysis
            print(f"Using general analysis for speaker={speaker}, phase={phase}")
            analysis["analysis_results"] = {
                "type": "general",
                "summary": f"General {phase} content from {speaker}"
            }
            
    except Exception as e:
        print(f"Error analyzing block: {e}")
        analysis["analysis_results"] = {
            "type": "error",
            "message": str(e)
        }
    
    return analysis

def generate_conversation_insights(analyzed_blocks: List[dict]) -> dict:
    """Generate overall insights from all analyzed conversation blocks."""
    
    insights = {
        "conversation_quality": "Medium",
        "key_themes": [],
        "investor_engagement": "Unknown",
        "founder_performance": "Unknown",
        "next_steps_suggested": [],
        "red_flags": [],
        "positive_signals": []
    }
    
    # Extract insights from blocks
    for block in analyzed_blocks:
        if block["block_info"]["speaker"] == "investor":
            # Extract investor insights
            if "data" in block["analysis_results"]:
                data = block["analysis_results"]["data"]
                insights["positive_signals"].extend(data.get("interest_signals", []))
                insights["red_flags"].extend(data.get("objections", []))
        
        elif block["block_info"]["speaker"] == "founder":
            # Extract founder insights
            if "key_strength" in block["analysis_results"]:
                insights["positive_signals"].append(block["analysis_results"]["key_strength"])
            if "key_weakness" in block["analysis_results"]:
                insights["red_flags"].append(block["analysis_results"]["key_weakness"])
    
    return insights

def extract_primary_analysis(analyzed_blocks: List[dict]) -> dict:
    """Extract the most relevant analysis from multiple conversation blocks."""
    
    print(f"=== EXTRACT_PRIMARY_ANALYSIS DEBUG ===")
    print(f"Number of analyzed blocks: {len(analyzed_blocks)}")
    
    primary_analysis = {
        "key_strength": "",
        "key_weakness": "",
        "investor_impression": "",
        "missed_opportunity": "",
        "confidence_rating": "Medium",
        "final_summary": "",
        "message": "Multi-block conversation analysis completed",
        "pitch_understanding": {},
        "investor_response": None,
        "communication_scores": None,
        "fundraising_analysis": None,
        "audit_results": None
    }
    
    # Collect insights from all blocks
    strengths = []
    weaknesses = []
    impressions = []
    opportunities = []
    
    investor_responses = {
        'objections': [],
        'questions': [],
        'interest_signals': [],
        'areas_of_skepticism': []
    }
    
    for i, block in enumerate(analyzed_blocks):
        print(f"\n--- Analyzing Block {i+1} ---")
        print(f"Block info: {block.get('block_info', {})}")
        
        analysis = block.get("analysis_results", {})
        print(f"Analysis keys: {list(analysis.keys())}")
        print(f"Analysis type: {analysis.get('type', 'unknown')}")
        
        # Extract standard analysis fields
        if "key_strength" in analysis and analysis["key_strength"]:
            print(f"Found key_strength: {analysis['key_strength'][:100]}...")
            strengths.append(analysis["key_strength"])
        if "key_weakness" in analysis and analysis["key_weakness"]:
            print(f"Found key_weakness: {analysis['key_weakness'][:100]}...")
            weaknesses.append(analysis["key_weakness"])
        if "investor_impression" in analysis and analysis["investor_impression"]:
            print(f"Found investor_impression: {analysis['investor_impression'][:100]}...")
            impressions.append(analysis["investor_impression"])
        if "missed_opportunity" in analysis and analysis["missed_opportunity"]:
            print(f"Found missed_opportunity: {analysis['missed_opportunity'][:100]}...")
            opportunities.append(analysis["missed_opportunity"])
            
        # Extract investor response data
        if analysis.get("type") == "investor_response" and "data" in analysis:
            print(f"Found investor response data!")
            data = analysis["data"]
            print(f"Investor data keys: {list(data.keys())}")
            investor_responses['objections'].extend(data.get('objections', []))
            investor_responses['questions'].extend(data.get('questions', []))
            investor_responses['interest_signals'].extend(data.get('interest_signals', []))
            investor_responses['areas_of_skepticism'].extend(data.get('areas_of_skepticism', []))
            print(f"Added {len(data.get('questions', []))} questions, {len(data.get('interest_signals', []))} signals")
        
        # Extract pitch understanding if available
        if "pitch_understanding" in analysis and analysis["pitch_understanding"]:
            print(f"Found pitch understanding data")
            primary_analysis["pitch_understanding"] = analysis["pitch_understanding"]
    
    # Combine insights
    primary_analysis["key_strength"] = "; ".join(strengths) if strengths else "Multiple positive aspects identified"
    primary_analysis["key_weakness"] = "; ".join(weaknesses) if weaknesses else "Several areas for improvement noted"
    primary_analysis["investor_impression"] = "; ".join(impressions) if impressions else "Mixed investor engagement observed"
    primary_analysis["missed_opportunity"] = "; ".join(opportunities) if opportunities else "Various optimization opportunities identified"
    
    # Add investor response data if available
    if any(investor_responses.values()):
        print(f"Adding investor response data to primary analysis")
        primary_analysis["investor_response"] = investor_responses
    else:
        print(f"No investor response data found")
    
    # Generate business-focused summary about the pitch content
    summary_parts = []
    
    # Determine conversation type and outcome
    has_investor_feedback = any(investor_responses.values())
    questions_count = len(investor_responses.get('questions', []))
    signals_count = len(investor_responses.get('interest_signals', []))
    objections_count = len(investor_responses.get('objections', []))
    
    # Start with pitch characterization
    if has_investor_feedback:
        if signals_count > objections_count:
            summary_parts.append("The pitch generated positive investor interest")
        elif objections_count > signals_count:
            summary_parts.append("The pitch raised several investor concerns")
        else:
            summary_parts.append("The pitch received mixed investor feedback")
    else:
        summary_parts.append("The pitch presentation covered key business fundamentals")
    
    # Add key strengths insight
    if strengths:
        # Extract the most compelling strength
        main_strength = strengths[0]  # Take first/primary strength
        if "unique" in main_strength.lower() or "competitive" in main_strength.lower():
            summary_parts.append("with strong competitive differentiation highlighted")
        elif "market" in main_strength.lower() or "opportunity" in main_strength.lower():
            summary_parts.append("with compelling market opportunity demonstrated")
        elif "team" in main_strength.lower() or "experience" in main_strength.lower():
            summary_parts.append("with strong team credentials presented")
        elif "traction" in main_strength.lower() or "growth" in main_strength.lower():
            summary_parts.append("with solid business traction evidenced")
        else:
            summary_parts.append("with notable business strengths demonstrated")
    
    # Add investor engagement characterization
    if has_investor_feedback:
        if questions_count >= 3:
            summary_parts.append(f"Investors showed high engagement with {questions_count} detailed questions")
        elif questions_count > 0:
            summary_parts.append(f"Investors showed interest with {questions_count} follow-up questions")
        
        # Add specific areas of investor focus
        if objections_count > 0:
            summary_parts.append("though some concerns were raised requiring further clarification")
    
    # Construct final business summary
    primary_analysis["final_summary"] = ". ".join(summary_parts) + "."
    
    print(f"\n=== FINAL PRIMARY ANALYSIS ===")
    print(f"Key strength: {primary_analysis['key_strength'][:100]}...")
    print(f"Investor response present: {primary_analysis['investor_response'] is not None}")
    if primary_analysis['investor_response']:
        print(f"Questions count: {len(primary_analysis['investor_response']['questions'])}")
        print(f"Interest signals count: {len(primary_analysis['investor_response']['interest_signals'])}")
    
    # Store analysis in memory and get contextual insights
    print(f"📝 Storing analysis in memory...")
    try:
        import asyncio
        
        # Store the analysis result (async)
        asyncio.create_task(memory_system.store_analysis_result(primary_analysis))
        
        # Get contextual insights from memory (async)
        try:
            contextual_insights = asyncio.run(memory_system.get_contextual_insights(primary_analysis))
            if contextual_insights and any(contextual_insights.values()):
                print(f"🧠 Retrieved contextual insights from memory")
                primary_analysis["memory_insights"] = contextual_insights
            else:
                print(f"🧠 No contextual insights available yet")
        except Exception as insight_error:
            print(f"🧠 Could not retrieve insights: {insight_error}")
            
    except Exception as e:
        print(f"❌ Memory operation failed: {e}")
    
    return primary_analysis

if __name__ == "__main__":
    import uvicorn
    import os
    
    print("🔍 Investor Analyzer Pro - Starting up...")
    
    # Check for required API keys
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found!")
        print("Please set your OpenAI API key:")
        print("  Windows: $env:OPENAI_API_KEY=\"your-key\"")
        print("  Linux/Mac: export OPENAI_API_KEY=\"your-key\"")
        exit(1)
    
    print("✅ OpenAI API key found")
    
    # Check for mem0 API key (optional)
    if os.getenv("MEM0_API_KEY"):
        print("✅ Mem0 API key found - Memory features enabled")
    else:
        print("⚠️ MEM0_API_KEY not found - Memory features disabled")
        print("  To enable: set MEM0_API_KEY=\"your-mem0-key\"")
    
    print("🚀 Server starting at http://localhost:8000")
    print("🔍 Debug panel available at http://localhost:8000/debug")
    
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000,
        reload=True,  # Enable auto-reload
        reload_dirs=["./"],  # Watch current directory
        reload_includes=["*.py"],  # Watch Python files
        reload_excludes=["__pycache__/*"]  # Exclude cache files
    ) 