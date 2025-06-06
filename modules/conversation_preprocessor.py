from typing import Dict, List, Tuple, Optional
from openai import OpenAI
from enum import Enum
import re

class SpeakerRole(str, Enum):
    FOUNDER = "founder"
    INVESTOR = "investor"
    MODERATOR = "moderator"
    UNKNOWN = "unknown"

class ConversationPhase(str, Enum):
    PITCH = "pitch"
    QA = "qa"
    OBJECTIONS = "objections"
    WRAP_UP = "wrap_up"

class ConversationBlock:
    def __init__(self, phase: ConversationPhase, speaker: SpeakerRole, text: str, start_idx: int, end_idx: int):
        self.phase = phase
        self.speaker = speaker
        self.text = text
        self.start_idx = start_idx
        self.end_idx = end_idx
        
    def to_dict(self):
        return {
            "phase": self.phase,
            "speaker": self.speaker,
            "text": self.text,
            "start_idx": self.start_idx,
            "end_idx": self.end_idx
        }

class ConversationPreprocessor:
    def __init__(self, client: OpenAI):
        self.client = client
        
    def preprocess_conversation(self, text: str) -> Dict:
        """Main preprocessing function that handles role assignment and thematic chunking."""
        
        # Step 1: Split into speakers if conversation format is detected
        speaker_segments = self._detect_and_split_speakers(text)
        
        # Step 2: Assign roles to each speaker segment
        role_assigned_segments = self._assign_speaker_roles(speaker_segments)
        
        # Step 3: Chunk into thematic blocks
        thematic_blocks = self._chunk_into_themes(role_assigned_segments)
        
        # Step 4: Generate analysis suggestions
        analysis_suggestions = self._generate_analysis_suggestions(thematic_blocks)
        
        return {
            "original_text": text,
            "speaker_segments": [seg.to_dict() for seg in role_assigned_segments],
            "thematic_blocks": [block.to_dict() for block in thematic_blocks],
            "analysis_suggestions": analysis_suggestions,
            "conversation_summary": self._generate_conversation_summary(thematic_blocks)
        }
    
    def _detect_and_split_speakers(self, text: str) -> List[Tuple[str, str]]:
        """Detect if text contains multiple speakers and split accordingly."""
        
        print(f"=== SPEAKER DETECTION DEBUG ===")
        print(f"Input text length: {len(text)}")
        print(f"Input text preview: {text[:300]}...")
        
        # More specific speaker indicators - handle cases where speaker is on separate line
        speaker_patterns = [
            r'^(Investor|Founder|Alex|CEO|CFO|Moderator|Host):\s*(.+)$',  # Specific roles/names with content
            r'^(Investor|Founder|Alex|CEO|CFO|Moderator|Host):\s*$',  # Specific roles/names on separate line
            r'^([A-Z][a-z]+)\s*:\s*(.+)$',  # Single capitalized word + colon (names) with content
            r'^([A-Z][a-z]+)\s*:\s*$',  # Single capitalized word + colon (names) on separate line
            r'^\[([A-Z][a-zA-Z\s]+)\]:\s*(.+)$',  # "[Speaker Name]: text"
        ]
        
        segments = []
        lines = text.split('\n')
        current_speaker = "Unknown"
        current_text = []
        
        print(f"Split into {len(lines)} lines")
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue
                
            print(f"Processing line {i+1}: '{line[:50]}...'")
                
            # Check for speaker patterns
            speaker_found = False
            for pattern in speaker_patterns:
                match = re.match(pattern, line)
                if match:
                    speaker_name = match.group(1).strip()
                    # Additional validation: avoid common false positives
                    false_positives = [
                        "The problem", "The solution", "The market", "The team", 
                        "The product", "The company", "Our strategy", "In summary"
                    ]
                    if speaker_name not in false_positives:
                        print(f"Found valid speaker pattern: {speaker_name}")
                        
                        # Save previous segment
                        if current_text:
                            segment_text = '\n'.join(current_text)
                            print(f"Saving segment for {current_speaker}, length: {len(segment_text)}")
                            segments.append((current_speaker, segment_text))
                        
                        # Start new segment
                        current_speaker = speaker_name
                        
                        # Check if there's content after the colon
                        if len(match.groups()) > 1 and match.group(2) and match.group(2).strip():
                            # Content on same line
                            current_text = [match.group(2).strip()]
                        else:
                            # Speaker on separate line, collect following lines
                            current_text = []
                            
                        speaker_found = True
                        break
                    else:
                        print(f"Skipping false positive: {speaker_name}")
            
            if not speaker_found:
                current_text.append(line)
                
            i += 1
        
        # Add final segment
        if current_text:
            segment_text = '\n'.join(current_text)
            print(f"Saving final segment for {current_speaker}, length: {len(segment_text)}")
            segments.append((current_speaker, segment_text))
        
        print(f"Detected {len(segments)} segments")
        for i, (speaker, text) in enumerate(segments):
            print(f"Segment {i+1}: {speaker} - {len(text)} chars - '{text[:100]}...'")
        
        # If no speakers detected, treat as single speaker
        if len(segments) <= 1:
            print("Only one segment detected, treating as single speaker")
            segments = [("Speaker", text)]
            
        return segments
    
    def _assign_speaker_roles(self, speaker_segments: List[Tuple[str, str]]) -> List[ConversationBlock]:
        """Assign roles (Founder/Investor/Moderator) to each speaker segment."""
        
        role_assigned = []
        
        for i, (speaker_name, segment_text) in enumerate(speaker_segments):
            # Use GPT to classify the role based on content
            role = self._classify_speaker_role(speaker_name, segment_text)
            
            block = ConversationBlock(
                phase=ConversationPhase.PITCH,  # Will be updated in chunking step
                speaker=role,
                text=segment_text,
                start_idx=i,
                end_idx=i
            )
            role_assigned.append(block)
        
        return role_assigned
    
    def _classify_speaker_role(self, speaker_name: str, text: str) -> SpeakerRole:
        """Use GPT and rules to classify speaker role."""
        
        # Rule-based classification first
        founder_indicators = [
            "our company", "our product", "we've built", "our team", "our revenue",
            "we're raising", "our business model", "we've seen", "our strategy",
            "let me explain", "to address your concern", "our metrics show"
        ]
        
        investor_indicators = [
            "what about", "how do you", "i'm concerned", "can you explain",
            "what's your", "compelling pitch", "a couple of questions",
            "what kind of traction", "how long is your runway", "what milestones"
        ]
        
        moderator_indicators = [
            "welcome everyone", "let's begin", "thank you for joining",
            "next question", "we have time for", "let's wrap up"
        ]
        
        text_lower = text.lower()
        
        founder_score = sum(1 for indicator in founder_indicators if indicator in text_lower)
        investor_score = sum(1 for indicator in investor_indicators if indicator in text_lower)
        moderator_score = sum(1 for indicator in moderator_indicators if indicator in text_lower)
        
        # If clear winner from rules, use that
        if founder_score > investor_score and founder_score > moderator_score and founder_score > 2:
            return SpeakerRole.FOUNDER
        elif investor_score > founder_score and investor_score > moderator_score and investor_score > 2:
            return SpeakerRole.INVESTOR
        elif moderator_score > 1:
            return SpeakerRole.MODERATOR
        
        # Otherwise use GPT for classification
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": """Classify the speaker role based on their content:
                    
                    FOUNDER: Presents company/product, explains business model, addresses concerns, provides metrics
                    INVESTOR: Asks questions, expresses concerns, provides feedback, discusses terms
                    MODERATOR: Facilitates discussion, manages time, introduces speakers
                    UNKNOWN: Cannot determine from content
                    
                    Consider both the speaker name and the content. Respond with only the role in caps."""},
                    {"role": "user", "content": f"Speaker: {speaker_name}\n\nContent: {text[:500]}"}
                ],
                temperature=0,
                max_tokens=10
            )
            
            result = response.choices[0].message.content.strip().upper()
            if result == "FOUNDER":
                return SpeakerRole.FOUNDER
            elif result == "INVESTOR":
                return SpeakerRole.INVESTOR
            elif result == "MODERATOR":
                return SpeakerRole.MODERATOR
            else:
                return SpeakerRole.UNKNOWN
                
        except Exception as e:
            print(f"Error in GPT role classification: {e}")
            return SpeakerRole.UNKNOWN
    
    def _chunk_into_themes(self, segments: List[ConversationBlock]) -> List[ConversationBlock]:
        """Chunk conversation into thematic blocks: Pitch, Q&A, Objections, Wrap-up."""
        
        if not segments:
            return []
        
        # Use GPT to analyze the conversation flow and identify phases
        conversation_text = "\n\n".join([f"{seg.speaker}: {seg.text}" for seg in segments])
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": """Analyze this conversation and identify the thematic phases:
                    
                    PITCH: Initial presentation, company overview, product demo
                    QA: Questions and answers, clarifications, discussions
                    OBJECTIONS: Concerns raised, pushback, challenging questions
                    WRAP_UP: Closing remarks, next steps, thank you messages
                    
                    For each segment, identify which phase it belongs to. Consider the flow and context."""},
                    {"role": "user", "content": f"""Analyze this conversation and assign each segment to a phase:
                    
                    {conversation_text[:2000]}
                    
                    Respond with segment numbers and phases:
                    Segment 1: PHASE
                    Segment 2: PHASE
                    etc."""}
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            phase_assignments = self._parse_phase_assignments(response.choices[0].message.content)
            
            # Apply phase assignments to segments
            for i, segment in enumerate(segments):
                if i < len(phase_assignments):
                    segment.phase = phase_assignments[i]
                else:
                    # Default assignment based on position and speaker
                    if i == 0 and segment.speaker == SpeakerRole.FOUNDER:
                        segment.phase = ConversationPhase.PITCH
                    elif segment.speaker == SpeakerRole.INVESTOR:
                        segment.phase = ConversationPhase.QA
                    else:
                        segment.phase = ConversationPhase.QA
                        
        except Exception as e:
            print(f"Error in thematic chunking: {e}")
            # Fallback: simple rule-based assignment
            for i, segment in enumerate(segments):
                if i == 0 and segment.speaker == SpeakerRole.FOUNDER:
                    segment.phase = ConversationPhase.PITCH
                elif segment.speaker == SpeakerRole.INVESTOR:
                    segment.phase = ConversationPhase.QA
                else:
                    segment.phase = ConversationPhase.QA
        
        return segments
    
    def _parse_phase_assignments(self, response_text: str) -> List[ConversationPhase]:
        """Parse GPT response to extract phase assignments."""
        assignments = []
        lines = response_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if 'PITCH' in line.upper():
                assignments.append(ConversationPhase.PITCH)
            elif 'QA' in line.upper() or 'Q&A' in line.upper():
                assignments.append(ConversationPhase.QA)
            elif 'OBJECTION' in line.upper():
                assignments.append(ConversationPhase.OBJECTIONS)
            elif 'WRAP' in line.upper():
                assignments.append(ConversationPhase.WRAP_UP)
        
        return assignments
    
    def _generate_analysis_suggestions(self, blocks: List[ConversationBlock]) -> Dict:
        """Generate suggestions for how to analyze each thematic block."""
        
        suggestions = {
            "pitch_blocks": [],
            "qa_blocks": [],
            "objection_blocks": [],
            "wrap_up_blocks": []
        }
        
        for block in blocks:
            block_info = {
                "speaker": block.speaker,
                "text_preview": block.text[:100] + "..." if len(block.text) > 100 else block.text,
                "suggested_analysis": self._get_analysis_suggestion(block.phase, block.speaker)
            }
            
            if block.phase == ConversationPhase.PITCH:
                suggestions["pitch_blocks"].append(block_info)
            elif block.phase == ConversationPhase.QA:
                suggestions["qa_blocks"].append(block_info)
            elif block.phase == ConversationPhase.OBJECTIONS:
                suggestions["objection_blocks"].append(block_info)
            elif block.phase == ConversationPhase.WRAP_UP:
                suggestions["wrap_up_blocks"].append(block_info)
        
        return suggestions
    
    def _get_analysis_suggestion(self, phase: ConversationPhase, speaker: SpeakerRole) -> str:
        """Get analysis suggestion based on phase and speaker."""
        
        if phase == ConversationPhase.PITCH and speaker == SpeakerRole.FOUNDER:
            return "Analyze pitch structure, value proposition, and presentation effectiveness"
        elif phase == ConversationPhase.QA and speaker == SpeakerRole.INVESTOR:
            return "Extract questions, concerns, and interest signals"
        elif phase == ConversationPhase.QA and speaker == SpeakerRole.FOUNDER:
            return "Analyze response quality and how well concerns are addressed"
        elif phase == ConversationPhase.OBJECTIONS:
            return "Identify objections, skepticism areas, and pushback"
        elif phase == ConversationPhase.WRAP_UP:
            return "Extract next steps, follow-up actions, and final impressions"
        else:
            return "General conversation analysis"
    
    def _generate_conversation_summary(self, blocks: List[ConversationBlock]) -> Dict:
        """Generate a summary of the conversation structure."""
        
        summary = {
            "total_blocks": len(blocks),
            "phases_present": list(set([block.phase for block in blocks])),
            "speakers_present": list(set([block.speaker for block in blocks])),
            "conversation_flow": [{"phase": block.phase, "speaker": block.speaker} for block in blocks]
        }
        
        return summary 