from typing import Dict, List, Optional, Any, Tuple
from openai import OpenAI
import json

class AuditorModule:
    def __init__(self, client: OpenAI):
        self.client = client
        self.score_threshold = 3  # Minimum acceptable score (1-5)
        
    def audit_analysis(self, analysis_results: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
        """
        Audit the complete analysis results from all modules.
        Returns tuple of (audit_results, total_token_count).
        """
        audit_results = {
            'scores': {},
            'flags': {},
            'overall_quality': 'good',
            'sections_to_regenerate': [],
            'audit_summary': ''
        }
        
        total_tokens = 0
        
        # Audit each module's output
        if 'pitch_understanding' in analysis_results:
            scores, tokens = self._audit_pitch_understanding(analysis_results['pitch_understanding'])
            audit_results['scores']['pitch_understanding'] = scores
            total_tokens += tokens
            
        if 'communication_scores' in analysis_results:
            scores, tokens = self._audit_communication_scores(analysis_results['communication_scores'])
            audit_results['scores']['communication_scores'] = scores
            total_tokens += tokens
            
        if 'fundraising_analysis' in analysis_results and analysis_results['fundraising_analysis']:
            scores, tokens = self._audit_fundraising_analysis(analysis_results['fundraising_analysis'])
            audit_results['scores']['fundraising_analysis'] = scores
            total_tokens += tokens
            
        if 'investor_response' in analysis_results and analysis_results['investor_response']:
            scores, tokens = self._audit_investor_response(analysis_results['investor_response'])
            audit_results['scores']['investor_response'] = scores
            total_tokens += tokens
        
        # Determine overall quality and flags
        audit_results = self._determine_flags_and_quality(audit_results)
        
        return audit_results, total_tokens
    
    def _audit_pitch_understanding(self, pitch_data: Dict[str, str]) -> Tuple[Dict[str, Any], int]:
        """Audit pitch understanding module output."""
        content_to_audit = json.dumps(pitch_data, indent=2)
        
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",  # Using faster model for auditing
            messages=[
                {"role": "system", "content": """You are a quality auditor for AI-generated pitch analysis. 
                Score each aspect on a 1-5 scale and provide brief explanations.
                Focus on being concise and actionable."""},
                {"role": "user", "content": f"""Audit this pitch understanding analysis:

{content_to_audit}

Score each aspect (1-5):

RELEVANCE (1-5): How well does each section address its intended purpose?
BREVITY (1-5): Is the content concise without being too sparse?
COACH_TONE (1-5): Does it maintain a professional, coaching tone?

Format your response as:
RELEVANCE: [score]
Explanation: [brief explanation]

BREVITY: [score] 
Explanation: [brief explanation]

COACH_TONE: [score]
Explanation: [brief explanation]

OVERALL_ASSESSMENT: [brief summary of quality]"""}
            ],
            temperature=0.3,
            max_tokens=300
        )
        
        token_count = response.usage.total_tokens if response.usage else 0
        return self._parse_audit_response(response.choices[0].message.content), token_count
    
    def _audit_communication_scores(self, comm_data: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
        """Audit communication scoring module output."""
        content_to_audit = json.dumps(comm_data, indent=2)
        
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": """You are a quality auditor for AI-generated communication analysis.
                Evaluate if the analysis is relevant, concise, and maintains a coaching tone."""},
                {"role": "user", "content": f"""Audit this communication analysis:

{content_to_audit}

Score each aspect (1-5):

RELEVANCE (1-5): Are the scores and quotes relevant to communication quality?
BREVITY (1-5): Is the feedback concise and actionable?
COACH_TONE (1-5): Does it sound like helpful coaching rather than harsh criticism?

Format your response as:
RELEVANCE: [score]
Explanation: [brief explanation]

BREVITY: [score]
Explanation: [brief explanation] 

COACH_TONE: [score]
Explanation: [brief explanation]

OVERALL_ASSESSMENT: [brief summary]"""}
            ],
            temperature=0.3,
            max_tokens=300
        )
        
        token_count = response.usage.total_tokens if response.usage else 0
        return self._parse_audit_response(response.choices[0].message.content), token_count
    
    def _audit_fundraising_analysis(self, fundraising_data: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
        """Audit fundraising analysis module output."""
        content_to_audit = json.dumps(fundraising_data, indent=2)
        
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": """You are a quality auditor for AI-generated fundraising analysis.
                Evaluate relevance, conciseness, and coaching tone."""},
                {"role": "user", "content": f"""Audit this fundraising analysis:

{content_to_audit}

Score each aspect (1-5):

RELEVANCE (1-5): Is the analysis relevant to fundraising quality assessment?
BREVITY (1-5): Are the suggestions concise and actionable?
COACH_TONE (1-5): Does it provide constructive guidance rather than generic advice?

Format your response as:
RELEVANCE: [score]
Explanation: [brief explanation]

BREVITY: [score]
Explanation: [brief explanation]

COACH_TONE: [score]
Explanation: [brief explanation]

OVERALL_ASSESSMENT: [brief summary]"""}
            ],
            temperature=0.3,
            max_tokens=300
        )
        
        token_count = response.usage.total_tokens if response.usage else 0
        return self._parse_audit_response(response.choices[0].message.content), token_count
    
    def _audit_investor_response(self, investor_data: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
        """Audit investor response analysis module output."""
        content_to_audit = json.dumps(investor_data, indent=2)
        
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": """You are a quality auditor for AI-generated investor response analysis.
                Evaluate relevance, brevity, and coaching tone."""},
                {"role": "user", "content": f"""Audit this investor response analysis:

{content_to_audit}

Score each aspect (1-5):

RELEVANCE (1-5): Does it accurately identify objections, questions, and signals?
BREVITY (1-5): Is the analysis concise without missing key points?
COACH_TONE (1-5): Does it provide helpful interpretation rather than just listing items?

Format your response as:
RELEVANCE: [score]
Explanation: [brief explanation]

BREVITY: [score]
Explanation: [brief explanation]

COACH_TONE: [score]
Explanation: [brief explanation]

OVERALL_ASSESSMENT: [brief summary]"""}
            ],
            temperature=0.3,
            max_tokens=300
        )
        
        token_count = response.usage.total_tokens if response.usage else 0
        return self._parse_audit_response(response.choices[0].message.content), token_count
    
    def _parse_audit_response(self, content: str) -> Dict[str, Any]:
        """Parse audit response into structured data."""
        result = {
            'relevance': {'score': 3, 'explanation': 'No explanation available'},
            'brevity': {'score': 3, 'explanation': 'No explanation available'},
            'coach_tone': {'score': 3, 'explanation': 'No explanation available'},
            'overall_assessment': 'No assessment available'
        }
        
        try:
            lines = content.split('\n')
            current_section = None
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                if line.startswith('RELEVANCE:'):
                    try:
                        score = int(line.split(':')[1].strip()[0])
                        result['relevance']['score'] = score
                    except (ValueError, IndexError):
                        pass
                    current_section = 'relevance'
                elif line.startswith('BREVITY:'):
                    try:
                        score = int(line.split(':')[1].strip()[0])
                        result['brevity']['score'] = score
                    except (ValueError, IndexError):
                        pass
                    current_section = 'brevity'
                elif line.startswith('COACH_TONE:'):
                    try:
                        score = int(line.split(':')[1].strip()[0])
                        result['coach_tone']['score'] = score
                    except (ValueError, IndexError):
                        pass
                    current_section = 'coach_tone'
                elif line.startswith('OVERALL_ASSESSMENT:'):
                    result['overall_assessment'] = line.replace('OVERALL_ASSESSMENT:', '').strip()
                    current_section = 'overall'
                elif line.startswith('Explanation:') and current_section in ['relevance', 'brevity', 'coach_tone']:
                    result[current_section]['explanation'] = line.replace('Explanation:', '').strip()
        
        except Exception as e:
            print(f"Error parsing audit response: {str(e)}")
        
        return result
    
    def _determine_flags_and_quality(self, audit_results: Dict[str, Any]) -> Dict[str, Any]:
        """Determine which sections need re-generation based on scores."""
        sections_to_regenerate = []
        overall_scores = []
        
        for module_name, scores in audit_results['scores'].items():
            module_scores = []
            
            # Check each score category
            for category in ['relevance', 'brevity', 'coach_tone']:
                if category in scores:
                    score = scores[category]['score']
                    module_scores.append(score)
                    
                    # Flag for regeneration if score is below threshold
                    if score < self.score_threshold:
                        flag_reason = f"{category} score too low ({score}/{5})"
                        audit_results['flags'][f"{module_name}_{category}"] = flag_reason
                        
                        if module_name not in sections_to_regenerate:
                            sections_to_regenerate.append(module_name)
            
            if module_scores:
                overall_scores.extend(module_scores)
        
        # Determine overall quality
        if overall_scores:
            avg_score = sum(overall_scores) / len(overall_scores)
            if avg_score >= 4:
                audit_results['overall_quality'] = 'excellent'
            elif avg_score >= 3:
                audit_results['overall_quality'] = 'good'
            elif avg_score >= 2:
                audit_results['overall_quality'] = 'needs_improvement'
            else:
                audit_results['overall_quality'] = 'poor'
        
        audit_results['sections_to_regenerate'] = sections_to_regenerate
        
        # Create audit summary
        if sections_to_regenerate:
            audit_results['audit_summary'] = f"Quality issues detected in: {', '.join(sections_to_regenerate)}. Consider regenerating these sections."
        else:
            audit_results['audit_summary'] = f"All sections meet quality standards. Overall quality: {audit_results['overall_quality']}"
        
        return audit_results 