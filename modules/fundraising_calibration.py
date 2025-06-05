from typing import Dict, Optional, List, Tuple
from openai import OpenAI
import re

class FundraisingCalibrationModule:
    def __init__(self, client: OpenAI):
        self.client = client

    def analyze(self, ask_text: str) -> Tuple[Optional[Dict[str, any]], int]:
        """
        Analyze fundraising ask using Chain of Thought prompting.
        Returns tuple of (analysis_results, token_count). Returns None for analysis if no funding mention.
        """
        # First check if there's any mention of funding/money
        if not self._contains_funding_mention(ask_text):
            return None, 0
            
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": """You are an expert VC analyst specializing in evaluating fundraising asks.
                Focus on clarity, realism, and whether the ask would inspire investor confidence.
                Provide specific analysis and actionable feedback."""},
                {"role": "user", "content": f"""Let's evaluate this fundraising ask step by step:

1. First, identify the funding request:
- How much money are they asking for?
- Is the amount clearly stated?
- Are there any ranges or flexibility mentioned?
- What stage of funding is this (pre-seed, seed, Series A, etc.)?

2. Analyze the use of funds:
- Is there a breakdown of how funds will be used?
- Are the categories specific and detailed?
- Do the allocations make sense for their stage?
- Are there any major gaps or oversights?

3. Evaluate realism:
- Is the amount realistic for their stage and traction?
- Does it align with typical funding rounds?
- Is the timeline for using funds reasonable?
- Are the expected outcomes achievable?

4. Assess VC confidence factors:
- Would this ask inspire confidence in investors?
- Is there evidence of careful planning?
- Are milestones and metrics mentioned?
- Does it demonstrate understanding of investor needs?

After this analysis, provide ratings and specific feedback:

FUNDING ASK CLARITY (1-5):
[Score]
Amount requested: [specific amount or "unclear"]
Supporting quote: "[exact quote about the ask]"

USE OF FUNDS ANALYSIS:
Match quality (1-5): [Score]
Breakdown provided: [Yes/No with details]
Supporting quote: "[exact quote about use of funds]"

REALISM AND CONFIDENCE (1-5):
[Score]
Stage appropriateness: [assessment]
VC confidence level: [High/Medium/Low]
Supporting quote: "[relevant quote]"

IMPROVEMENT SUGGESTIONS:
[List 2-3 specific suggestions for improving the ask]

OVERALL ASSESSMENT:
[2-3 sentence summary of the fundraising ask quality]

Here's the fundraising ask to analyze:
{ask_text}"""}
            ],
            temperature=0.7
        )

        # Track token usage
        token_count = response.usage.total_tokens if response.usage else 0

        return self._parse_response(response.choices[0].message.content), token_count

    def _contains_funding_mention(self, text: str) -> bool:
        """Check if text contains any mention of funding/money."""
        funding_keywords = [
            'funding', 'investment', 'raise', 'capital', 'money', 'dollars', 
            'million', 'thousand', 'seed', 'series', 'round', 'investor',
            'vc', 'venture', '$', 'equity', 'valuation', 'ask', 'asking'
        ]
        
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in funding_keywords)

    def _parse_response(self, content: str) -> Dict[str, any]:
        """Parse the response into structured data."""
        result = {
            'funding_ask_clarity': {
                'score': 1,
                'amount_requested': 'Not specified',
                'quote': 'No quote available'
            },
            'use_of_funds': {
                'match_score': 1,
                'breakdown_provided': False,
                'quote': 'No quote available'
            },
            'realism_confidence': {
                'score': 1,
                'stage_appropriateness': 'Unknown',
                'vc_confidence_level': 'Low',
                'quote': 'No quote available'
            },
            'improvement_suggestions': [],
            'overall_assessment': 'No assessment available'
        }
        
        try:
            current_section = None
            current_data = {}
            
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check for section headers
                upper_line = line.upper()
                if 'FUNDING ASK CLARITY' in upper_line:
                    if current_section and current_data:
                        self._update_section(result, current_section, current_data)
                    current_section = 'funding_ask_clarity'
                    current_data = {}
                elif 'USE OF FUNDS ANALYSIS' in upper_line:
                    if current_section and current_data:
                        self._update_section(result, current_section, current_data)
                    current_section = 'use_of_funds'
                    current_data = {}
                elif 'REALISM AND CONFIDENCE' in upper_line:
                    if current_section and current_data:
                        self._update_section(result, current_section, current_data)
                    current_section = 'realism_confidence'
                    current_data = {}
                elif 'IMPROVEMENT SUGGESTIONS' in upper_line:
                    if current_section and current_data:
                        self._update_section(result, current_section, current_data)
                    current_section = 'improvement_suggestions'
                elif 'OVERALL ASSESSMENT' in upper_line:
                    if current_section and current_data:
                        self._update_section(result, current_section, current_data)
                    current_section = 'overall_assessment'
                    result['overall_assessment'] = ''
                elif current_section == 'overall_assessment':
                    result['overall_assessment'] += ' ' + line
                elif current_section == 'improvement_suggestions':
                    if line.startswith('-') or line.startswith('•'):
                        suggestion = line.lstrip('-• ').strip()
                        if suggestion:
                            result['improvement_suggestions'].append(suggestion)
                elif current_section:
                    if line[0].isdigit() and current_section in ['funding_ask_clarity', 'realism_confidence']:
                        try:
                            score = int(line[0])
                            if 1 <= score <= 5:
                                current_data['score'] = score
                        except ValueError:
                            pass
                    elif line.startswith('Match quality') and current_section == 'use_of_funds':
                        try:
                            score = int(line.split(':')[1].strip()[0])
                            if 1 <= score <= 5:
                                current_data['match_score'] = score
                        except (ValueError, IndexError):
                            pass
                    elif line.startswith('Amount requested:'):
                        current_data['amount_requested'] = line.replace('Amount requested:', '').strip()
                    elif line.startswith('Breakdown provided:'):
                        breakdown = line.replace('Breakdown provided:', '').strip().lower()
                        current_data['breakdown_provided'] = breakdown.startswith('yes')
                    elif line.startswith('Stage appropriateness:'):
                        current_data['stage_appropriateness'] = line.replace('Stage appropriateness:', '').strip()
                    elif line.startswith('VC confidence level:'):
                        current_data['vc_confidence_level'] = line.replace('VC confidence level:', '').strip()
                    elif line.startswith('Supporting quote:'):
                        current_data['quote'] = line.replace('Supporting quote:', '').strip().strip('"')
            
            # Don't forget to update the last section
            if current_section and current_data and current_section not in ['improvement_suggestions', 'overall_assessment']:
                self._update_section(result, current_section, current_data)
            
            # Clean up the overall assessment
            result['overall_assessment'] = result['overall_assessment'].strip()
            if not result['overall_assessment']:
                result['overall_assessment'] = 'No assessment available'
            
        except Exception as e:
            print(f"Error parsing fundraising analysis: {str(e)}")
            import traceback
            print(traceback.format_exc())
        
        return result

    def _update_section(self, result: Dict, section: str, data: Dict):
        """Update the result dictionary with parsed data."""
        if section in result:
            for key, value in data.items():
                if key in result[section]:
                    result[section][key] = value 