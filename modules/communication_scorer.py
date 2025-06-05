from typing import Dict, List, Tuple
from openai import OpenAI

class CommunicationScorerModule:
    def __init__(self, client: OpenAI):
        self.client = client

    def analyze(self, speech_text: str) -> Tuple[Dict[str, any], int]:
        """
        Analyze founder's communication style using Chain of Thought prompting.
        Returns tuple of (analysis_results, token_count).
        """
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": """You are an expert communication coach analyzing founder pitches.
                Focus on presentation style, clarity, and potential communication issues.
                Provide specific examples and scores for each aspect."""},
                {"role": "user", "content": f"""Let's analyze this founder's communication style step by step:

1. First, analyze how they present ideas:
- Is there a logical structure to their presentation?
- Do they transition smoothly between points?
- Do they use effective examples or analogies?
- Do they emphasize key points appropriately?

2. Evaluate clarity and directness:
- Are points made clearly and directly?
- Is the language accessible to non-experts?
- Are complex concepts well explained?
- Is there unnecessary complexity or jargon?

3. Check for communication issues:
- Count filler words (um, uh, like, you know)
- List any unnecessary jargon or technical terms
- Identify any circular or rambling explanations
- Note any redundant or repetitive statements

4. Look for pacing and interaction issues:
- Are there signs of over-talking?
- Do they interrupt or talk over others?
- Is the pacing appropriate?
- Do they leave room for questions/interaction?

After this analysis, provide scores and specific examples:

IDEA PRESENTATION (1-5):
[Score]
Supporting quote: "[exact quote showing presentation style]"
Issues found: [list any issues]

CLARITY AND DIRECTNESS (1-5):
[Score]
Supporting quote: "[exact quote demonstrating clarity/lack of clarity]"
Issues found: [list any issues]

COMMUNICATION ISSUES:
Filler word count: [number]
Jargon found: [list technical terms/jargon]
Rambling instances: [quote examples]

PACING AND INTERACTION (1-5):
[Score]
Supporting quote: "[relevant quote]"
Issues found: [list any issues]

OVERALL ASSESSMENT:
[2-3 sentence summary of main communication strengths and weaknesses]

Here's the founder's speech to analyze:
{speech_text}"""}
            ],
            temperature=0.7
        )

        # Track token usage
        token_count = response.usage.total_tokens if response.usage else 0

        return self._parse_response(response.choices[0].message.content), token_count

    def _parse_response(self, content: str) -> Dict[str, any]:
        """Parse the response into structured data."""
        result = {
            'scores': {
                'idea_presentation': {'score': 1, 'quote': 'No quote available', 'issues': []},
                'clarity_directness': {'score': 1, 'quote': 'No quote available', 'issues': []},
                'pacing_interaction': {'score': 1, 'quote': 'No quote available', 'issues': []}
            },
            'communication_issues': {
                'filler_word_count': 0,
                'jargon': [],
                'rambling_instances': []
            },
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
                if 'IDEA PRESENTATION' in upper_line:
                    if current_section and current_data:
                        self._update_scores(result, current_section, current_data)
                    current_section = 'idea_presentation'
                    current_data = {}
                elif 'CLARITY AND DIRECTNESS' in upper_line:
                    if current_section and current_data:
                        self._update_scores(result, current_section, current_data)
                    current_section = 'clarity_directness'
                    current_data = {}
                elif 'COMMUNICATION ISSUES:' in upper_line:
                    if current_section and current_data:
                        self._update_scores(result, current_section, current_data)
                    current_section = 'communication_issues'
                    current_data = {}
                elif 'PACING AND INTERACTION' in upper_line:
                    if current_section and current_data:
                        self._update_scores(result, current_section, current_data)
                    current_section = 'pacing_interaction'
                    current_data = {}
                elif 'OVERALL ASSESSMENT' in upper_line:
                    if current_section and current_data:
                        self._update_scores(result, current_section, current_data)
                    current_section = 'overall'
                    result['overall_assessment'] = ''
                elif current_section == 'overall':
                    result['overall_assessment'] += ' ' + line
                elif current_section == 'communication_issues':
                    if line.startswith('Filler word count:'):
                        try:
                            count = int(''.join(filter(str.isdigit, line)))
                            result['communication_issues']['filler_word_count'] = count
                        except ValueError:
                            pass
                    elif line.startswith('Jargon found:'):
                        jargon = line.replace('Jargon found:', '').strip('[] ').split(',')
                        result['communication_issues']['jargon'] = [j.strip() for j in jargon if j.strip()]
                    elif line.startswith('Rambling instances:'):
                        rambling = line.replace('Rambling instances:', '').strip('[] ')
                        if rambling:
                            result['communication_issues']['rambling_instances'].append(rambling)
                elif current_section:
                    if line[0].isdigit():
                        try:
                            score = int(line[0])
                            if 1 <= score <= 5:
                                current_data['score'] = score
                        except ValueError:
                            pass
                    elif line.startswith('Supporting quote:'):
                        current_data['quote'] = line.replace('Supporting quote:', '').strip().strip('"')
                    elif line.startswith('Issues found:'):
                        issues = line.replace('Issues found:', '').strip('[] ').split(',')
                        current_data['issues'] = [i.strip() for i in issues if i.strip()]
            
            # Don't forget to update the last section
            if current_section and current_data and current_section != 'overall':
                self._update_scores(result, current_section, current_data)
            
            # Clean up the overall assessment
            result['overall_assessment'] = result['overall_assessment'].strip()
            if not result['overall_assessment']:
                result['overall_assessment'] = 'No assessment available'
            
        except Exception as e:
            print(f"Error parsing communication analysis: {str(e)}")
            import traceback
            print(traceback.format_exc())
        
        return result

    def _update_scores(self, result: Dict, section: str, data: Dict):
        """Update the scores dictionary with parsed data."""
        if section in result['scores']:
            result['scores'][section].update({
                'score': data.get('score', 1),
                'quote': data.get('quote', 'No quote available'),
                'issues': data.get('issues', [])
            }) 