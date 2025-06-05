from typing import Dict, Tuple
from openai import OpenAI
import os

class PitchUnderstandingModule:
    def __init__(self, client: OpenAI):
        self.client = client

    def analyze(self, pitch_text: str) -> Tuple[Dict[str, str], int]:
        """
        Analyze a pitch using Chain of Thought prompting.
        Returns tuple of (analysis_results, token_count).
        """
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": """You are an expert pitch analyzer. Break down startup pitches systematically and thoroughly.
                Always structure your response with clear section headers and detailed analysis under each section."""},
                {"role": "user", "content": f"""Let's analyze this pitch step by step:

1. First, let's identify what the startup does:
- What is their core product/service?
- What is their unique value proposition?

2. Next, let's understand the problem:
- What specific problem are they solving?
- Who experiences this problem?
- Why is this problem significant?

3. Now, examine their solution:
- How does their product/service work?
- What makes it unique?
- What are the key features/benefits?

4. Let's look at the team and traction:
- Who are the key team members?
- What relevant experience do they have?
- Any notable achievements or milestones?
- Market size or opportunity mentioned?

5. Finally, what are they asking for:
- How much funding are they seeking?
- What will the funds be used for?
- Any other specific asks?

After this analysis, provide your output in the following format:

VALUE PROPOSITION:
[Clear statement of what they do and their unique value]

PROBLEM:
[Detailed description of the problem they're solving]

SOLUTION:
[Comprehensive explanation of their solution]

TEAM/TRACTION:
[Team details, experience, and any traction metrics]

ASK:
[Funding request and use of funds]

FINAL SUMMARY:
[2-3 sentence synthesis of the key points]

Here's the pitch to analyze:
{pitch_text}"""}
            ],
            temperature=0.7
        )

        # Track token usage
        token_count = response.usage.total_tokens if response.usage else 0
        
        return self._parse_response(response.choices[0].message.content), token_count

    def _parse_response(self, content: str) -> Dict[str, str]:
        """Parse the response into structured data."""
        sections = {
            'value_proposition': '',
            'problem': '',
            'solution': '',
            'team_traction': '',
            'ask': '',
            'final_summary': ''
        }
        
        current_section = None
        current_content = []
        
        # Split content into lines and process
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check for section headers
            upper_line = line.upper()
            if 'VALUE PROPOSITION:' in upper_line:
                if current_section and current_content:
                    sections[current_section] = ' '.join(current_content)
                current_section = 'value_proposition'
                current_content = []
            elif 'PROBLEM:' in upper_line:
                if current_section and current_content:
                    sections[current_section] = ' '.join(current_content)
                current_section = 'problem'
                current_content = []
            elif 'SOLUTION:' in upper_line:
                if current_section and current_content:
                    sections[current_section] = ' '.join(current_content)
                current_section = 'solution'
                current_content = []
            elif 'TEAM/TRACTION:' in upper_line or 'TEAM AND TRACTION:' in upper_line:
                if current_section and current_content:
                    sections[current_section] = ' '.join(current_content)
                current_section = 'team_traction'
                current_content = []
            elif 'ASK:' in upper_line or 'FUNDING ASK:' in upper_line:
                if current_section and current_content:
                    sections[current_section] = ' '.join(current_content)
                current_section = 'ask'
                current_content = []
            elif 'FINAL SUMMARY:' in upper_line or 'SUMMARY:' in upper_line:
                if current_section and current_content:
                    sections[current_section] = ' '.join(current_content)
                current_section = 'final_summary'
                current_content = []
            elif current_section is not None:
                # Remove any bullet points or dashes at the start of lines
                line = line.lstrip('•-* ')
                current_content.append(line)
        
        # Don't forget to add the last section
        if current_section and current_content:
            sections[current_section] = ' '.join(current_content)
        
        # Clean up the sections
        for key in sections:
            sections[key] = sections[key].strip()
            if not sections[key]:
                sections[key] = f"No {key.replace('_', ' ')} provided"
        
        return sections 