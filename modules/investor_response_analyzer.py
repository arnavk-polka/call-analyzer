from typing import Dict, List
from openai import OpenAI
import asyncio

class InvestorResponseModule:
    def __init__(self, client: OpenAI, memory_system=None):
        self.client = client
        self.memory_system = memory_system

    def analyze(self, investor_text: str) -> Dict[str, List[str]]:
        """
        Analyze investor responses using Chain of Thought prompting with historical context.
        Returns structured analysis of questions, objections, and interest signals.
        """
        
        # Get historical context from mem0
        historical_context = ""
        if self.memory_system:
            try:
                context_memories = self.memory_system.search_context(f"Investor feedback patterns: {investor_text[:200]}", limit=2)
                if context_memories:
                    historical_context = f"\n\n**Historical Context from Similar Investor Responses:**\n"
                    for i, memory in enumerate(context_memories, 1):
                        historical_context += f"{i}. {memory}\n"
                    print(f"🧠 Using context from {len(context_memories)} similar investor responses")
            except Exception as e:
                print(f"⚠️ Could not retrieve investor response context: {e}")
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": """You are an expert at analyzing investor responses and feedback with access to historical patterns.
                Use any provided historical context to better understand typical investor behavior patterns.
                Focus on identifying subtle cues, implicit concerns, and genuine interest signals.
                Structure your response clearly with specific sections."""},
                {"role": "user", "content": f"""Let's analyze these investor responses step by step:{historical_context}

1. First, let's identify all questions asked:
- Direct questions about the product/service
- Questions about the market
- Questions about financials/metrics
- Questions about the team
- Questions about future plans

2. Next, let's identify concerns or objections:
- Explicit concerns stated directly
- Implicit concerns hidden in questions
- Skepticism about assumptions
- Doubts about execution
- Market or competition concerns

3. Now, let's look for interest signals:
- Positive comments made
- Follow-up requests
- Engagement level in discussion
- Forward-looking statements
- Any hints at next steps

4. Finally, identify areas of skepticism:
- What aspects seem to make them uncomfortable?
- Where do they seem unconvinced?
- What assumptions are they challenging?

After this analysis, provide your output in the following format:

OBJECTIONS (EXPLICIT + IMPLICIT):
- [List each objection/concern]

QUESTIONS:
- [List each question identified]

INTEREST SIGNALS:
- [List each positive signal]

AREAS OF SKEPTICISM:
- [List areas where investor shows doubt]

Here's the investor response to analyze:
{investor_text}"""}
            ],
            temperature=0.7
        )

        return self._parse_response(response.choices[0].message.content)

    def _parse_response(self, content: str) -> Dict[str, List[str]]:
        """Parse the response into structured data."""
        sections = {
            'objections': [],
            'questions': [],
            'interest_signals': [],
            'areas_of_skepticism': []
        }
        
        current_section = None
        
        # Split content into lines and process
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check for section headers
            upper_line = line.upper()
            if 'OBJECTIONS' in upper_line:
                current_section = 'objections'
            elif 'QUESTIONS:' in upper_line:
                current_section = 'questions'
            elif 'INTEREST SIGNALS:' in upper_line:
                current_section = 'interest_signals'
            elif 'AREAS OF SKEPTICISM:' in upper_line:
                current_section = 'areas_of_skepticism'
            elif current_section is not None and line.startswith('-'):
                # Remove bullet point and clean the line
                item = line.lstrip('- ').strip()
                if item:  # Only add non-empty items
                    sections[current_section].append(item)
        
        # Ensure each section has at least one item
        for key in sections:
            if not sections[key]:
                sections[key].append(f"No {key.replace('_', ' ')} identified")
        
        return sections 