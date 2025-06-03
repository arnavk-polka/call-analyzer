# Call Analyst

A simple tool to analyze investor call transcripts using AI. The tool provides structured insights including strengths, weaknesses, investor impression, and missed opportunities.

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Set your OpenAI API key as an environment variable:
```bash
# Windows PowerShell
$env:OPENAI_API_KEY="your-api-key-here"

# Linux/Mac
export OPENAI_API_KEY="your-api-key-here"
```

3. Start the backend server:
```bash
python main.py
```

4. Open `index.html` in your web browser.

## Usage

1. Either paste your transcript directly into the text area or upload a text file.
2. Click "Analyze" to process the transcript.
3. View the results in the structured sections below:
   - Strengths
   - Weaknesses
   - Investor Impression
   - Missed Opportunities
   - Confidence Rating
   - Final Recommendation

## Features

- Modern, responsive UI
- File upload support
- Real-time analysis using GPT-3.5
- Structured insights
- Confidence rating system
- Loading indicator for better UX

## Requirements

- Python 3.7+
- OpenAI API key
- Modern web browser 