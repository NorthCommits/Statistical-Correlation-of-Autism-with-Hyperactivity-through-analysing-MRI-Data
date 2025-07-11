# AI Agent for Hyperactivity Trait Analysis

An intelligent system for analyzing conversational transcripts to identify hyperactivity-related speech patterns and traits in clinical settings.

## Overview

This project provides an AI-powered analysis tool that processes CHAT format transcript files (`.cha`) and uses Large Language Models (LLMs) to identify specific hyperactivity traits in conversational speech. The system is designed for clinical research and assessment of speech patterns that may indicate attention-deficit/hyperactivity disorder (ADHD) or related conditions.

## Features

- **CHAT File Parser**: Extracts speaker utterances and timestamps from CHAT format transcripts
- **Trait Classification**: Uses OpenAI GPT models to classify speech patterns into 20 specific hyperactivity traits
- **Batch Processing**: Handles large transcript files efficiently with batch processing
- **Structured Output**: Generates CSV files with detailed analysis results
- **Clinical Knowledge Base**: Comprehensive trait definitions with examples and severity weights

## Project Structure

```
AI-Agent/
├── agent.py                          # Main analysis agent
├── parser.py                         # CHAT file parser
├── HyperactivityKnowledge.py         # Knowledge base loader
├── HyperactivityKnowledge.yaml       # Trait definitions and examples
├── requirements.txt                  # Python dependencies
├── Nadig/                           # Sample transcript files
│   ├── *.cha                        # CHAT format transcripts
│   └── *_analyzed.csv               # Analysis results
└── ParserOutput/                    # Parsed transcript outputs
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd AI-Agent
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   Create a `.env` file in the project root:
   ```
   OPENAI_KEY=your_openai_api_key_here
   # OR for Azure OpenAI:
   AZURE_OPENAI_API_KEY=your_azure_openai_key_here
   OPENAI_MODEL=gpt-4o-mini  # Optional, defaults to gpt-4o-mini
   ```

## Usage

### Basic Analysis

1. **Run the main agent**:
   ```bash
   python agent.py
   ```

2. **Enter the path to your .cha file** when prompted:
   ```
   Enter the path to your .cha file: Nadig/101.cha
   ```

3. **Review the results**:
   The system will generate an `*_analyzed.csv` file with detailed analysis.

### Standalone Parser

To parse CHAT files without analysis:
```bash
python parser.py
```

## Trait Categories

The system identifies 20 specific hyperactivity traits:

### Core Speech Patterns
- **T01**: Talks Excessively
- **T02**: Rapid Speech (Pressured Speech)
- **T03**: Interruptive Speech
- **T04**: Topic Switching / Tangential Speech
- **T05**: Blurting Out Answers

### Behavioral Indicators
- **T06**: Difficulty Waiting Their Turn
- **T07**: Fidgeting & Vocal Restlessness
- **T08**: Disfluency & Stuttering-like Patterns
- **T09**: Echolalia / Repetition
- **T10**: Stilted or Pedantic Speech

### Emotional & Cognitive Features
- **T11**: Emotional Reactivity in Speech
- **T12**: Inattentiveness
- **T13**: Impulsive Responses
- **T14**: Working Memory Lapses
- **T15**: Motor Restlessness

### Additional ADHD-like Features
- **T16**: Delay Aversion
- **T17**: Overactive Gesturing
- **T18**: Sensory Distractibility
- **T19**: Reward-Seeking Urgency
- **T20**: Emotional Dysregulation

## Output Format

The analysis generates CSV files with the following columns:

| Column | Description |
|--------|-------------|
| Index | Zero-based utterance index |
| Timestamp | Time marker from transcript |
| Speaker | Speaker identifier (e.g., MOT, CHI) |
| Utterance | The actual speech content |
| Traits | Comma-separated trait IDs (e.g., "T01,T03") |
| Justification | Brief explanation for trait classification |

## Sample Output

```csv
Index,Timestamp,Speaker,Utterance,Traits,Justification
0,,MOT,what animal's on there Tracy ?,T01,"T01: Talks Excessively - The speaker is asking a question and engaging in conversation about the animal on the phone, indicating a desire to dominate the dialogue."
1,,CHI,hm: horsie ?,NONE,"NONE - no traits"
```

## Configuration

### Model Settings
- **Model**: Configurable via `OPENAI_MODEL` environment variable
- **Temperature**: Set to 0.0 for consistent results
- **Max Tokens**: 500 per response
- **Batch Size**: 50 utterances per API call

### Knowledge Base
Trait definitions are stored in `HyperactivityKnowledge.yaml` and include:
- Unique trait IDs
- Descriptive labels
- Detailed descriptions
- Behavioral cues
- Example utterances
- Severity weights (1-3 scale)

## Dependencies

- **openai**: OpenAI API client
- **pandas**: Data manipulation and CSV handling
- **PyYAML**: YAML file parsing
- **python-dotenv**: Environment variable management
- **tqdm**: Progress tracking (for large files)
- **colorama**: Terminal color output

## Clinical Applications

This tool is designed for:
- **Research**: Analyzing speech patterns in ADHD studies
- **Assessment**: Supporting clinical evaluations
- **Monitoring**: Tracking treatment progress
- **Documentation**: Creating detailed speech analysis reports

## Limitations

- Requires OpenAI API access
- Analysis quality depends on LLM performance
- CHAT format specific
- English language transcripts only
- Requires manual review for clinical decisions

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

Copyright (c) 2024 Swapnil Bhattacharya

This project is proprietary and confidential. All rights reserved.

This software and associated documentation files (the "Software") are the exclusive property of the copyright holder. The Software is provided for research and clinical support purposes only.

**Unauthorized copying, distribution, or use of this Software is strictly prohibited.**

For licensing inquiries, please contact the copyright holder.

## Support

For questions or issues, please [create an issue](link-to-issues) or contact the development team.

---

**Note**: This tool is designed for research and clinical support purposes. Clinical decisions should always be made by qualified healthcare professionals. 