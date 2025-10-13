# Statistical Correlation of Autism with Hyperactivity through Analyzing MRI Data

A comprehensive research project that combines multiple AI and machine learning approaches to analyze the correlation between Autism Spectrum Disorder (ASD) and hyperactivity through speech pattern analysis and gene expression studies.

## Project Overview

This project implements a multi-faceted approach to understanding the relationship between autism and hyperactivity through:

1. **Speech Pattern Analysis**: AI-powered analysis of conversational transcripts to identify hyperactivity traits
2. **Machine Learning Clustering**: Statistical analysis and clustering of speech patterns using ML techniques
3. **Gene Expression Analysis**: Bioinformatics analysis of ASD-related genes and their correlation with brain tumor markers
4. **Clinical Research Integration**: Comprehensive framework for clinical research applications

## Project Structure

```
Statistical-Correlation-of-Autism-with-Hyperactivity-through-analysing-MRI-Data/
├── AI Framework/                    # Core AI analysis framework
│   ├── LLM Agent/                  # Large Language Model-based speech analysis
│   │   ├── agent.py               # Main LLM agent for trait classification
│   │   ├── parser.py              # CHAT file parser
│   │   ├── HyperactivityKnowledge.py # Knowledge base loader
│   │   ├── HyperactivityKnowledge.yaml # Trait definitions
│   │   ├── requirements.txt       # Dependencies
│   │   ├── Nadig/                 # Sample transcript data (.cha files)
│   │   └── Outputs/               # Analysis results (CSV files)
│   └── ML+LLM/                    # Machine Learning + LLM integration
│       ├── ML+LLM.py              # Combined ML and LLM analysis
│       ├── LLM_Results_Plots.py   # Visualization tools
│       └── Nadig/                 # Transcript data for ML analysis
├── AI-Solution/                    # Integrated solution components
│   ├── agent.py                   # Alternative agent implementation
│   ├── agent1.py                  # Improved agent with validation
│   ├── ASD+BRAIN.py              # Gene expression analysis
│   ├── HyperactivityKnowledge.py  # Knowledge base
│   ├── parser.py                  # Transcript parser
│   ├── requirements.txt           # Dependencies
│   ├── README.md                  # Detailed usage documentation
│   └── Nadig/                     # Sample data
└── README.md                      # This file
```

## Key Features

### 1. Speech Pattern Analysis (LLM Agent)
- **CHAT Format Support**: Processes clinical transcript files in CHAT format (.cha)
- **Trait Classification**: Identifies 50 specific hyperactivity traits using OpenAI GPT models
- **Clinical Validation**: Built-in validation to reduce false positives
- **Batch Processing**: Efficient handling of large transcript files
- **Structured Output**: Generates detailed CSV reports with confidence scores

### 2. Machine Learning Analysis (ML+LLM)
- **Feature Extraction**: Extracts 50+ linguistic features from speech patterns
- **Clustering Analysis**: Hierarchical clustering to identify speech pattern groups
- **Statistical Correlation**: Correlation analysis between ML features and LLM classifications
- **Visualization**: Comprehensive plotting and visualization tools
- **Domain Mapping**: Maps traits to hyperactivity, impulsivity, and inattention domains

### 3. Gene Expression Analysis (ASD+BRAIN)
- **SFARI Gene Database**: Analysis of autism-related genes from SFARI database
- **Tumor Marker Correlation**: Comparison with brain tumor marker genes
- **Pathway Enrichment**: Reactome pathway analysis using hypergeometric testing
- **Machine Learning Classification**: Elastic net logistic regression for gene classification
- **Statistical Testing**: Fisher's exact test and permutation testing

## Installation

### Prerequisites
- Python 3.8 or higher
- OpenAI API key (for LLM analysis)
- Required data files (see Data Requirements section)

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Statistical-Correlation-of-Autism-with-Hyperactivity-through-analysing-MRI-Data
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   # For LLM Agent
   cd "AI Framework/LLM Agent"
   pip install -r requirements.txt
   
   # For ML+LLM analysis
   cd "../ML+LLM"
   pip install -r requirements.txt
   
   # For gene expression analysis
   cd "../../AI-Solution"
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

## Data Requirements

### For Speech Analysis
- **CHAT Format Files**: Clinical transcript files (.cha) containing conversational data
- **Sample Data**: The project includes sample data in the `Nadig/` directories

### For Gene Expression Analysis
- **SFARI Gene Database**: `SFARI-Gene_genes_07-08-2025release_08-11-2025export.csv`
- **Brain Tumor Markers**: `brain_tumor_genes.csv`
- **Reactome Pathways**: `ReactomePathways.gmt`
- **Measured Genes**: `TCGA_LGG_measured_genes.txt` (optional)

## Usage

### 1. Speech Pattern Analysis

#### Basic Analysis
```bash
cd "AI Framework/LLM Agent"
python agent.py
# Enter path to .cha file when prompted
```

#### Improved Analysis (Recommended)
```bash
cd "AI-Solution"
python agent1.py
# Enter path to .cha file when prompted
```

### 2. Machine Learning Analysis
```bash
cd "AI Framework/ML+LLM"
python ML+LLM.py --k 2 --kmin 2 --kmax 8
```

### 3. Gene Expression Analysis
```bash
cd "AI-Solution"
python ASD+BRAIN.py
```

## Output Files

### Speech Analysis Outputs
- `*_analyzed.csv`: Detailed trait classification results
- `clustering_results.csv`: ML clustering results
- `merged_with_llm.csv`: Combined ML and LLM results
- `one_child_utterance_traits.csv`: Per-utterance trait annotations

### Gene Analysis Outputs
- `ASD+TumorMarkers.csv`: Overlap between ASD and tumor marker genes
- `ASD_TumorMarkers_Reactome_enrichment.csv`: Pathway enrichment results
- `ML_TopPathwayWeights.csv`: Machine learning pathway weights
- `ML_GeneSetScores.csv`: Gene set classification scores

## Trait Categories

The system identifies 50 specific hyperactivity traits organized into three domains:

### Hyperactivity Domain
- **T01**: Talks Excessively
- **T02**: Rapid Speech
- **T03**: Interruptive Speech
- **T04**: Topic Switching
- **T07**: Filler Overuse
- **T15**: Motor Restlessness
- **T17**: Overactive Gesturing
- **T18**: Sensory Distractibility
- **T26**: Vocal Fillers Overuse
- **T27**: Rapid Laughter
- **T28**: Abrupt Volume Changes
- **T33**: Nonstop Oral Motor Noise
- **T34**: Peripheral Awareness Distraction
- **T36**: Over-enthusiastic Tone
- **T41**: Peripheral Vocalizations
- **T46**: Spatial Drift
- **T47**: Content Overload
- **T48**: Rapid Ideation

### Impulsivity Domain
- **T05**: Blurting Out Answers
- **T06**: Difficulty Waiting Turn
- **T11**: Emotional Reactivity
- **T13**: Impulsive Responses
- **T16**: Delay Aversion
- **T19**: Reward-Seeking Urgency
- **T20**: Emotional Dysregulation
- **T24**: Rapid Topic Overrun
- **T30**: Frequent Self-Correction
- **T37**: Forced Humor
- **T40**: Meta-Talk Overuse
- **T44**: Pseudo-Questions
- **T45**: Immediate Repair
- **T50**: Anticipatory Speech

### Inattention Domain
- **T09**: Echolalia
- **T10**: Stilted/Pedantic Speech
- **T12**: Inattentiveness Phrases
- **T14**: Working Memory Lapses
- **T21**: Perseverative Speech
- **T22**: Monosyllabic Responses
- **T23**: Excessive Inquiries
- **T25**: Sentence Fragments
- **T29**: Garbled Articulation
- **T31**: Auditory Overload Response
- **T32**: Sensory Hyperfocus
- **T35**: Rapid Topic Recap
- **T38**: Distracted Eye Contact
- **T39**: Repetitive Questioning
- **T42**: Segmented Speech
- **T43**: Excessive Qualifiers
- **T49**: Externalization

## Technical Details

### Dependencies
- **Core**: pandas, numpy, scikit-learn, matplotlib
- **AI/ML**: openai, scipy, seaborn
- **Data Processing**: PyYAML, python-dotenv, tqdm
- **Bioinformatics**: (for gene analysis) - standard scientific Python stack

### Model Specifications
- **LLM Model**: GPT-4o-mini (configurable)
- **Temperature**: 0.0 for consistent results
- **Batch Size**: 50 utterances per API call
- **Clustering**: Agglomerative clustering with Ward linkage
- **Classification**: Elastic net logistic regression

### Performance Metrics
- **Confidence Scoring**: Clarity, uniqueness, and quality metrics
- **Silhouette Analysis**: Cluster quality assessment
- **Correlation Analysis**: Pearson and Spearman correlations
- **Statistical Testing**: FDR correction, permutation testing

## Clinical Applications

This project is designed for:
- **Research**: Analyzing speech patterns in ADHD/ASD studies
- **Assessment**: Supporting clinical evaluations
- **Monitoring**: Tracking treatment progress
- **Documentation**: Creating detailed analysis reports

## Limitations

- Requires OpenAI API access for LLM analysis
- Analysis quality depends on LLM performance
- CHAT format specific for speech analysis
- English language transcripts only
- Requires manual review for clinical decisions
- Gene analysis requires specific data files

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

For questions or issues, please contact the development team.

---

**Note**: This tool is designed for research and clinical support purposes. Clinical decisions should always be made by qualified healthcare professionals.
