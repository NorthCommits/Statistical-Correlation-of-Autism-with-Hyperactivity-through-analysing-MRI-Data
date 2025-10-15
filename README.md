# Statistical Correlation of Autism with Hyperactivity through Analyzing MRI Data

A comprehensive research project combining neuroimaging (MRI: structural, DTI, and resting-state fMRI) with AI/ML-based speech analysis to study relationships between Autism Spectrum Disorder (ASD) and hyperactivity.

## Project Overview

This project implements a multi-faceted approach to understanding the relationship between autism and hyperactivity through:

1. **Neuroimaging Analysis**: Structural MRI, DTI-derived metrics (e.g., FA, MD), and rs-fMRI features (e.g., ReHo, ALFF), including ROI-based analyses and group comparisons
2. **Speech Pattern Analysis**: LLM-powered analysis of conversational transcripts to identify hyperactivity traits
3. **ML Integration**: Clustering and correlation analyses linking speech-derived traits with quantitative features
4. **Clinical Research Integration**: Reproducible workflows for research and exploratory clinical support

## Project Structure

```
Statistical-Correlation-of-Autism-with-Hyperactivity-through-analysing-MRI-Data/
├── AI Framework/
│   ├── ASD-ADHD-Research/                 # Neuroimaging workflows and data
│   │   ├── Dataset/                       # Source MRI data (e.g., ABIDEII-STANFORD)
│   │   ├── OutputFiles/                   # Derived NIfTI/NumPy/CSV outputs and masks
│   │   └── codes/                         # Jupyter notebooks and helper CSVs
│   ├── LLM Agent/                         # LLM-based speech analysis
│   │   ├── agent.py                       # Main agent for trait classification
│   │   ├── parser.py                      # CHAT file parser
│   │   ├── HyperactivityKnowledge.py      # Knowledge base loader
│   │   ├── HyperactivityKnowledge.yaml    # Trait definitions
│   │   ├── Nadig/                         # Sample transcript data (.cha files)
│   │   ├── Outputs/                       # Analysis results (CSV files)
│   │   └── requirements.txt               # Dependencies (LLM agent)
│   └── ML+LLM/                            # Machine Learning + LLM integration
│       ├── ML+LLM.py                      # Combined ML and LLM analysis
│       ├── LLM_Results_Plots.py           # Visualization utilities
│       └── Nadig/                         # Transcript data for ML analysis
└── README.md
```

## Key Features

### 1. Neuroimaging (MRI, DTI, rs-fMRI)
- **Structural preprocessing**: Brain extraction, tissue segmentation, masks
- **DTI processing**: FA/MD/L1-L3/MO maps, tract-focused analyses (e.g., corpus callosum, SLF)
- **rs-fMRI features**: ReHo, mALFF, nuisance regression, motion correction
- **ROI analyses**: Prefrontal, cingulate, temporal lobe, corpus callosum masks
- **Group comparisons**: t-statistics, p-value maps, FDR/BH correction

### 2. Speech Pattern Analysis (LLM Agent)
- **CHAT format support**: Processes clinical transcript files (.cha)
- **Trait classification**: Identifies 50 traits related to hyperactivity/impulsivity/inattention
- **Batch processing**: Efficient handling of large transcripts with structured CSV outputs

### 3. Machine Learning Analysis (ML+LLM)
- **Feature extraction**: Linguistic features and LLM-derived traits
- **Clustering/correlation**: Hierarchical clustering and correlation studies
- **Visualization**: Result plots and exploratory analytics

## Installation

### Prerequisites
- Python 3.9 or higher
- OpenAI API key (for LLM analysis)
- MRI datasets (see Data Requirements)

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
   # LLM Agent
   cd "AI Framework/LLM Agent"
   pip install -r requirements.txt

   # ML+LLM utilities (uses core scientific Python stack)
   cd "../ML+LLM"
   # If a requirements.txt is present, install it; otherwise ensure numpy, pandas, scikit-learn, matplotlib, seaborn are available
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

### For Neuroimaging
- **ABIDE II STANFORD** (example dataset in `AI Framework/ASD-ADHD-Research/Dataset/ABIDEII-STANFORD/`)
- Optional modality files: structural T1 (`anat.nii.gz`), DTI (`dti.nii.gz` with `.bval/.bvec`), resting-state fMRI (`rest.nii.gz`)
- Provided ROI masks in `AI Framework/ASD-ADHD-Research/OutputFiles/Masks/`

### For Speech Analysis
- **CHAT format files** (.cha) in the `Nadig/` directories

## Usage

### 1. Neuroimaging Workflows (Notebooks)
- Launch Jupyter and open notebooks under `AI Framework/ASD-ADHD-Research/codes/` (e.g., `main.ipynb`, `code.ipynb`, `wmh.ipynb`). These notebooks walk through preprocessing, feature extraction (ReHo, mALFF, DTI metrics), and group comparisons. Many intermediate and final outputs are written to `AI Framework/ASD-ADHD-Research/OutputFiles/`.

### 2. Speech Pattern Analysis (LLM Agent)
```bash
cd "AI Framework/LLM Agent"
python agent.py
# Enter path to .cha file when prompted
```

### 3. Machine Learning Analysis (ML+LLM)
```bash
cd "AI Framework/ML+LLM"
python ML+LLM.py --k 2 --kmin 2 --kmax 8
```

## Output Files

### Neuroimaging Outputs (examples)
- NIfTI volumes: `anat_brain.nii.gz`, `dti_FA.nii.gz`, `reho_asd.nii.gz`, `alff_adhd.nii.gz`
- ROI derivatives: `fa_corpus_callosum.nii.gz`, `fa_prefrontal_cortex.nii.gz`, etc.
- Group stats: `t_stat_group_comparison.nii.gz`, `p_value_group_comparison.nii.gz`, corrected maps (`p_map_corrected.nii.gz`)
- Intermediate artifacts: motion parameters (`*.par`), masks (`*_mask.nii.gz`), cleaned series (`rest_cleaned_*.nii.gz`)

### Speech/ML Outputs
- `*_analyzed.csv`: Per-utterance trait classification
- `clustering_results.csv`: ML clustering results
- `merged_with_llm.csv`: Combined ML and LLM results
- Feature arrays: `features_asd.npy`, `features_adhd.npy`, regional metrics (`regional_malff_*.npy`)

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
- **Core**: numpy, pandas, scipy, matplotlib, seaborn, tqdm
- **LLM Agent**: openai, PyYAML, python-dotenv
- **Imaging**: nibabel, nilearn (and external neuroimaging tools if used in notebooks)

### Model/Analysis Specifications
- **LLM model**: GPT-4o-mini (configurable)
- **Temperature**: 0.0 for consistency
- **Batch size**: 50 utterances per API call
- **Clustering**: Agglomerative (Ward linkage)

### Performance Metrics
- **Confidence scoring**: Clarity, uniqueness, quality metrics
- **Silhouette analysis**: Cluster quality assessment
- **Correlation analysis**: Pearson and Spearman correlations
- **Statistical testing**: Multiple-comparison correction where applicable

## Clinical Applications

This project is designed for:
- **Research**: Analyzing neuroimaging and speech patterns in ASD/ADHD
- **Assessment**: Supporting clinical evaluations
- **Monitoring**: Tracking treatment progress
- **Documentation**: Creating detailed analysis reports

## Limitations

- Requires OpenAI API access for LLM analysis
- Analysis quality depends on LLM performance
- CHAT format specific for speech analysis
- English language transcripts only
- Requires manual review for clinical decisions

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

Copyright (c) 2025 Dr Qamar Natsheh, Daniela Cifuentes Barrios and Swapnil Bhattacharya

This project is proprietary and confidential. All rights reserved.

This software and associated documentation files (the "Software") are the exclusive property of the copyright holders. The Software is provided for research and clinical support purposes only.

**Unauthorized copying, distribution, or use of this Software is strictly prohibited.**

For licensing inquiries, please contact the copyright holders.

## Support

For questions or issues, please contact the development team.

---

**Note**: This tool is designed for research and clinical support purposes. Clinical decisions should always be made by qualified healthcare professionals.
