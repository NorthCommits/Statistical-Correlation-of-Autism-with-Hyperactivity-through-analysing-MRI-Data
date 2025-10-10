# Pragmatic Feature Extraction and Clustering from the Nadig Corpus

**Author:** Daniela Cifuentes Barrios  
**Copyright © 2025 Daniela Cifuentes Barrios**

## Overview

This project analyzes child language transcripts from the ASDBank English Nadig Corpus to extract pragmatic and linguistic features, and applies unsupervised clustering to identify ADHD-like language patterns. The workflow includes feature extraction, normalization, clustering, and visualization.

## Dataset

- **Corpus:** ASDBank English Nadig Corpus
- **Creators:** Aparna Nadig, Janet Bang
- **Language:** English
- **Description:** Transcripts of children with ASD and typically developing controls
- **Country:** Canada
- **Date:** 2012-07-01
- **DOI:** [10.21415/T54P4Q](https://doi.org/10.21415/T54P4Q)
- **Types:** cross, toyplay, ASD (see `Nadig/0types.txt`)

The `.cha` files in the `Nadig/` directory are CHAT-format transcripts. Metadata is in `Nadig/0metadata.cdc`.

## Features Extracted
- Fillers (e.g., "uh", "um")
- Repairs (e.g., "I mean", "sorry")
- Repetitions
- Topic shifts
- Interruptions
- Short utterances (<3 words)
- Utterance and word counts
- Mean/median tokens per utterance
- Feature densities (per 100 words/utterances)

## Clustering & Analysis
- Features are standardized using z-scores
- KMeans clustering (k=2) is applied
- An ADHD-like index is computed
- Cluster assignments and silhouette scores are reported
- Elbow and silhouette plots help determine optimal cluster count

## Setup

1. **Install dependencies:**
   - Python 3.8+
   - Required packages:
     - pandas
     - numpy
     - scikit-learn
     - matplotlib
     - IPython (for display)

   Install with:
   ```bash
   pip install pandas numpy scikit-learn matplotlib ipython
   ```

2. **Dataset:**
   - Place the Nadig corpus `.cha` files in the `Nadig/` directory (already included).

## Usage

Run the main script:
```bash
python code.py
```

- The script will extract features, perform clustering, and output CSV files with results.
- Visualizations will be displayed for cluster analysis.

## Outputs
- `features_all.csv`: Extracted features for all files (path hardcoded in script)
- `clustered.csv`: Clustering results (path hardcoded in script)
- Elbow and silhouette analysis plot (displayed)

## Notes
- File paths in `code.py` are currently hardcoded for Windows. Adjust as needed for your environment.
- The script expects the Nadig corpus in the `Nadig/` directory.

## License

This project is licensed under the MIT License:

```
MIT License

Copyright (c) 2025 Daniela Cifuentes Barrios

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
``` 
