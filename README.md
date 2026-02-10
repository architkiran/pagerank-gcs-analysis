# PageRank Analysis on Google Cloud Storage

## Project Information
- **Course**: CS528 Cloud Computing
- **GCP Project ID**: utopian-planet-485618-b3
- **Bucket Name**: bu-cs528-architkk
- **Bucket Region**: us-central1

## Files
- `generate_files.py` - Generates 20K HTML files with random links
- `pagerank_analysis.py` - Analyzes PageRank from GCS bucket
- `test_pagerank.py` - Tests PageRank algorithm correctness

## Prerequisites
```bash
pip install google-cloud-storage
```

## Authentication
```bash
gcloud auth application-default login
gcloud config set project utopian-planet-485618-b3
```

## Usage

### Run Tests
```bash
python3 test_pagerank.py
```

### Run PageRank Analysis
```bash
python3 pagerank_analysis.py bu-cs528-architkk --workers 20
```

## Parameters
- `bucket_name` (required): Name of your GCS bucket
- `--workers` (optional): Number of parallel download workers (default: 20)

## Algorithm
Implements the iterative PageRank algorithm:
```
PR(A) = (1-d)/n + d * (dangling_sum/n + sum(PR(Ti)/C(Ti)))
```
Where:
- d = 0.85 (damping factor)
- n = total number of pages
- Ti = pages pointing to page A
- C(Ti) = outgoing link count from page Ti

Converges when sum of PageRanks changes by < 0.5%

## Results
See PDF report for detailed analysis and timing comparisons.
