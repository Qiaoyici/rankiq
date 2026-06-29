# RankIQ

Actuarial AI candidate ranking system for the Redrob Intelligent Candidate Discovery & Ranking Challenge.

## Setup Instructions

### 1. Install Dependencies
Make sure you have Python 3.12+ installed, and install dependencies:
```bash
pip install -r requirements.txt
```

### 2. PyTorch Windows DLL Fix (if applicable)
If you encounter a `WinError 1114` DLL error loading PyTorch on Windows, run the following command to reinstall the CPU-only version:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu --force-reinstall
```

## Reproducibility Command

Run the following command to load the precomputed features, rank candidates, generate reasonings, and output the exactly 100-row `submission.csv` (takes <15 seconds on CPU):
```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

---

## Architecture Overview

RankIQ is decoupled into a high-performance two-phase pipeline:

### Phase 1: Offline Feature Extraction (`extract_features.py`)
- **Semantic JD Embedding:** Embeds structural requirements using `sentence-transformers` (`all-MiniLM-L6-v2`) on CPU, computing `CAREER_FIT_SCORE` comparing the JD to candidate career history narratives.
- **Honeypot Rules:** Validates profiles against 5 timeline and keyword stuffing constraints (impossible tenure, skills overloading, uncorroborated expert skills, experience mismatches).
- **Behavioral Signals:** Aggregates 23 platform signals prioritizing responsiveness, recency of activity, and identity validation.
- **Expected Financial Loss:** Computes the actuarial hire risk (mis-hire expected loss in INR lakhs).
- Outputs features to `features.parquet`.

### Phase 2: Dynamic Fast Ranker (`rank.py`)
- Loads features from `features.parquet`.
- Evaluates the final scoring formula, filtering out honeypots.
- Generates template-diversified 1-2 sentence reasonings referencing specific profile facts and matching the candidate's rank tier confidence.

---

## Scoring Formula & Weights

RankIQ scores and ranks candidates using a weighted actuarial formula:

$$\text{final\_score} = \left( 0.40 \times \text{career\_fit\_score} + 0.25 \times \text{failure\_mode\_score} + 0.20 \times \text{behavioral\_availability\_score} + 0.15 \times (1.0 - \text{normalized\_hire\_risk\_score}) \right) \times (1.0 - \text{honeypot\_flag})$$

Where:
- **`career_fit_score` (40%)**: Cosine similarity of career history embeddings relative to the JD requirements.
- **`failure_mode_score` (25%)**: Penalty-based score deducting points for anti-patterns (consulting only, research focus, domain mismatch).
- **`behavioral_availability_score` (20%)**: Weighted linear combination of Redrob engagement metrics.
- **`normalized_hire_risk_score` (15%)**: Normalized expected financial loss from a potential mis-hire.
- **`honeypot_flag`**: Binary multiplier (0 or 1). If any honeypot rules are triggered, the final score is set to 0.
