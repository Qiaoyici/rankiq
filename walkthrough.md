# RankIQ Implementation Walkthrough

This document outlines the final implementation and validation of the candidate ranking system **RankIQ**.

## Summary of Accomplishments

1. **Phase 1: Feature Extraction (`extract_features.py`)**  
   - Processes the full 100K candidate pool offline in 31 minutes.
   - Uses `sentence-transformers` (`all-MiniLM-L6-v2`) on CPU to embed career history descriptions.
   - Evaluates behavioral availability, failure mode anti-patterns, and actuarial mis-hire financial risk.
   - Saves precomputed scores to `features.parquet`.

2. **Phase 2: Fast Ranker (`rank.py`)**  
   - Loads the parquet file and parses `candidates.jsonl` dynamically.
   - Ranks candidates using the weighted formula and filters out honeypots.
   - Generates customized 1-2 sentence reasonings matching the tone of each rank.
   - Execution time is **under 15 seconds**, easily satisfying the <5 minutes CPU budget.

3. **Format & Constraint Validation**  
   - Validated the output file using `validate_submission.py`. The submission CSV is **100% valid** and compliant.

---

## Phase 1 Feature Extraction Statistics

The offline feature extraction on `candidates.jsonl` produced the following metrics:

| Metric | Value |
| :--- | :--- |
| **Total Candidates Processed** | 100,000 |
| **Total Honeypots Flagged** | 15,512 |
| &nbsp;&nbsp;&nbsp;&nbsp;*- Impossible Tenure (Krutrim/Sarvam AI < 2023)* | 73 |
| &nbsp;&nbsp;&nbsp;&nbsp;*- Expert Skill Zero Endorsements/Duration* | 15 |
| &nbsp;&nbsp;&nbsp;&nbsp;*- Years of Experience Mismatch* | 49 |
| &nbsp;&nbsp;&nbsp;&nbsp;*- Skills Overload (Uncorroborated > 8, Low Endorsement/Duration)* | 97 |
| &nbsp;&nbsp;&nbsp;&nbsp;*- Unrelated Titles* | 0 |
| &nbsp;&nbsp;&nbsp;&nbsp;*- Impossible Skill Duration (Skill Duration > YoE * 1.1)* | 15,311 |
| **Average Career Fit Score** | 0.4564 |
| **Average Behavioral Availability Score** | 0.5084 |
| **Candidates Open to Work** | 35,339 |
| **Candidates Located in India** | 75,113 |

---

## Validation Results

Running the challenge's official validator:
```bash
python validate_submission.py submission.csv
```
**Output:**
```text
Submission is valid.
```

The output file `submission.csv` contains exactly 100 data rows, ranks 1-100, and non-increasing scores, matching all non-negotiable hackathon constraints.
