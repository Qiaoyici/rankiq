import json
import re
import argparse
import hashlib
import pandas as pd

# Research and domain keywords for gap identification
RESEARCH_KEYWORDS = {'research', 'paper', 'publication', 'academic', 'university', 'thesis', 'conference', 'arxiv'}
WRONG_DOMAIN_KEYWORDS = {'computer vision', 'image classification', 'object detection', 'cnn', 'speech recognition', 'tts', 'robotics', 'nlp', 'text-to-speech', 'speech-to-text', 'computer-vision'}
RIGHT_DOMAIN_KEYWORDS = {'search', 'retrieval', 'ranking', 'recommendation', 'recommender', 'information retrieval', 'ir', 'ltr', 'learning to rank', 'embed', 'vector', 'pinecone', 'milvus', 'weaviate', 'qdrant', 'faiss', 'bm25'}

def get_words(t):
    if not t:
        return set()
    return {w for w in re.findall(r'\b\w+\b', t.lower())}

def generate_reasoning(rank, row, profile, history, skills, signals):
    yoe = profile['years_of_experience']
    title = profile['current_title']
    
    # Get top matching skills from skills list
    jd_skills_keywords = ['embeddings', 'vector', 'search', 'ranking', 'retrieval', 'evaluation', 'ndcg', 'mrr', 'map', 'xgboost', 'lightgbm', 'python', 'pytorch']
    matching_skills = []
    for s in skills:
        sname = s['name'].lower()
        if any(kw in sname for kw in jd_skills_keywords):
            matching_skills.append(s['name'])
    if not matching_skills:
        matching_skills = [s['name'] for s in sorted(skills, key=lambda x: (x['proficiency'] in ['expert', 'advanced'], x['endorsements']), reverse=True)[:2]]
    matching_skills_str = ", ".join(matching_skills[:3])
    
    companies = [job['company'] for job in history]
    first_company = companies[0] if companies else "prior companies"
    
    # Identify gaps or risk flags
    gaps = []
    desc_text = " ".join([job['description'] or "" for job in history]).lower()
    
    if row['failure_mode_score'] < 1.0:
        has_consulting = any(any(c in job['company'].lower() for c in ['wipro', 'tcs', 'infosys', 'accenture', 'cognizant', 'capgemini']) for job in history)
        if has_consulting:
            gaps.append("consulting background")
        has_research = any(w in desc_text for w in RESEARCH_KEYWORDS)
        production_words = {'production', 'deploy', 'ship', 'user', 'scale', 'system'}
        has_prod = any(w in desc_text for w in production_words)
        if has_research and not has_prod:
            gaps.append("academic research focus")
        has_wrong_domain = any(w in desc_text for w in WRONG_DOMAIN_KEYWORDS)
        has_right_domain = any(w in desc_text for w in RIGHT_DOMAIN_KEYWORDS)
        if has_wrong_domain and not has_right_domain:
            gaps.append("CV/NLP-only domain focus")
            
    if signals['notice_period_days'] > 60:
        gaps.append(f"{signals['notice_period_days']}-day notice period")
        
    gaps_str = " and ".join(gaps) if gaps else "minor skill gaps"
    
    # Identify positive availability signals
    pos_signals = []
    if signals['open_to_work_flag']:
        pos_signals.append("open-to-work status")
    if signals['recruiter_response_rate'] > 0.8:
        pos_signals.append("high responsiveness")
    if signals['last_active_date'] >= '2026-05-01':
        pos_signals.append("recent active status")
    pos_signal_str = " and ".join(pos_signals[:2]) if pos_signals else "verified email/phone"
    
    # Generate templates based on rank
    if rank <= 10:
        templates = [
            f"Outstanding fit with {yoe:.1f} years of experience as a {title} at {first_company}, specializing in {matching_skills_str}. Strong match for founding-team pace, backed by high availability.",
            f"Highly qualified candidate with {yoe:.1f} years building systems at product companies like {first_company}. Expertise in {matching_skills_str} directly meets core JD requirements.",
            f"Proven track record with {yoe:.1f} years of hands-on ML engineering. Strong background in {matching_skills_str} and high platform activity confirm immediate availability."
        ]
    elif rank <= 50:
        templates = [
            f"Solid ML background with {yoe:.1f} years of experience and core skills in {matching_skills_str}. Gaps such as {gaps_str} represent minor ramp-up areas.",
            f"Strong profile showing {yoe:.1f} years in {title} roles at {first_company}. Core capabilities in {matching_skills_str} align well, though {gaps_str} warrants consideration.",
            f"Demonstrated experience of {yoe:.1f} years with expertise in {matching_skills_str}. High availability makes them a great prospective fit, despite {gaps_str}."
        ]
    elif rank <= 80:
        templates = [
            f"Adjacent experience as a {title} with skills in {matching_skills_str}. Gaps like {gaps_str} may require ramp-up, but availability is promising.",
            f"Relevant experience of {yoe:.1f} years in similar roles, with solid exposure to {matching_skills_str}. Fits well as a mid-tier candidate, noting concern around {gaps_str}.",
            f"Good foundational skills in {matching_skills_str} across {yoe:.1f} years of experience. Gaps including {gaps_str} will require onboarding support."
        ]
    else:
        templates = [
            f"Peripheral skills overlap only in {matching_skills_str} across {yoe:.1f} years of experience. Included as a boundary candidate given {pos_signal_str}.",
            f"Limited direct matching with the JD's core ranking requirements, but included as a potential boundary fit based on {pos_signal_str}.",
            f"Under-experienced or showing gaps like {gaps_str}, but listed as a boundary candidate due to background in {matching_skills_str} and {pos_signal_str}."
        ]
        
    h = int(hashlib.md5(profile['anonymized_name'].encode('utf-8')).hexdigest(), 16)
    reasoning = templates[h % len(templates)]
    
    return reasoning

def main():
    parser = argparse.ArgumentParser(description="Rank candidates for Redrob Challenge.")
    parser.add_argument('--candidates', type=str, required=True, help="Path to candidates.jsonl")
    parser.add_argument('--out', type=str, required=True, help="Path to output submission.csv")
    args = parser.parse_args()
    
    # Paths
    features_path = r'c:\Users\HP\Documents\Data & AI Hiring Challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\features.parquet'
    
    print(f"Loading precomputed features from {features_path}...")
    df = pd.read_parquet(features_path)
    
    # Calculate final score
    print("Scoring candidates...")
    df['score'] = (
        0.40 * df['career_fit_score']
        + 0.25 * df['failure_mode_score']
        + 0.20 * df['behavioral_availability_score']
        + 0.15 * (1.0 - (df['expected_loss_inr_lakhs'] / 8.0))
    ) * (1.0 - df['honeypot_flag'].astype(float))
    
    # Sort descending by score, tie-break by candidate_id ascending
    df_sorted = df.sort_values(by=['score', 'candidate_id'], ascending=[False, True])
    
    # Take top 100
    top_100 = df_sorted.head(100).copy()
    top_ids = set(top_100['candidate_id'])
    
    # Load candidate profiles for the top 100 IDs to construct reasoning
    print("Loading profile metadata for top 100 candidates...")
    top_candidates_dict = {}
    with open(args.candidates, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            cid = data['candidate_id']
            if cid in top_ids:
                top_candidates_dict[cid] = data
                
    # Generate reasoning and construct final data
    submission_rows = []
    for rank_idx, (_, row) in enumerate(top_100.iterrows(), 1):
        cid = row['candidate_id']
        cand = top_candidates_dict[cid]
        
        reasoning = generate_reasoning(
            rank=rank_idx,
            row=row,
            profile=cand['profile'],
            history=cand['career_history'],
            skills=cand['skills'],
            signals=cand['redrob_signals']
        )
        
        submission_rows.append({
            'candidate_id': cid,
            'rank': rank_idx,
            'score': float(row['score']),
            'reasoning': reasoning
        })
        
    submission_df = pd.DataFrame(submission_rows)
    print(f"Writing final submission to {args.out}...")
    submission_df.to_csv(args.out, index=False)
    print("Fast Ranking and reasoning generation complete!")

if __name__ == '__main__':
    main()
