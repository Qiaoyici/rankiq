from sentence_transformers import SentenceTransformer
import json
import re
import os
from datetime import datetime
import pandas as pd
from tqdm import tqdm
import numpy as np

# Reference Date for availability recency
REF_DATE = datetime(2026, 5, 28)

# Newly founded AI companies (founded in 2023)
NEW_AI_COMPANIES = {'Krutrim', 'Sarvam AI'}

# Consulting/Services companies
CONSULTING_COMPANIES = {'tcs', 'wipro', 'infosys', 'accenture', 'cognizant', 'capgemini', 'mindtree'}

# Stopwords for skill/title corroboration
STOPWORDS = {'and', 'or', 'of', 'in', 'at', 'the', 'for', 'a', 'an', 'with', 'to', 'on', 'by', 'using', 'from', 'systems', 'platform', 'framework', '&', '-', '/'}

# Research keywords vs Production keywords
RESEARCH_KEYWORDS = {'research', 'paper', 'publication', 'academic', 'university', 'thesis', 'conference', 'arxiv'}
PRODUCTION_KEYWORDS = {'production', 'deploy', 'ship', 'user', 'scale', 'system', 'infrastructure', 'real-time', 'latency', 'throughput', 'kubernetes', 'docker', 'aws', 'gcp', 'fastapi', 'flask', 'serve', 'pipeline', 'client', 'customer'}

# Wrong domain (CV/NLP/robotics) keywords vs Right domain (IR/Search/Ranking) keywords
WRONG_DOMAIN_KEYWORDS = {'computer vision', 'image classification', 'object detection', 'cnn', 'speech recognition', 'tts', 'robotics', 'nlp', 'text-to-speech', 'speech-to-text', 'computer-vision'}
RIGHT_DOMAIN_KEYWORDS = {'search', 'retrieval', 'ranking', 'recommendation', 'recommender', 'information retrieval', 'ir', 'ltr', 'learning to rank', 'embed', 'vector', 'pinecone', 'milvus', 'weaviate', 'qdrant', 'faiss', 'bm25'}

def get_words(t):
    if not t:
        return set()
    return {w for w in re.findall(r'\b\w+\b', t.lower()) if w not in STOPWORDS and len(w) >= 3}

def is_mentioned(skill_name, descriptions_text):
    skill_name = skill_name.lower().strip()
    descriptions_text = descriptions_text.lower()
    
    # 1. Direct substring check
    if skill_name in descriptions_text:
        return True
        
    # 2. Singular/plural or minor variations
    if skill_name.endswith('s') and skill_name[:-1] in descriptions_text:
        return True
    if skill_name + 's' in descriptions_text:
        return True
        
    # 3. Mappings for common skills
    mappings = {
        'fine-tuning llms': ['fine-tuning', 'fine tuning', 'llm', 'llms'],
        'large language models': ['llm', 'llms', 'large language model'],
        'recommendation systems': ['recommendation', 'recommender'],
        'natural language processing': ['nlp'],
        'computer vision': ['cv', 'vision'],
        'neural networks': ['neural network', 'deep learning'],
        'generative adversarial networks': ['gan', 'gans'],
        'rest apis': ['rest api', 'restful', 'api', 'apis'],
        'weights & biases': ['wandb', 'weights and biases'],
        'scikit-learn': ['sklearn'],
        'tensorflow': ['tf'],
        'hugging face transformers': ['huggingface', 'transformers', 'transformer'],
        'sentence transformers': ['sentence-transformers', 'transformers', 'transformer'],
        'google cloud platform': ['gcp', 'google cloud'],
        'amazon web services': ['aws', 'amazon'],
        'powerpoint': ['ppt'],
        'excel': ['spreadsheet'],
        'six sigma': ['6 sigma']
    }
    if skill_name in mappings:
        for alt in mappings[skill_name]:
            if alt in descriptions_text:
                return True
                
    # 4. Check if any major keyword (length >= 3 and not stopword) is present
    words = re.findall(r'\b\w+\b', skill_name)
    keywords = [w for w in words if w not in STOPWORDS and len(w) >= 3]
    if keywords:
        if all(kw in descriptions_text for kw in keywords):
            return True
            
    return False

def get_seniority_level(title):
    title = title.lower()
    if any(w in title for w in ['junior', 'associate', 'intern', 'graduate', 'analyst', 'trainee']):
        return 1
    elif any(w in title for w in ['senior', 'sr.']):
        return 3
    elif any(w in title for w in ['staff', 'lead', 'principal', 'director', 'vp', 'head', 'chief']):
        return 4
    else:
        return 2

def normalize(val, max_val, invert=False):
    val = max(0.0, min(float(val), float(max_val)))
    if invert:
        return (max_val - val) / max_val
    else:
        return val / max_val

def main():
    print("Initializing sentence-transformers (all-MiniLM-L6-v2) on CPU...")
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    
    # JD Implied Requirements text
    jd_text = (
        "Production ML systems at product companies. Experience with embeddings, vector search, "
        "ranking, and retrieval systems. Strong Python programming. Designing and deploying "
        "evaluation frameworks (NDCG, MRR, MAP) for search or recommendation systems."
    )
    print("Encoding JD requirements...")
    jd_embedding = model.encode(jd_text, convert_to_numpy=True)
    jd_embedding = jd_embedding / np.linalg.norm(jd_embedding)
    
    # Path to candidates.jsonl
    candidates_path = r'c:\Users\HP\Documents\Data & AI Hiring Challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\candidates.jsonl'
    features_output_path = r'c:\Users\HP\Documents\Data & AI Hiring Challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\features.parquet'
    
    print(f"Reading candidates from {candidates_path}...")
    candidates = []
    with open(candidates_path, 'r', encoding='utf-8') as f:
        for line in f:
            candidates.append(json.loads(line))
            
    print(f"Loaded {len(candidates)} candidates.")
    
    # Batch compute embeddings for career narratives
    print("Building career narratives...")
    narratives = []
    for cand in candidates:
        profile = cand['profile']
        history = cand['career_history']
        narrative = f"Current title: {profile['current_title']}. Headline: {profile['headline']}. Summary: {profile['summary']}. "
        narrative += " ".join([f"Job title: {job['title']} at {job['company']}. Job description: {job['description']}" for job in history])
        narratives.append(narrative)
        
    print("Encoding career narratives in batches...")
    embeddings = model.encode(narratives, batch_size=256, show_progress_bar=True, convert_to_numpy=True)
    
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    embeddings = embeddings / norms
    
    # Compute similarity
    similarities = np.dot(embeddings, jd_embedding)
    
    # Feature extraction
    records = []
    
    # Stats counters
    impossible_tenure_count = 0
    expert_skill_zero_count = 0
    yoe_mismatch_count = 0
    skills_overload_count = 0
    title_unrelated_count = 0
    open_to_work_count = 0
    in_india_count = 0
    
    print("Extracting features for each candidate...")
    for idx, cand in enumerate(tqdm(candidates)):
        cid = cand['candidate_id']
        profile = cand['profile']
        history = cand['career_history']
        skills = cand['skills']
        signals = cand['redrob_signals']
        
        # 1. CAREER_FIT_SCORE
        sim = float(similarities[idx])
        # Product company boost
        companies = {job['company'].lower() for job in history}
        is_product_only = not (companies & CONSULTING_COMPANIES)
        boost = 0.10 if is_product_only else 0.0
        
        # Pure research penalty
        desc_text = " ".join([job['description'] or "" for job in history]).lower()
        has_research = any(w in desc_text for w in RESEARCH_KEYWORDS)
        has_production = any(w in desc_text for w in PRODUCTION_KEYWORDS)
        penalty = 0.15 if (has_research and not has_production) else 0.0
        
        career_fit_score = max(0.0, min(1.0, sim + boost - penalty))
        
        # 2. HONEYPOT_FLAG
        impossible_tenure = False
        for job in history:
            comp = job['company']
            if comp in NEW_AI_COMPANIES:
                start_year = int(job['start_date'].split('-')[0])
                if start_year < 2023:
                    impossible_tenure = True
                    break
                    
        expert_skill_zero = False
        desc_words = get_words(desc_text + " " + (profile['summary'] or "") + " " + (profile['headline'] or ""))
        for s in skills:
            if s['proficiency'] in ['expert', 'advanced'] and s['endorsements'] == 0 and s['duration_months'] == 0:
                if not is_mentioned(s['name'], desc_text + " " + (profile['summary'] or "") + " " + (profile['headline'] or "")):
                    expert_skill_zero = True
                    break
                    
        total_months_sum = sum(job['duration_months'] for job in history)
        claimed_months = profile['years_of_experience'] * 12
        yoe_mismatch = False
        if claimed_months == 0:
            if total_months_sum > 0:
                yoe_mismatch = True
        else:
            diff_ratio = abs(claimed_months - total_months_sum) / max(claimed_months, 1.0)
            if diff_ratio > 0.20:
                yoe_mismatch = True
                
        skills_overload = False
        uncorroborated_count = 0
        for s in skills:
            if s['proficiency'] in ['expert', 'advanced']:
                if not is_mentioned(s['name'], desc_text + " " + (profile['summary'] or "") + " " + (profile['headline'] or "")):
                    uncorroborated_count += 1
        if uncorroborated_count > 8:
            skills_overload = True
            
        title_unrelated = False
        
        if impossible_tenure:
            impossible_tenure_count += 1
        if expert_skill_zero:
            expert_skill_zero_count += 1
        if yoe_mismatch:
            yoe_mismatch_count += 1
        if skills_overload:
            skills_overload_count += 1
        if title_unrelated:
            title_unrelated_count += 1
            
        honeypot_flag = impossible_tenure or expert_skill_zero or yoe_mismatch or skills_overload or title_unrelated
        
        # 3. BEHAVIORAL_AVAILABILITY_SCORE
        active_date = datetime.strptime(signals['last_active_date'], "%Y-%m-%d")
        days_since_last_active = (REF_DATE - active_date).days
        
        availability = 0.0
        availability += 0.20 * normalize(days_since_last_active, max_val=365, invert=True)
        availability += 0.15 * (1.0 if signals['open_to_work_flag'] else 0.0)
        availability += 0.15 * max(0.0, min(1.0, float(signals['recruiter_response_rate'])))
        
        np_days = signals['notice_period_days']
        np_val = 1.0 if np_days <= 30 else 0.7 if np_days <= 60 else 0.4 if np_days <= 90 else 0.1
        availability += 0.10 * np_val
        
        availability += 0.10 * max(0.0, min(1.0, float(signals['interview_completion_rate'])))
        availability += 0.10 * normalize(signals['profile_completeness_score'], max_val=100.0)
        
        oar = float(signals['offer_acceptance_rate'])
        availability += 0.05 * (oar if oar != -1.0 else 0.5)
        availability += 0.05 * (1.0 if (signals['verified_email'] and signals['verified_phone']) else 0.0)
        
        gh = float(signals['github_activity_score'])
        gh_val = 0.3 if gh == -1.0 else normalize(gh, max_val=100.0)
        availability += 0.05 * gh_val
        
        loc = profile['location'] or ""
        target_cities = ["pune", "noida", "hyderabad", "mumbai", "delhi", "bangalore"]
        loc_val = 1.0 if (signals['willing_to_relocate'] or any(city in loc.lower() for city in target_cities)) else 0.0
        availability += 0.05 * loc_val
        
        # General stats tracking
        if signals['open_to_work_flag']:
            open_to_work_count += 1
        if profile['country'] and profile['country'].strip().lower() == 'india':
            in_india_count += 1
            
        # 4. FAILURE_MODE_SCORE
        failure_penalties = 0.0
        
        recent_jobs = []
        for job in history:
            start_year = int(job['start_date'].split('-')[0])
            if start_year >= 2022:
                recent_jobs.append(job)
        if len({job['company'] for job in recent_jobs}) >= 3:
            sorted_jobs = sorted(recent_jobs, key=lambda x: x['start_date'])
            levels = [get_seniority_level(job['title']) for job in sorted_jobs]
            is_increasing = False
            is_strictly_non_decreasing = True
            for i in range(len(levels) - 1):
                if levels[i+1] > levels[i]:
                    is_increasing = True
                if levels[i+1] < levels[i]:
                    is_strictly_non_decreasing = False
            if is_increasing and is_strictly_non_decreasing:
                failure_penalties += 0.3
                
        has_framework = any(f_name in [s['name'].lower() for s in skills] for f_name in ['langchain', 'llamaindex', 'autogpt', 'babyagi'])
        has_pre_llm = any(kw in desc_text for kw in RIGHT_DOMAIN_KEYWORDS)
        if has_framework and not has_pre_llm:
            failure_penalties += 0.4
            
        is_pure_services = all(any(c in job['company'].lower() for c in CONSULTING_COMPANIES) for job in history)
        if is_pure_services:
            failure_penalties += 0.5
            
        if has_research and not has_production:
            failure_penalties += 0.4
            
        has_wrong_domain = any(kw in desc_text for kw in WRONG_DOMAIN_KEYWORDS)
        has_right_domain = any(kw in desc_text for kw in RIGHT_DOMAIN_KEYWORDS)
        if has_wrong_domain and not has_right_domain:
            failure_penalties += 0.3
            
        if history and history[0].get('is_current'):
            curr_job = history[0]
            curr_title = curr_job['title'].lower()
            is_lead = any(w in curr_title for w in ['lead', 'manager', 'director', 'architect'])
            is_coder = any(w in curr_title for w in ['engineer', 'developer', 'scientist', 'programmer', 'coder'])
            if is_lead and not is_coder and curr_job['duration_months'] >= 18:
                failure_penalties += 0.2
                
        failure_mode_score = max(0.0, 1.0 - failure_penalties)
        
        # 5. HIRE_RISK_SCORE
        prob_of_mishire = 1.0 - (career_fit_score * 0.5 + failure_mode_score * 0.5)
        expected_loss = prob_of_mishire * 8.0
        
        records.append({
            'candidate_id': cid,
            'career_fit_score': career_fit_score,
            'honeypot_flag': bool(honeypot_flag),
            'behavioral_availability_score': float(availability),
            'failure_mode_score': failure_mode_score,
            'expected_loss_inr_lakhs': expected_loss,
            'mishre_probability': prob_of_mishire
        })
        
    df = pd.DataFrame(records)
    
    # Print the requested statistics
    print("================== Phase 1 Stats ==================")
    print(f"Total candidates processed: {len(df)}")
    print(f"Total honeypots flagged: {df['honeypot_flag'].sum()}")
    print(f"  - Impossible tenure: {impossible_tenure_count}")
    print(f"  - Expert skill zero endorsements/duration: {expert_skill_zero_count}")
    print(f"  - Years of experience mismatch: {yoe_mismatch_count}")
    print(f"  - Skills overload (uncorrogorated > 8): {skills_overload_count}")
    print(f"  - Unrelated titles: {title_unrelated_count}")
    print(f"Average career_fit_score: {df['career_fit_score'].mean():.4f}")
    print(f"Average behavioral_availability_score: {df['behavioral_availability_score'].mean():.4f}")
    print(f"Candidates with open_to_work_flag = True: {open_to_work_count}")
    print(f"Candidates located in India: {in_india_count}")
    print("===================================================")
    
    print(f"Saving extracted features to {features_output_path}...")
    df.to_parquet(features_output_path, index=False)
    print("Phase 1 Offline Feature Extraction complete!")

if __name__ == '__main__':
    main()
