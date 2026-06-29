import json
import re
import hashlib
from datetime import datetime
import gradio as gr
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Constants matching rank.py & extract_features.py
REF_DATE = datetime(2026, 5, 28)
NEW_AI_COMPANIES = {'Krutrim', 'Sarvam AI'}
CONSULTING_COMPANIES = {'tcs', 'wipro', 'infosys', 'accenture', 'cognizant', 'capgemini', 'mindtree'}
STOPWORDS = {'and', 'or', 'of', 'in', 'at', 'the', 'for', 'a', 'an', 'with', 'to', 'on', 'by', 'using', 'from', 'systems', 'platform', 'framework', '&', '-', '/'}

# Research & Domain keywords for gap / failure mode detection
RESEARCH_KEYWORDS = {'research', 'paper', 'publication', 'academic', 'university', 'thesis', 'conference', 'arxiv'}
WRONG_DOMAIN_KEYWORDS = {'computer vision', 'image classification', 'object detection', 'cnn', 'speech recognition', 'tts', 'robotics', 'nlp', 'text-to-speech', 'speech-to-text', 'computer-vision'}
RIGHT_DOMAIN_KEYWORDS = {'search', 'retrieval', 'ranking', 'recommendation', 'recommender', 'information retrieval', 'ir', 'ltr', 'learning to rank', 'embed', 'vector', 'pinecone', 'milvus', 'weaviate', 'qdrant', 'faiss', 'bm25'}

def get_words(t):
    if not t:
        return set()
    return {w for w in re.findall(r'\b\w+\b', t.lower())}

def is_mentioned(skill_name, descriptions_text):
    skill_name = skill_name.lower().strip()
    descriptions_text = descriptions_text.lower()
    if skill_name in descriptions_text:
        return True
    if skill_name.endswith('s') and skill_name[:-1] in descriptions_text:
        return True
    if skill_name + 's' in descriptions_text:
        return True
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

def generate_reasoning(rank, row, profile, history, skills, signals):
    yoe = profile['years_of_experience']
    title = profile['current_title']
    
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
    
    gaps = []
    desc_text = " ".join([job['description'] or "" for job in history]).lower()
    
    if row['failure_mode_score'] < 1.0:
        is_pure_services = all(any(c in job['company'].lower() for c in CONSULTING_COMPANIES) for job in history)
        if is_pure_services:
            gaps.append("services-only consulting background")
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
    
    pos_signals = []
    if signals['open_to_work_flag']:
        pos_signals.append("open-to-work status")
    if signals['recruiter_response_rate'] > 0.8:
        pos_signals.append("high responsiveness")
    if signals['last_active_date'] >= '2026-05-01':
        pos_signals.append("recent active status")
    pos_signal_str = " and ".join(pos_signals[:2]) if pos_signals else "verified email/phone"
    
    if rank <= 1:
        templates = [
            f"Outstanding fit with {yoe:.1f} years of experience as a {title} at {first_company}, specializing in {matching_skills_str}. Strong match for founding-team pace, backed by high availability.",
            f"Highly qualified candidate with {yoe:.1f} years building systems at product companies like {first_company}. Expertise in {matching_skills_str} directly meets core JD requirements."
        ]
    elif rank <= 2:
        templates = [
            f"Solid ML background with {yoe:.1f} years of experience and core skills in {matching_skills_str}. Gaps such as {gaps_str} represent minor ramp-up areas.",
            f"Strong profile showing {yoe:.1f} years in {title} roles at {first_company}. Core capabilities in {matching_skills_str} align well, though {gaps_str} warrants consideration."
        ]
    else:
        templates = [
            f"Peripheral skills overlap only in {matching_skills_str} across {yoe:.1f} years of experience. Included as a boundary candidate given {pos_signal_str}.",
            f"Under-experienced or showing gaps like {gaps_str}, but listed as a boundary candidate due to background in {matching_skills_str} and {pos_signal_str}."
        ]
        
    h = int(hashlib.md5(profile['anonymized_name'].encode('utf-8')).hexdigest(), 16)
    reasoning = templates[h % len(templates)]
    return reasoning

def process_candidates(candidates_json, jd_text):
    try:
        candidates = json.loads(candidates_json)
    except Exception as e:
        return pd.DataFrame([{"Error": f"Invalid JSON format: {str(e)}"}])
        
    if not isinstance(candidates, list):
        return pd.DataFrame([{"Error": "Input must be a JSON array of candidate objects."}])
        
    # Build candidate narratives for TF-IDF
    narratives = []
    for cand in candidates:
        profile = cand['profile']
        history = cand['career_history']
        narrative = f"Current title: {profile['current_title']}. Headline: {profile['headline']}. Summary: {profile['summary']}. "
        narrative += " ".join([f"Job title: {job['title']} at {job['company']}. Job description: {job['description']}" for job in history])
        narratives.append(narrative)
        
    # Calculate TF-IDF Semantic Career Fit Score
    try:
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([jd_text] + narratives)
        similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    except Exception as e:
        similarities = [0.5] * len(candidates)
        
    records = []
    for idx, cand in enumerate(candidates):
        cid = cand['candidate_id']
        profile = cand['profile']
        history = cand['career_history']
        skills = cand['skills']
        signals = cand['redrob_signals']
        
        # 1. CAREER_FIT_SCORE
        sim = float(similarities[idx])
        companies = {job['company'].lower() for job in history}
        is_product_only = not (companies & CONSULTING_COMPANIES)
        boost = 0.10 if is_product_only else 0.0
        
        desc_text = " ".join([job['description'] or "" for job in history]).lower()
        has_research = any(w in desc_text for w in RESEARCH_KEYWORDS)
        production_words = {'production', 'deploy', 'ship', 'user', 'scale', 'system'}
        has_production = any(w in desc_text for w in production_words)
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
            
        honeypot_flag = impossible_tenure or expert_skill_zero or yoe_mismatch or skills_overload
        
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
            
        has_wrong_domain = any(kw in desc_text for w in WRONG_DOMAIN_KEYWORDS)
        has_right_domain = any(kw in desc_text for w in RIGHT_DOMAIN_KEYWORDS)
        if has_wrong_domain and not has_right_domain:
            failure_penalties += 0.3
            
        failure_mode_score = max(0.0, 1.0 - failure_penalties)
        
        # 5. HIRE_RISK_SCORE
        prob_of_mishire = 1.0 - (career_fit_score * 0.5 + failure_mode_score * 0.5)
        expected_loss = prob_of_mishire * 8.0
        
        # Final composite score
        score = (
            0.40 * career_fit_score
            + 0.25 * failure_mode_score
            + 0.20 * availability
            + 0.15 * (1.0 - (expected_loss / 8.0))
        ) * (0.0 if honeypot_flag else 1.0)
        
        records.append({
            'candidate_id': cid,
            'score': float(score),
            'career_fit_score': career_fit_score,
            'failure_mode_score': failure_mode_score,
            'availability_score': availability,
            'expected_loss_lakhs': expected_loss,
            'honeypot_flag': honeypot_flag,
            'profile': profile,
            'history': history,
            'skills': skills,
            'signals': signals
        })
        
    # Sort and rank
    df = pd.DataFrame(records)
    df = df.sort_values(by=['score', 'candidate_id'], ascending=[False, True]).reset_index(drop=True)
    
    ranked_rows = []
    for rank_idx, row in df.iterrows():
        rank = rank_idx + 1
        reasoning = generate_reasoning(
            rank=rank,
            row=row,
            profile=row['profile'],
            history=row['history'],
            skills=row['skills'],
            signals=row['signals']
        )
        if row['honeypot_flag']:
            reasoning = "FILTERED: Candidate flagged as honeypot due to timeline anomalies or uncorroborated expert skills."
            
        ranked_rows.append({
            "Rank": rank,
            "Candidate ID": row['candidate_id'],
            "Score": f"{row['score']:.4f}",
            "Honeypot": "True (Score Zeroed)" if row['honeypot_flag'] else "False",
            "Reasoning": reasoning
        })
        
    return pd.DataFrame(ranked_rows)

# Default Sample JSON
default_candidates_json = """[
  {
    "candidate_id": "CAND_0000031",
    "profile": {
      "anonymized_name": "Ela Singh",
      "current_title": "Recommendation Systems Engineer",
      "headline": "ML Engineer specializing in Recommendation Systems",
      "summary": "Building scalable search and recommendation systems.",
      "years_of_experience": 6.0,
      "location": "Bangalore, India",
      "country": "India"
    },
    "skills": [
      {"name": "Embeddings", "proficiency": "expert", "endorsements": 10, "duration_months": 36},
      {"name": "Information Retrieval", "proficiency": "expert", "endorsements": 8, "duration_months": 24},
      {"name": "Python", "proficiency": "expert", "endorsements": 15, "duration_months": 72}
    ],
    "career_history": [
      {
        "company": "Swiggy",
        "title": "Recommendation Systems Engineer",
        "start_date": "2023-01-01",
        "end_date": "2026-05-27",
        "duration_months": 41,
        "is_current": true,
        "description": "Developed search and ranking layers, deploying vector search with FAISS and hybrid search pipelines."
      }
    ],
    "redrob_signals": {
      "last_active_date": "2026-05-25",
      "open_to_work_flag": true,
      "recruiter_response_rate": 0.95,
      "notice_period_days": 30,
      "interview_completion_rate": 0.9,
      "profile_completeness_score": 100,
      "offer_acceptance_rate": 0.8,
      "verified_email": true,
      "verified_phone": true,
      "github_activity_score": 85,
      "willing_to_relocate": true
    }
  },
  {
    "candidate_id": "CAND_Honeypot",
    "profile": {
      "anonymized_name": "Fake Candidate",
      "current_title": "AI Architect",
      "headline": "Expert in all AI fields",
      "summary": "10 years of experience in generative AI and LLMs.",
      "years_of_experience": 5.0,
      "location": "Mumbai, India",
      "country": "India"
    },
    "skills": [
      {"name": "LangChain", "proficiency": "expert", "endorsements": 0, "duration_months": 60},
      {"name": "LlamaIndex", "proficiency": "expert", "endorsements": 0, "duration_months": 48}
    ],
    "career_history": [
      {
        "company": "Sarvam AI",
        "title": "AI Architect",
        "start_date": "2018-01-01",
        "end_date": "2023-01-01",
        "duration_months": 60,
        "is_current": false,
        "description": "Built LLM agents and retrieval pipelines."
      }
    ],
    "redrob_signals": {
      "last_active_date": "2026-05-20",
      "open_to_work_flag": true,
      "recruiter_response_rate": 0.80,
      "notice_period_days": 60,
      "interview_completion_rate": 0.8,
      "profile_completeness_score": 90,
      "offer_acceptance_rate": 0.7,
      "verified_email": true,
      "verified_phone": true,
      "github_activity_score": 40,
      "willing_to_relocate": false
    }
  },
  {
    "candidate_id": "CAND_0035315",
    "profile": {
      "anonymized_name": "Aarav Sharma",
      "current_title": "Data Scientist",
      "headline": "Data Scientist at Wipro",
      "summary": "Experience building machine learning models.",
      "years_of_experience": 3.3,
      "location": "Pune, India",
      "country": "India"
    },
    "skills": [
      {"name": "Python", "proficiency": "advanced", "endorsements": 5, "duration_months": 36},
      {"name": "Semantic Search", "proficiency": "intermediate", "endorsements": 2, "duration_months": 12}
    ],
    "career_history": [
      {
        "company": "Wipro",
        "title": "Data Scientist",
        "start_date": "2023-01-01",
        "end_date": "2026-05-27",
        "duration_months": 41,
        "is_current": true,
        "description": "Developed predictive models for clients. Background is mostly services."
      }
    ],
    "redrob_signals": {
      "last_active_date": "2026-05-15",
      "open_to_work_flag": false,
      "recruiter_response_rate": 0.60,
      "notice_period_days": 120,
      "interview_completion_rate": 0.7,
      "profile_completeness_score": 85,
      "offer_acceptance_rate": 0.5,
      "verified_email": true,
      "verified_phone": false,
      "github_activity_score": -1,
      "willing_to_relocate": true
    }
  }
]"""

default_jd = (
    "Production ML systems at product companies. Experience with embeddings, vector search, "
    "ranking, and retrieval systems. Strong Python programming. Designing and deploying "
    "evaluation frameworks (NDCG, MRR, MAP) for search or recommendation systems."
)

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="indigo")) as demo:
    gr.Markdown("# RankIQ Sandbox Demo")
    gr.Markdown(
        "**[DEMO MODE]** *This is a lightweight interactive sandbox simulating the RankIQ evaluation pipeline. "
        "It uses a CPU-friendly TF-IDF representation for semantic similarity instead of the full Sentence-Transformers model to run instantly without downloading heavy dependencies.*"
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            jd_input = gr.Textbox(
                label="Job Description (JD)",
                value=default_jd,
                lines=5,
                placeholder="Enter job description..."
            )
            candidates_input = gr.Textbox(
                label="Candidates JSON Array (3-5 objects)",
                value=default_candidates_json,
                lines=15,
                placeholder="Enter candidates JSON..."
            )
            submit_btn = gr.Button("Rank Candidates", variant="primary")
            
        with gr.Column(scale=1.5):
            gr.Markdown("### Ranked Output Table")
            output_table = gr.Dataframe(
                headers=["Rank", "Candidate ID", "Score", "Honeypot", "Reasoning"],
                datatype=["str", "str", "str", "str", "str"],
                wrap=True
            )
            
    submit_btn.click(
        fn=process_candidates,
        inputs=[candidates_input, jd_input],
        outputs=output_table
    )

if __name__ == "__main__":
    demo.launch()
