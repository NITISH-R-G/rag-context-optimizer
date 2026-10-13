# ruff: noqa: F821
# Whitelist for vulture to ignore false positives in FastAPI endpoints and Pydantic validators

# app.py endpoints
home_page
reset_endpoint
step_endpoint
state_endpoint
health_endpoint
tasks_endpoint
corpus_families_endpoint
optimize_step_endpoint
optimize_prompt_endpoint

# env/corpus.py
relevance_tags
get_chunks_by_domain
get_chunk_by_id

# env/llm_services.py
judge_answer

# env/models.py
model_config
validate_non_empty_text
cls
validate_keywords
case_id
progress_signals
last_action_feedback
validate_required_strings
validate_ids
validate_report_requirements
validate_feedback
validate_budget_and_aliases
normalize_optional_strings
validate_action_semantics
RagReward
answer_quality
retrieval_precision
penalty
validate_total_bound

# env/retriever.py
get_ground_truth_relevant

# env/tasks.py
objective_type

# tests/
kwargs
key
side_effect
other
pos_weight
lr
cross_encoder_score
embedding_similarity
q1
q2
return_value
do_POST
log_message
format
