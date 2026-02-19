# NLP Guide:

## How to run it 
```bash
# Step 1: install dependencies
pip install -r requirements_phase1.txt

# Step 2: run (expects your processed_data/ and MINDlarge_train/ in current dir)
python run_phase1_encoding.py

# Optional: override paths without editing the file
TRAIN_DIR=./MINDsmall_train EMBEDDINGS_DIR=./embeddings_small python run_phase1_encoding.py
```
On CPU with MINDlarge (~101K articles), expect ~30-40 minutes — almost all of that is SBERT. Run it once overnight and you never run it again. Every downstream phase just calls NewsEncoderPhase1.load('./embeddings').

## What Phase 2 will look like:
```bash 
from nlp_encoder import NewsEncoderPhase1, AttentionUserProfiler

encoder  = NewsEncoderPhase1.load('./embeddings')   # ~2s load
profiler = AttentionUserProfiler(decay_lambda=0.3)

# For each recommendation request:
profile   = profiler.build_profile(user_history_ids, encoder.final_embeddings,
                                    encoder.news_id_to_idx)
cand_ids, scores = encoder.retriever.retrieve(profile, k=100,
                                               exclude_ids=user_history_ids)
# → feeds directly into Phase 2 hybrid scorer, then Phase 4 re-ranker
```