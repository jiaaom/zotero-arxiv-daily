import numpy as np
from sentence_transformers import SentenceTransformer
from paper import ArxivPaper
from datetime import datetime

def rerank_paper(candidate:list[ArxivPaper],corpus:list[dict],model:str='google/embeddinggemma-300m', diversity_lambda:float=0.3) -> list[ArxivPaper]:
    encoder = SentenceTransformer(model)
    #sort corpus by date, from newest to oldest
    corpus = sorted(corpus,key=lambda x: datetime.strptime(x['data']['dateAdded'], '%Y-%m-%dT%H:%M:%SZ'),reverse=True)
    time_decay_weight = 1 / (1 + np.log10(np.arange(len(corpus)) + 1))
    time_decay_weight = time_decay_weight / time_decay_weight.sum()
    corpus_feature = encoder.encode([paper['data']['abstractNote'] for paper in corpus])
    candidate_feature = encoder.encode([paper.summary for paper in candidate])
    sim = encoder.similarity(candidate_feature,corpus_feature) # [n_candidate, n_corpus]
    scores = (sim * time_decay_weight).sum(axis=1) * 10 # [n_candidate]

    # Apply diversity-aware ranking (MMR-style)
    if len(candidate) > 1 and diversity_lambda > 0:
        candidate = _apply_diversity_ranking(candidate, candidate_feature, scores, diversity_lambda)
    else:
        # Original ranking when diversity is disabled or only one candidate
        for s,c in zip(scores,candidate):
            c.score = s.item()
        candidate = sorted(candidate,key=lambda x: x.score,reverse=True)

    return candidate

def _apply_diversity_ranking(candidate: list[ArxivPaper], candidate_feature: np.ndarray,
                           relevance_scores: np.ndarray, diversity_lambda: float) -> list[ArxivPaper]:
    """Apply diversity-aware ranking using MMR-style algorithm"""
    n_candidates = len(candidate)
    selected_indices = []
    remaining_indices = list(range(n_candidates))

    # Compute pairwise similarities between candidates
    candidate_sim = np.dot(candidate_feature, candidate_feature.T)

    for _ in range(n_candidates):
        if not remaining_indices:
            break

        best_score = -float('inf')
        best_idx = None

        for idx in remaining_indices:
            # Relevance component
            relevance = relevance_scores[idx]

            # Diversity component (max similarity to already selected papers)
            if selected_indices:
                max_sim_to_selected = max(candidate_sim[idx][sel_idx] for sel_idx in selected_indices)
            else:
                max_sim_to_selected = 0

            # MMR score: balance relevance and diversity
            mmr_score = diversity_lambda * relevance - (1 - diversity_lambda) * max_sim_to_selected * 10

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx is not None:
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
            candidate[best_idx].score = best_score

    # Return papers in selection order (highest MMR scores first)
    reordered_candidate = [candidate[idx] for idx in selected_indices]

    return reordered_candidate