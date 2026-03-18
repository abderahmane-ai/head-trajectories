"""
Unit tests for probing/scores.py — all five scoring functions.
"""

import pytest
import torch
import numpy as np
from probing.scores import (
    sink_score,
    prev_token_score,
    induction_score,
    positional_score,
    semantic_score,
    score_head,
)


class TestSinkScore:
    """Test sink_score function."""
    
    def test_perfect_sink(self):
        """A head that always attends to position 0 should score ~1.0."""
        N, T = 10, 16
        attn = torch.zeros(N, T, T)
        attn[:, :, 0] = 1.0  # All attention to position 0
        
        score = sink_score(attn)
        assert score > 0.95, f"Perfect sink should score >0.95, got {score}"
    
    def test_uniform_attention(self):
        """Uniform attention should score ~1/T."""
        N, T = 10, 16
        attn = torch.ones(N, T, T) / T  # Uniform distribution
        
        score = sink_score(attn)
        expected = 1.0 / T
        assert abs(score - expected) < 0.05, f"Uniform should score ~{expected}, got {score}"
    
    def test_shape_validation(self):
        """Test with various input shapes."""
        for N, T in [(5, 8), (20, 32), (100, 64)]:
            attn = torch.softmax(torch.randn(N, T, T), dim=-1)
            score = sink_score(attn)
            assert 0.0 <= score <= 1.0, f"Score out of range: {score}"


class TestPrevTokenScore:
    """Test prev_token_score function."""
    
    def test_perfect_prev_token(self):
        """A head that always attends to t-1 should score 1.0."""
        N, T = 10, 16
        attn = torch.zeros(N, T, T)
        # Set diagonal below main diagonal to 1.0
        for i in range(1, T):
            attn[:, i, i-1] = 1.0
        
        score = prev_token_score(attn)
        assert score > 0.95, f"Perfect prev-token should score >0.95, got {score}"
    
    def test_no_prev_token(self):
        """A head that never attends to t-1 should score ~0.0."""
        N, T = 10, 16
        attn = torch.zeros(N, T, T)
        # Attend to position 0 only (not t-1)
        attn[:, :, 0] = 1.0
        
        score = prev_token_score(attn)
        assert score < 0.1, f"No prev-token should score <0.1, got {score}"
    
    def test_uniform_attention(self):
        """Uniform attention should score ~1/T."""
        N, T = 10, 16
        attn = torch.ones(N, T, T)
        # Make causal
        attn = torch.tril(attn)
        # Normalize rows
        attn = attn / attn.sum(dim=-1, keepdim=True)
        
        score = prev_token_score(attn)
        # For uniform causal, probability of t-1 varies by position
        # Average should be reasonable
        assert 0.0 <= score <= 1.0, f"Score out of range: {score}"


class TestInductionScore:
    """Test induction_score function."""
    
    def test_perfect_induction(self):
        """Perfect induction head should score 1.0."""
        N, T = 10, 64
        attn = torch.zeros(N, T, T)
        
        # Create probe indices
        p1 = torch.randint(5, 20, (N,))
        p2 = torch.randint(25, 40, (N,))
        
        # Set perfect induction: at p2, attend to p1+1
        for i in range(N):
            attn[i, p2[i], p1[i] + 1] = 1.0
        
        score = induction_score(attn, p1, p2)
        assert score > 0.95, f"Perfect induction should score >0.95, got {score}"
    
    def test_no_induction(self):
        """No induction behavior should score ~0.0."""
        N, T = 10, 64
        attn = torch.zeros(N, T, T)
        attn[:, :, 0] = 1.0  # Always attend to position 0
        
        p1 = torch.randint(5, 20, (N,))
        p2 = torch.randint(25, 40, (N,))
        
        score = induction_score(attn, p1, p2)
        assert score < 0.1, f"No induction should score <0.1, got {score}"
    
    def test_bounds_checking(self):
        """Test that out-of-bounds indices are handled gracefully."""
        N, T = 10, 64
        attn = torch.softmax(torch.randn(N, T, T), dim=-1)
        
        # Create valid indices (N must match attn.shape[0])
        p1 = torch.randint(5, 20, (N,))
        p2 = torch.randint(25, 40, (N,))
        
        # Set some to out of bounds
        p1[5] = 60  # Out of bounds
        p2[5] = 65  # Out of bounds
        
        score = induction_score(attn, p1, p2)
        assert 0.0 <= score <= 1.0, f"Score out of range: {score}"


class TestPositionalScore:
    """Test positional_score function."""
    
    def test_identical_patterns(self):
        """Identical attention patterns should score 1.0."""
        N, T = 10, 32
        # Create one pattern
        pattern = torch.softmax(torch.randn(T, T), dim=-1)
        
        # Duplicate it for all sequences
        attn = pattern.unsqueeze(0).expand(N, T, T).clone()
        
        # Create pairs: (0,1), (2,3), (4,5), (6,7), (8,9)
        pairs = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])
        
        score = positional_score(attn, pairs)
        assert score > 0.95, f"Identical patterns should score >0.95, got {score}"
    
    def test_random_patterns(self):
        """Random patterns should score low."""
        N, T = 10, 32
        attn = torch.softmax(torch.randn(N, T, T), dim=-1)
        
        pairs = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])
        
        score = positional_score(attn, pairs)
        # Random patterns should have low similarity
        assert 0.0 <= score <= 1.0, f"Score out of range: {score}"
    
    def test_content_dependent(self):
        """Content-dependent patterns should score low."""
        N, T = 10, 32
        attn = torch.zeros(N, T, T)
        
        # Make each sequence attend to different positions
        for i in range(N):
            attn[i, :, i % T] = 1.0
        
        pairs = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])
        
        score = positional_score(attn, pairs)
        assert score < 0.5, f"Content-dependent should score <0.5, got {score}"


class TestSemanticScore:
    """Test semantic_score function."""
    
    def test_perfect_semantic_alignment(self):
        """Attention perfectly aligned with cosine similarity should score high."""
        N, T, V, D = 20, 32, 1000, 128
        torch.manual_seed(42)
        
        token_ids = torch.randint(0, V, (N, T))
        embedding_matrix = torch.randn(V, D)
        
        # Normalize embeddings
        import torch.nn.functional as F
        embed_norm = F.normalize(embedding_matrix, dim=-1)
        
        # Create attention that matches cosine similarity
        attn = torch.zeros(N, T, T)
        for s in range(N):
            seq_embeds = embed_norm[token_ids[s]]
            sim_matrix = torch.mm(seq_embeds, seq_embeds.t())
            # Make causal and normalize
            sim_matrix = torch.tril(sim_matrix)
            sim_matrix = sim_matrix / (sim_matrix.sum(dim=-1, keepdim=True) + 1e-9)
            attn[s] = sim_matrix
        
        score = semantic_score(attn, token_ids, embedding_matrix)
        # Should be high positive correlation
        assert score > 0.5, f"Perfect alignment should score >0.5, got {score}"
    
    def test_random_attention(self):
        """Random attention should have low correlation with semantics."""
        N, T, V, D = 20, 32, 1000, 128
        torch.manual_seed(42)
        
        token_ids = torch.randint(0, V, (N, T))
        embedding_matrix = torch.randn(V, D)
        attn = torch.softmax(torch.randn(N, T, T), dim=-1)
        
        score = semantic_score(attn, token_ids, embedding_matrix)
        # Random should be near zero
        assert -0.5 <= score <= 0.5, f"Random should be near 0, got {score}"
    
    def test_vectorization_correctness(self):
        """Test that vectorized implementation produces consistent results."""
        N, T, V, D = 50, 64, 1000, 128
        torch.manual_seed(123)
        
        token_ids = torch.randint(0, V, (N, T))
        embedding_matrix = torch.randn(V, D)
        attn = torch.softmax(torch.randn(N, T, T), dim=-1)
        
        # Run twice with same inputs
        score1 = semantic_score(attn, token_ids, embedding_matrix)
        score2 = semantic_score(attn, token_ids, embedding_matrix)
        
        assert abs(score1 - score2) < 1e-6, "Should be deterministic"
    
    def test_handles_degenerate_cases(self):
        """Test that degenerate cases don't crash."""
        N, T, V, D = 10, 8, 100, 64
        
        # All same token
        token_ids = torch.zeros(N, T, dtype=torch.long)
        embedding_matrix = torch.randn(V, D)
        attn = torch.softmax(torch.randn(N, T, T), dim=-1)
        
        score = semantic_score(attn, token_ids, embedding_matrix)
        assert np.isfinite(score), f"Should handle degenerate case, got {score}"


class TestScoreHead:
    """Test the batch scorer that computes all 5 scores."""
    
    def test_score_head_returns_five_scores(self):
        """Test that score_head returns exactly 5 scores."""
        N_gen, N_ind, N_pos, T = 50, 10, 10, 64
        V, D = 1000, 128
        
        torch.manual_seed(42)
        
        general_attn = torch.softmax(torch.randn(N_gen, T, T), dim=-1)
        induction_attn = torch.softmax(torch.randn(N_ind, T, T), dim=-1)
        positional_attn = torch.softmax(torch.randn(N_pos, T, T), dim=-1)
        
        induction_p1 = torch.randint(5, 20, (N_ind,))
        induction_p2 = torch.randint(25, 40, (N_ind,))
        positional_pairs = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])
        
        token_ids = torch.randint(0, V, (N_gen, T))
        embedding_matrix = torch.randn(V, D)
        
        scores = score_head(
            general_attn,
            induction_attn,
            positional_attn,
            induction_p1,
            induction_p2,
            positional_pairs,
            token_ids,
            embedding_matrix,
        )
        
        assert len(scores) == 5, f"Should return 5 scores, got {len(scores)}"
        assert all(isinstance(s, float) for s in scores), "All scores should be floats"
        assert all(np.isfinite(s) for s in scores), "All scores should be finite"
    
    def test_score_head_ranges(self):
        """Test that all scores are in reasonable ranges."""
        N_gen, N_ind, N_pos, T = 100, 20, 20, 64
        V, D = 1000, 128
        
        torch.manual_seed(777)
        
        general_attn = torch.softmax(torch.randn(N_gen, T, T), dim=-1)
        induction_attn = torch.softmax(torch.randn(N_ind, T, T), dim=-1)
        positional_attn = torch.softmax(torch.randn(N_pos, T, T), dim=-1)
        
        induction_p1 = torch.randint(5, 20, (N_ind,))
        induction_p2 = torch.randint(25, 40, (N_ind,))
        positional_pairs = torch.tensor([[i, i+1] for i in range(0, N_pos, 2)])
        
        token_ids = torch.randint(0, V, (N_gen, T))
        embedding_matrix = torch.randn(V, D)
        
        sink, prev, ind, pos, sem = score_head(
            general_attn,
            induction_attn,
            positional_attn,
            induction_p1,
            induction_p2,
            positional_pairs,
            token_ids,
            embedding_matrix,
        )
        
        # Sink, prev, ind, pos should be in [0, 1]
        assert 0.0 <= sink <= 1.0, f"Sink score out of range: {sink}"
        assert 0.0 <= prev <= 1.0, f"Prev score out of range: {prev}"
        assert 0.0 <= ind <= 1.0, f"Induction score out of range: {ind}"
        assert 0.0 <= pos <= 1.0, f"Positional score out of range: {pos}"
        
        # Semantic can be negative (correlation)
        assert -1.0 <= sem <= 1.0, f"Semantic score out of range: {sem}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
