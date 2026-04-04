import torch

from data.probe import build_natural_induction_probes
from probing.scores import induction_score, natural_induction_score

def test_perfect_induction():
    N = 10
    T = 64
    
    # Randomly pick first occurrence (p1) and second occurrence (p2)
    # p1 can be from index 5 to 20
    # p2 can be from index 30 to 50
    induction_p1 = torch.randint(5, 20, (N,))
    induction_p2 = torch.randint(30, 50, (N,))
    
    # Create an attention head tensor (N, T, T)
    # Initialize with uniform causal attention just so it sums to 1
    attn_head = torch.zeros(N, T, T)
    for t in range(T):
        attn_head[:, t, :t+1] = 1.0 / (t + 1)
        
    # Now, explicitly "inject" perfect induction behavior into the queries at p2
    for i in range(N):
        p1 = induction_p1[i].item()
        p2 = induction_p2[i].item()
        
        # At query position p2, perfect induction means attending 100% to p1 + 1
        # Clear out the row first
        attn_head[i, p2, :] = 0.0
        # Set the induction target to 1.0
        attn_head[i, p2, p1 + 1] = 1.0
        
    # Verify the rows still sum to 1.0
    assert torch.allclose(attn_head.sum(dim=-1), torch.ones(N, T)), "Attention rows do not sum to 1"
    
    # Compute score
    score = induction_score(attn_head, induction_p1, induction_p2)
    
    print(f"Synthetic perfect induction score: {score:.4f}")
    assert abs(score - 1.0) < 1e-6, f"Expected 1.0, got {score}"
    
    # Check what happens if a head explicitly avoids induction
    # i.e., at query p2, it attends 0% to p1+1
    for i in range(N):
        p1 = induction_p1[i].item()
        p2 = induction_p2[i].item()
        
        attn_head[i, p2, :] = 1.0 / (p2)
        attn_head[i, p2, p1 + 1] = 0.0
        # redistribute the mass
        attn_head[i, p2] = attn_head[i, p2] / attn_head[i, p2].sum()
        
    bad_score = induction_score(attn_head, induction_p1, induction_p2)
    print(f"Synthetic anti-induction score: {bad_score:.4f}")
    assert abs(bad_score - 0.0) < 1e-6, f"Expected 0.0, got {bad_score}"
    
    # --- PARTIAL INDUCTION CASE (50%) ---
    # 50% of sequences perfectly induce, 50% attend uniformly to past context
    for i in range(N):
        p1 = induction_p1[i].item()
        p2 = induction_p2[i].item()
        
        if i % 2 == 0:
            # Perfect induction
            attn_head[i, p2, :] = 0.0
            attn_head[i, p2, p1 + 1] = 1.0
        else:
            # Uniform attention to all past context including p2
            attn_head[i, p2, :] = 0.0
            attn_head[i, p2, :p2 + 1] = 1.0 / (p2 + 1)

    partial_score = induction_score(attn_head, induction_p1, induction_p2)
    
    # For roughly 50% perfect induction and 50% uniform,
    # Sequence average should be approx 0.5 + 0.5 * (1/p2) which is very close to 0.5.
    print(f"Synthetic partial induction score (50%): {partial_score:.4f}")
    assert abs(partial_score - 0.5) < 0.05, f"Expected ~0.5, got {partial_score}"
    
    print("SUCCESS: induction_score perfectly detects induction head behavior!")


def test_natural_induction_score_matches_core_metric():
    N = 4
    T = 32
    induction_p1 = torch.tensor([4, 5, 6, 7], dtype=torch.long)
    induction_p2 = torch.tensor([16, 17, 18, 19], dtype=torch.long)
    attn_head = torch.zeros(N, T, T)
    for t in range(T):
        attn_head[:, t, :t + 1] = 1.0 / (t + 1)
    for i in range(N):
        attn_head[i, induction_p2[i], :] = 0.0
        attn_head[i, induction_p2[i], induction_p1[i] + 1] = 1.0

    assert natural_induction_score(attn_head, induction_p1, induction_p2) == induction_score(
        attn_head, induction_p1, induction_p2
    )


def test_build_natural_induction_probes_allow_partial_returns_partial_set():
    raw_sequences = []
    block_size = 64
    for i in range(20):
        seq = list(range(block_size))
        if i < 6:
            start1 = 5
            start2 = 35
            seq[start2 : start2 + 5] = seq[start1 : start1 + 5]
        raw_sequences.append(seq)

    seqs, p1, p2 = build_natural_induction_probes(
        raw_sequences,
        n_probes=10,
        block_size=block_size,
        seed=0,
        allow_partial=True,
        min_probes=4,
    )

    assert seqs.shape[0] == 6
    assert p1.shape[0] == 6
    assert p2.shape[0] == 6

if __name__ == "__main__":
    test_perfect_induction()
