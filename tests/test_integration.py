"""
Integration tests — end-to-end workflows.
"""

import pytest
import torch
import tempfile
from pathlib import Path
from model import TransformerLM, ModelConfig
from probing.extractor import extract_attention_maps
from probing.scores import score_head
from probing.classifier import HeadClassifier


class TestEndToEndProbing:
    """Test the full probing pipeline on a small model."""
    
    def test_extract_and_score_workflow(self):
        """Test extracting attention and scoring a head."""
        # Create small model
        config = ModelConfig(
            n_layers=2,
            n_heads=4,
            d_model=64,
            d_ffn=256,
            block_size=32,
            vocab_size=1000,
        )
        model = TransformerLM(config)
        model.eval()
        
        # Create fake probe sequences
        N_gen, N_ind, N_pos, T = 20, 5, 5, 32
        general_seqs = torch.randint(0, 1000, (N_gen, T))
        induction_seqs = torch.randint(0, 1000, (N_ind, T))
        positional_seqs = torch.randint(0, 1000, (N_pos, T))
        
        # Extract attention maps
        with torch.no_grad():
            general_maps = extract_attention_maps(model, general_seqs, torch.device('cpu'), batch_size=10)
            induction_maps = extract_attention_maps(model, induction_seqs, torch.device('cpu'), batch_size=5)
            positional_maps = extract_attention_maps(model, positional_seqs, torch.device('cpu'), batch_size=5)
        
        assert len(general_maps) == config.n_layers
        assert general_maps[0].shape == (N_gen, config.n_heads, T, T)
        
        # Score one head
        induction_p1 = torch.randint(5, 10, (N_ind,))
        induction_p2 = torch.randint(15, 20, (N_ind,))
        positional_pairs = torch.tensor([[0, 1], [2, 3]])
        
        embedding_matrix = model.get_embedding_matrix()
        
        scores = score_head(
            general_attn=general_maps[0][:, 0, :, :],  # First head
            induction_attn=induction_maps[0][:, 0, :, :],
            positional_attn=positional_maps[0][:, 0, :, :],
            induction_p1=induction_p1,
            induction_p2=induction_p2,
            positional_pairs=positional_pairs,
            token_ids=general_seqs,
            embedding_matrix=embedding_matrix,
        )
        
        assert len(scores) == 5
        assert all(isinstance(s, float) for s in scores)
    
    def test_full_classification_workflow(self):
        """Test the full classification workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ties_path = Path(tmpdir) / "ties.csv"
            output_path = Path(tmpdir) / "results.pt"
            
            # Create classifier
            n_ckpts, n_layers, n_heads = 3, 2, 4
            classifier = HeadClassifier(
                n_checkpoints=n_ckpts,
                n_layers=n_layers,
                n_heads=n_heads,
                seed=42,
                ties_log_path=ties_path,
            )
            
            # Simulate scoring all heads at all checkpoints
            for ckpt in range(n_ckpts):
                classifier.register_step(ckpt * 1000)
                
                for layer in range(n_layers):
                    for head in range(n_heads):
                        # Generate random scores
                        scores = tuple(torch.rand(5).tolist())
                        classifier.record(ckpt, ckpt * 1000, layer, head, scores)
            
            # Save
            classifier.save(output_path)
            
            # Load and verify
            loaded = HeadClassifier.load(output_path)
            assert loaded["label_tensor"].shape == (n_ckpts, n_layers, n_heads)
            assert loaded["score_tensor"].shape == (n_ckpts, n_layers, n_heads, 5)
            assert len(loaded["step_index"]) == n_ckpts


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
