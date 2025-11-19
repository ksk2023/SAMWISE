"""
AttentionCME: Attention-Weighted Conditional Memory Encoder
Enhanced CME module using cross-modal attention and gated fusion
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionCME(nn.Module):
    """
    Attention-based Conditional Memory Encoder using cross-modal attention and gated fusion.

    This module replaces the simple self-attention based CME with a more sophisticated
    approach that:
    1. Performs bidirectional cross-modal attention between memory and memoryless features
    2. Uses gated fusion to adaptively combine the two representations
    3. Provides interpretable attention maps and gate weights

    Args:
        dim (int): Feature dimension. Default: 256
        num_heads (int): Number of attention heads. Default: 8
        dropout (float): Dropout rate. Default: 0.1
    """
    def __init__(self, dim=256, num_heads=8, dropout=0.1):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads

        # Cross-modal attention: memory features attend to memoryless features
        self.cross_attn_m2l = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Cross-modal attention: memoryless features attend to memory features
        self.cross_attn_l2m = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Gated fusion mechanism
        # Takes concatenated features and produces adaptive weights
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )

        # Feature projection layers for better representation
        self.proj_mem = nn.Linear(dim, dim)
        self.proj_nomem = nn.Linear(dim, dim)

        # Decision head: binary classification (use memory vs. no memory)
        self.decision_head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.LayerNorm(dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, 2)
        )

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat_mem, feat_nomem):
        """
        Forward pass with cross-modal attention and gated fusion.

        Args:
            feat_mem: Features from memory-based decoder [B, dim] or [B, N, dim]
            feat_nomem: Features from memoryless decoder [B, dim] or [B, N, dim]

        Returns:
            decision_logits: Binary classification logits [B, 2]
            fused_features: Gated fusion of both representations [B, dim]
            attention_maps: Dictionary containing attention weights for analysis
            gate_weights: Gate values for interpretability [B, dim]
        """
        # Handle different input shapes
        if feat_mem.dim() == 2:
            feat_mem = feat_mem.unsqueeze(1)  # [B, 1, dim]
        if feat_nomem.dim() == 2:
            feat_nomem = feat_nomem.unsqueeze(1)  # [B, 1, dim]

        batch_size = feat_mem.size(0)

        # Project features
        feat_mem_proj = self.proj_mem(feat_mem)  # [B, N, dim]
        feat_nomem_proj = self.proj_nomem(feat_nomem)  # [B, N, dim]

        # Cross-modal attention: memory features attend to memoryless features
        # Query: memory, Key/Value: memoryless
        feat_m2l, attn_m2l = self.cross_attn_m2l(
            query=feat_mem_proj,
            key=feat_nomem_proj,
            value=feat_nomem_proj,
            need_weights=True
        )

        # Cross-modal attention: memoryless features attend to memory features
        # Query: memoryless, Key/Value: memory
        feat_l2m, attn_l2m = self.cross_attn_l2m(
            query=feat_nomem_proj,
            key=feat_mem_proj,
            value=feat_mem_proj,
            need_weights=True
        )

        # Pool to single vector per batch if needed
        feat_m2l_pooled = feat_m2l.mean(dim=1)  # [B, dim]
        feat_l2m_pooled = feat_l2m.mean(dim=1)  # [B, dim]

        # Gated fusion
        # Concatenate both cross-attended features
        concat_features = torch.cat([feat_m2l_pooled, feat_l2m_pooled], dim=-1)  # [B, dim*2]

        # Generate adaptive gate weights
        gate_weights = self.gate(concat_features)  # [B, dim]

        # Fused features: weighted combination
        # gate_weights controls the balance between memory and memoryless representations
        fused_features = gate_weights * feat_m2l_pooled + (1 - gate_weights) * feat_l2m_pooled

        # Decision logits
        decision_logits = self.decision_head(fused_features)  # [B, 2]

        # Prepare attention maps for analysis
        attention_maps = {
            'memory_to_memoryless': attn_m2l,  # [B, num_heads, N, N] or [B, N, N]
            'memoryless_to_memory': attn_l2m,
        }

        return decision_logits, fused_features, attention_maps, gate_weights


class AttentionCMEWithDecisionToken(AttentionCME):
    """
    Extended version of AttentionCME that uses a learnable decision token
    similar to the original CME implementation.
    """
    def __init__(self, dim=256, num_heads=8, dropout=0.1):
        super().__init__(dim=dim, num_heads=num_heads, dropout=dropout)

        # Learnable decision token
        self.decision_token = nn.Embedding(1, dim)

        # Self-attention block for processing with decision token
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm = nn.LayerNorm(dim)

    def forward(self, feat_mem, feat_nomem):
        """
        Forward pass with decision token integration.

        Args:
            feat_mem: Features from memory-based decoder [B, dim]
            feat_nomem: Features from memoryless decoder [B, dim]

        Returns:
            decision_logits: Binary classification logits [B, 2]
            fused_features: Gated fusion of both representations [B, dim]
            attention_maps: Dictionary containing attention weights
            gate_weights: Gate values [B, dim]
        """
        # Get base cross-modal attention results
        decision_logits, fused_features, attention_maps, gate_weights = super().forward(
            feat_mem, feat_nomem
        )

        batch_size = feat_mem.size(0) if feat_mem.dim() == 2 else feat_mem.size(0)

        # Add decision token
        decision_tok = self.decision_token.weight.unsqueeze(0).expand(batch_size, -1, -1)  # [B, 1, dim]

        # Prepare sequence: [decision_token, fused_features]
        fused_seq = fused_features.unsqueeze(1)  # [B, 1, dim]
        sequence = torch.cat([decision_tok, fused_seq], dim=1)  # [B, 2, dim]

        # Self-attention
        attended_seq, self_attn_weights = self.self_attn(
            query=sequence,
            key=sequence,
            value=sequence,
            need_weights=True
        )

        # Extract decision token output
        decision_output = attended_seq[:, 0, :]  # [B, dim]
        decision_output = self.norm(decision_output)

        # Final decision logits from decision token
        decision_logits = self.decision_head(decision_output)

        # Add self-attention weights to attention maps
        attention_maps['self_attention'] = self_attn_weights

        return decision_logits, fused_features, attention_maps, gate_weights


def build_attention_cme(args):
    """
    Build AttentionCME module based on args configuration.

    Args:
        args: Configuration arguments containing model hyperparameters

    Returns:
        AttentionCME or AttentionCMEWithDecisionToken module
    """
    dim = getattr(args, 'adapter_dim', 256)
    num_heads = getattr(args, 'cme_num_heads', 8)
    dropout = getattr(args, 'cme_dropout', 0.1)
    use_decision_token = getattr(args, 'cme_use_decision_token', False)

    if use_decision_token:
        return AttentionCMEWithDecisionToken(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout
        )
    else:
        return AttentionCME(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout
        )
