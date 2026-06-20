from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.autograd import Function
import torch.nn.functional as F


RESIDUES = "ACDEFGHIKLMNPQRSTVWYX"
PTM_STATES = ["unmodified", "phospho", "acetyl", "methyl", "sumoyl", "ubiquityl", "glycosyl", "other"]
NO_STRUCTURE_ID = 20


@dataclass(frozen=True)
class PTMCipherConfig:
    residue_vocab: int = 21
    structure_vocab: int = 21
    ptm_vocab: int = 8
    dim: int = 192
    heads: int = 6
    layers: int = 4
    ff_dim: int = 512
    dropout: float = 0.15
    graph_layers: int = 3
    classes: int = 2
    head_input: str = "delta"
    adversary_dims: dict[str, int] | None = None


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.alpha * grad_output, None


class GradientReversal(nn.Module):
    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.alpha)


class FactoredProteoformEmbedding(nn.Module):
    """Residue x 3Di x PTM-state embedding without a sparse 3,528-row table."""

    def __init__(self, config: PTMCipherConfig) -> None:
        super().__init__()
        dim = config.dim
        self.residue = nn.Embedding(config.residue_vocab, dim)
        self.structure = nn.Embedding(config.structure_vocab, dim)
        self.ptm = nn.Embedding(config.ptm_vocab, dim)
        self.residue_ptm = nn.Embedding(config.residue_vocab * config.ptm_vocab, dim)
        self.residue_structure = nn.Embedding(config.residue_vocab * config.structure_vocab, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, residue_ids: torch.Tensor, structure_ids: torch.Tensor, ptm_state_ids: torch.Tensor) -> torch.Tensor:
        residue_ptm_id = residue_ids * self.ptm.num_embeddings + ptm_state_ids
        residue_structure_id = residue_ids * self.structure.num_embeddings + structure_ids
        x = (
            self.residue(residue_ids)
            + self.structure(structure_ids)
            + self.ptm(ptm_state_ids)
            + self.residue_ptm(residue_ptm_id)
            + self.residue_structure(residue_structure_id)
        )
        return self.norm(x)


class DenseContactPropagation(nn.Module):
    """Small dense message-passing block for cropped residue-contact adjacencies."""

    def __init__(self, config: PTMCipherConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(config.dim * 2, config.dim),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                    nn.LayerNorm(config.dim),
                )
                for _ in range(config.graph_layers)
            ]
        )

    def forward(self, hidden: torch.Tensor, adjacency: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # adjacency: [B, L, L]. Normalize rows, preserve self through residuals.
        adjacency = adjacency.float() * mask[:, :, None].float() * mask[:, None, :].float()
        degree = adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)
        norm_adj = adjacency / degree
        x = hidden
        for layer in self.layers:
            msg = torch.bmm(norm_adj, x)
            x = x + layer(torch.cat([x, msg], dim=-1))
            x = x * mask[:, :, None].float()
        return x


class InterfaceCrossAttention(nn.Module):
    def __init__(self, config: PTMCipherConfig) -> None:
        super().__init__()
        self.cross = nn.MultiheadAttention(config.dim, config.heads, dropout=config.dropout, batch_first=True)
        self.contact_gate = nn.Parameter(torch.tensor(0.2))
        self.norm = nn.LayerNorm(config.dim)

    def forward(
        self,
        query: torch.Tensor,
        partner: torch.Tensor,
        query_mask: torch.Tensor,
        partner_mask: torch.Tensor,
        contact_mask: torch.Tensor,
    ) -> torch.Tensor:
        key_padding_mask = ~partner_mask.bool()
        attn, _ = self.cross(query, partner, partner, key_padding_mask=key_padding_mask, need_weights=False)
        contact = contact_mask.float() * query_mask[:, :, None].float() * partner_mask[:, None, :].float()
        contact_degree = contact.sum(dim=-1, keepdim=True).clamp_min(1.0)
        contact_context = torch.bmm(contact / contact_degree, partner)
        return self.norm(query + attn + self.contact_gate * contact_context)


class PTMCipher(nn.Module):
    def __init__(self, config: PTMCipherConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = FactoredProteoformEmbedding(config)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.dim,
            nhead=config.heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.layers)
        self.cross_attention = InterfaceCrossAttention(config)
        self.graph = DenseContactPropagation(config)
        self.delta_norm = nn.LayerNorm(config.dim)
        self.classifier = nn.Sequential(
            nn.Linear(config.dim, config.dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim, config.classes),
        )
        self.evidence = nn.Sequential(nn.Linear(config.dim, config.classes), nn.Softplus())
        self.grl = GradientReversal(alpha=1.0)
        self.adversaries = nn.ModuleDict()
        for name, width in (config.adversary_dims or {}).items():
            self.adversaries[name] = nn.Sequential(
                nn.Linear(config.dim, config.dim // 2),
                nn.GELU(),
                nn.Linear(config.dim // 2, int(width)),
            )

    def _encode(
        self,
        residue_ids: torch.Tensor,
        structure_ids: torch.Tensor,
        ptm_state_ids: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.embedding(residue_ids, structure_ids, ptm_state_ids)
        encoded = self.encoder(x, src_key_padding_mask=~mask.bool())
        return encoded * mask[:, :, None].float()

    @staticmethod
    def _inject_ptm_delta(hidden: torch.Tensor, delta: torch.Tensor, ptm_index: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        bsz, length, _ = hidden.shape
        one_hot = F.one_hot(ptm_index.clamp(0, length - 1), num_classes=length).float().to(hidden.device)
        injected = hidden + one_hot[:, :, None] * delta[:, None, :]
        return injected * mask[:, :, None].float()

    @staticmethod
    def _interface_pool(hidden: torch.Tensor, contact_mask: torch.Tensor, sequence_mask: torch.Tensor) -> torch.Tensor:
        interface_mask = contact_mask.any(dim=-1) & sequence_mask.bool()
        fallback = sequence_mask.bool()
        pool_mask = torch.where(interface_mask.any(dim=1, keepdim=True), interface_mask, fallback)
        weights = pool_mask.float() / pool_mask.float().sum(dim=1, keepdim=True).clamp_min(1.0)
        return torch.bmm(weights[:, None, :], hidden).squeeze(1)

    def forward(
        self,
        mod_residue_ids: torch.Tensor,
        mod_structure_ids: torch.Tensor,
        mod_ptm_state_ids: torch.Tensor,
        unmod_ptm_state_ids: torch.Tensor,
        mod_mask: torch.Tensor,
        partner_residue_ids: torch.Tensor,
        partner_structure_ids: torch.Tensor,
        partner_mask: torch.Tensor,
        ptm_index: torch.Tensor,
        contact_mask: torch.Tensor,
        residue_adjacency: torch.Tensor,
        adversary_alpha: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        partner_ptm = torch.zeros_like(partner_residue_ids)
        h_unmod = self._encode(mod_residue_ids, mod_structure_ids, unmod_ptm_state_ids, mod_mask)
        h_mod = self._encode(mod_residue_ids, mod_structure_ids, mod_ptm_state_ids, mod_mask)
        h_partner = self._encode(partner_residue_ids, partner_structure_ids, partner_ptm, partner_mask)

        z_unmod = self.cross_attention(h_unmod, h_partner, mod_mask, partner_mask, contact_mask)
        z_mod = self.cross_attention(h_mod, h_partner, mod_mask, partner_mask, contact_mask)

        ptm_delta = h_mod[torch.arange(h_mod.shape[0], device=h_mod.device), ptm_index] - h_unmod[
            torch.arange(h_unmod.shape[0], device=h_unmod.device), ptm_index
        ]
        g_mod_input = self._inject_ptm_delta(z_mod, ptm_delta, ptm_index, mod_mask)
        g_mod = self.graph(g_mod_input, residue_adjacency, mod_mask)
        g_unmod = self.graph(z_unmod, residue_adjacency, mod_mask)

        i_mod = self._interface_pool(z_mod + g_mod, contact_mask, mod_mask)
        i_unmod = self._interface_pool(z_unmod + g_unmod, contact_mask, mod_mask)
        delta = self.delta_norm(i_mod - i_unmod)
        if self.config.head_input == "modified":
            head_vec = self.delta_norm(i_mod)
        elif self.config.head_input == "unmodified":
            head_vec = self.delta_norm(i_unmod)
        else:
            head_vec = delta
        logits = self.classifier(head_vec)
        evidence_alpha = self.evidence(head_vec) + 1.0

        self.grl.alpha = adversary_alpha
        adversary_logits = {name: head(self.grl(i_mod)) for name, head in self.adversaries.items()}
        return {
            "logits": logits,
            "evidence_alpha": evidence_alpha,
            "delta": delta,
            "i_mod": i_mod,
            "i_unmod": i_unmod,
            "adversary_logits": adversary_logits,
        }


def ptm_cipher_loss(
    outputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    adversary_targets: dict[str, torch.Tensor] | None = None,
    lambda_brier: float = 0.1,
    lambda_adversary: float = 0.2,
) -> dict[str, torch.Tensor]:
    logits = outputs["logits"]
    probs = logits.softmax(dim=-1)
    cls = F.cross_entropy(logits, labels)
    one_hot = F.one_hot(labels, num_classes=logits.shape[-1]).float()
    brier = ((probs - one_hot) ** 2).sum(dim=-1).mean()
    adv_loss = logits.new_tensor(0.0)
    if adversary_targets:
        for name, target in adversary_targets.items():
            if name in outputs["adversary_logits"]:
                adv_loss = adv_loss + F.cross_entropy(outputs["adversary_logits"][name], target)
    total = cls + lambda_brier * brier + lambda_adversary * adv_loss
    return {"loss": total, "classification": cls, "brier": brier, "adversary": adv_loss}
