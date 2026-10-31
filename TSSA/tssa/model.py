from __future__ import annotations
import math
import re
from pathlib import Path
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

def load_swin_checkpoint(model: nn.Module, checkpoint_path: str | Path) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    source = checkpoint.get('model', checkpoint)
    target = model.state_dict()
    mapped = {}
    for key, value in source.items():
        if key.startswith('head.') or 'relative_position_index' in key or 'attn_mask' in key:
            continue
        downsample_match = re.match('layers\\.(\\d+)\\.downsample\\.(.+)', key)
        if downsample_match:
            stage = int(downsample_match.group(1)) + 1
            key = f'layers.{stage}.downsample.{downsample_match.group(2)}'
        if key in target and target[key].shape == value.shape:
            mapped[key] = value
    message = model.load_state_dict(mapped, strict=False)
    if len(mapped) < 160:
        raise RuntimeError(f'Only {len(mapped)} Swin tensors matched; expected at least 160. Check the checkpoint and timm model version.')
    return {'matched': len(mapped), 'missing': list(message.missing_keys), 'unexpected': list(message.unexpected_keys)}

class SwinHierarchicalEncoder(nn.Module):

    def __init__(self, checkpoint_path: str | Path):
        super().__init__()
        self.backbone = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=0)
        self.stage3_width = 384
        self.stage4_width = int(self.backbone.num_features)
        self.load_report = load_swin_checkpoint(self.backbone, checkpoint_path)

    def forward(self, images: torch.Tensor):
        feature_maps = self.backbone.forward_intermediates(images, indices=(2, 3), norm=True, output_fmt='NCHW', intermediates_only=True)
        if len(feature_maps) != 2:
            raise RuntimeError('Swin-T did not return the requested Stage-3/4 features')
        fine, coarse = feature_maps
        if fine.shape[1:] != (self.stage3_width, 14, 14):
            raise RuntimeError(f'Unexpected Swin Stage-3 shape: {tuple(fine.shape)}')
        if coarse.shape[1:] != (self.stage4_width, 7, 7):
            raise RuntimeError(f'Unexpected Swin Stage-4 shape: {tuple(coarse.shape)}')
        return (fine, coarse)

class FeedForward(nn.Module):

    def __init__(self, embed_dim: int, dropout: float, expansion: int=2):
        super().__init__()
        hidden_dim = embed_dim * int(expansion)
        self.network = nn.Sequential(nn.Linear(embed_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, embed_dim), nn.Dropout(dropout))

    def forward(self, inputs: torch.Tensor):
        return self.network(inputs)

class ResidualFeatureProjection(nn.Linear):

    def __init__(self, affine: nn.Linear, dropout: float, expansion: int):
        nn.Module.__init__(self)
        self.in_features, self.out_features = (affine.in_features, affine.out_features)
        self.weight, self.bias = (affine.weight, affine.bias)
        self.adapter_norm = nn.LayerNorm(self.out_features)
        self.adapter = FeedForward(self.out_features, dropout, expansion)
        nn.init.zeros_(self.adapter.network[3].weight)
        nn.init.zeros_(self.adapter.network[3].bias)
        self.adapter_norm.requires_grad_(affine.weight.requires_grad)
        self.adapter.requires_grad_(affine.weight.requires_grad)

    def forward(self, inputs: torch.Tensor):
        projected = F.linear(inputs, self.weight, self.bias)
        return projected + self.adapter(self.adapter_norm(projected))

class AdaptiveFeatureFusion(nn.Module):

    def __init__(self, embed_dim: int, initial_gate: float=0.5):
        super().__init__()
        initial_gate = float(initial_gate)
        if not 0.0 < initial_gate < 1.0:
            raise ValueError('fusion initial_gate must be in (0, 1)')
        self.context_norm = nn.LayerNorm(embed_dim * 3)
        self.gate = nn.Linear(embed_dim * 3, embed_dim)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, math.log(initial_gate / (1.0 - initial_gate)))
        self.nonlinear_gate = None

    def enable_nonlinear_gate(self, dropout: float):
        if self.nonlinear_gate is not None:
            raise RuntimeError('Nonlinear fusion gate is already installed')
        width = self.gate.out_features
        self.nonlinear_gate = nn.Sequential(nn.Linear(self.gate.in_features, width), nn.GELU(), nn.Dropout(dropout), nn.Linear(width, width))
        nn.init.zeros_(self.nonlinear_gate[3].weight)
        nn.init.zeros_(self.nonlinear_gate[3].bias)

    def forward(self, anchor: torch.Tensor, evidence: torch.Tensor):
        anchor = F.normalize(anchor, dim=-1)
        evidence = F.normalize(evidence, dim=-1)
        context = self.context_norm(torch.cat([anchor, evidence, anchor * evidence], dim=-1))
        gate_logits = self.gate(context)
        if self.nonlinear_gate is not None:
            gate_logits = gate_logits + self.nonlinear_gate(context)
        gate = torch.sigmoid(gate_logits)
        fused = (1.0 - gate) * anchor + gate * evidence
        return (F.normalize(fused, dim=-1), gate)

def _map_to_tokens(feature_map: torch.Tensor) -> torch.Tensor:
    return feature_map.flatten(2).transpose(1, 2)

class SparseRelationPropagation(nn.Module):

    def __init__(self, embed_dim: int, num_heads: int, dropout: float):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError('SpA embed_dim must be divisible by num_heads')
        self.num_heads = int(num_heads)
        self.head_dim = embed_dim // self.num_heads
        self.spatial_query = nn.Linear(embed_dim, embed_dim)
        self.spatial_key = nn.Linear(embed_dim, embed_dim)
        self.spatial_value = nn.Linear(embed_dim, embed_dim)
        self.spatial_output = nn.Linear(embed_dim, embed_dim)
        self.semantic_query = nn.Linear(embed_dim, embed_dim)
        self.semantic_key = nn.Linear(embed_dim, embed_dim)
        self.semantic_value = nn.Linear(embed_dim, embed_dim)
        self.semantic_output = nn.Linear(embed_dim, embed_dim)
        self.spatial_norm = nn.LayerNorm(embed_dim)
        self.semantic_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _eight_neighbors(values: torch.Tensor, height: int, width: int):
        batch_size, token_count, embed_dim = values.shape
        value_map = values.transpose(1, 2).reshape(batch_size, embed_dim, height, width)
        unfolded = F.unfold(value_map, kernel_size=3, padding=1)
        unfolded = unfolded.reshape(batch_size, embed_dim, 9, token_count)
        unfolded = unfolded.permute(0, 3, 2, 1)
        neighbor_indices = (0, 1, 2, 3, 5, 6, 7, 8)
        return unfolded[:, :, neighbor_indices]

    def spatial_message(self, tokens: torch.Tensor, height: int, width: int):
        batch_size, token_count, embed_dim = tokens.shape
        if token_count != height * width:
            raise ValueError('Spatial token count does not match the requested grid')
        queries = self.spatial_query(tokens).reshape(batch_size, token_count, self.num_heads, self.head_dim)
        keys = self._eight_neighbors(self.spatial_key(tokens), height, width).reshape(batch_size, token_count, 8, self.num_heads, self.head_dim)
        values = self._eight_neighbors(self.spatial_value(tokens), height, width).reshape(batch_size, token_count, 8, self.num_heads, self.head_dim)
        scores = torch.einsum('bnhd,bnkhd->bnhk', queries, keys)
        scores = scores / math.sqrt(self.head_dim)
        validity = F.unfold(tokens.new_ones(1, 1, height, width), kernel_size=3, padding=1).transpose(1, 2)
        validity = validity[:, :, (0, 1, 2, 3, 5, 6, 7, 8)].bool()
        scores = scores.masked_fill(~validity.unsqueeze(2), -10000.0)
        weights = F.softmax(scores, dim=-1)
        message = torch.einsum('bnhk,bnkhd->bnhd', weights, values)
        message = message.reshape(batch_size, token_count, embed_dim)
        message = self.spatial_output(message)
        return self.dropout(self.spatial_norm(message))

    def semantic_message(self, tokens: torch.Tensor, semantic_k: int, context_bias: torch.Tensor | None=None):
        batch_size, token_count, embed_dim = tokens.shape
        token_count = tokens.shape[1]
        if token_count < 2:
            raise ValueError('Semantic propagation requires at least two nodes')
        k = min(int(semantic_k), token_count - 1)
        queries = self.semantic_query(tokens).reshape(batch_size, token_count, self.num_heads, self.head_dim)
        keys = self.semantic_key(tokens).reshape(batch_size, token_count, self.num_heads, self.head_dim)
        values = self.semantic_value(tokens).reshape(batch_size, token_count, self.num_heads, self.head_dim)
        if context_bias is not None:
            if context_bias.shape != (batch_size, token_count):
                raise ValueError('context_bias must have shape [batch, tokens]')
            with torch.autocast(device_type=tokens.device.type, enabled=False):
                similarities = torch.einsum('bnhd,bmhd->bhnm', queries.float(), keys.float()) / math.sqrt(self.head_dim)
                similarities = similarities + context_bias.float()[:, None, None, :]
        else:
            similarities = torch.einsum('bnhd,bmhd->bhnm', queries, keys)
            similarities = similarities / math.sqrt(self.head_dim)
        diagonal = torch.eye(token_count, dtype=torch.bool, device=tokens.device)
        similarities = similarities.masked_fill(diagonal[None, None], -float('inf') if context_bias is not None else -10000.0)
        scores, indices = similarities.topk(k, dim=-1)
        values = values.permute(0, 2, 1, 3)
        batch_indices = torch.arange(batch_size, device=tokens.device)[:, None, None, None]
        head_indices = torch.arange(self.num_heads, device=tokens.device)[None, :, None, None]
        neighbors = values[batch_indices, head_indices, indices]
        weights = F.softmax(scores, dim=-1)
        message = (weights.unsqueeze(-1) * neighbors).sum(dim=3)
        message = message.permute(0, 2, 1, 3).reshape(batch_size, token_count, embed_dim)
        message = self.semantic_output(message)
        return self.dropout(self.semantic_norm(message))

    def forward(self, tokens: torch.Tensor, height: int, width: int, semantic_k: int, context_bias: torch.Tensor | None=None):
        return (self.spatial_message(tokens, height, width), self.semantic_message(tokens, semantic_k, context_bias))

class HierarchicalSpatialRelationPropagation(nn.Module):

    def __init__(self, embed_dim: int, semantic_k: int, num_heads: int, dropout: float, ffn_expansion: int=2, context_guided: bool=True, context_eta_initial: float=1.0):
        super().__init__()
        self.semantic_k = int(semantic_k)
        self.fine_relations = SparseRelationPropagation(embed_dim, num_heads, dropout)
        self.coarse_relations = SparseRelationPropagation(embed_dim, num_heads, dropout)
        self.bottom_up = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.top_down = nn.Linear(embed_dim, embed_dim)
        self.fine_same_norm = nn.LayerNorm(embed_dim)
        self.coarse_same_norm = nn.LayerNorm(embed_dim)
        self.bottom_up_norm = nn.LayerNorm(embed_dim)
        self.top_down_norm = nn.LayerNorm(embed_dim)
        self.fine_ffn_norm = nn.LayerNorm(embed_dim)
        self.coarse_ffn_norm = nn.LayerNorm(embed_dim)
        self.fine_ffn = FeedForward(embed_dim, dropout, ffn_expansion)
        self.coarse_ffn = FeedForward(embed_dim, dropout, ffn_expansion)
        self.pool_query = nn.Linear(embed_dim, embed_dim)
        self.fine_pool_key = nn.Linear(embed_dim, embed_dim)
        self.coarse_pool_key = nn.Linear(embed_dim, embed_dim)
        self.output_norm = nn.LayerNorm(embed_dim)
        self.relation_logits = nn.Parameter(torch.zeros(3))
        self.scale_logits = nn.Parameter(torch.zeros(2))
        self.dropout = nn.Dropout(dropout)
        self.context_guided = bool(context_guided)
        if self.context_guided:
            if not math.isfinite(context_eta_initial) or context_eta_initial <= 0:
                raise ValueError('context_eta_initial must be finite and positive')
            with torch.random.fork_rng(devices=[]):
                self.context_projection = nn.Linear(embed_dim, embed_dim)
                self.fine_relevance = nn.Linear(embed_dim, embed_dim)
                self.coarse_relevance = nn.Linear(embed_dim, embed_dim)
            inverse_softplus = context_eta_initial + math.log(-math.expm1(-context_eta_initial))
            self.context_theta = nn.Parameter(torch.tensor(inverse_softplus))

    def semantic_context_biases(self, fine: torch.Tensor, coarse: torch.Tensor):
        context = self.context_projection(coarse.mean(dim=1))
        fine_relevance = self.fine_relevance(fine)
        coarse_relevance = self.coarse_relevance(coarse)
        with torch.autocast(device_type=fine.device.type, enabled=False):
            eta = F.softplus(self.context_theta.float())
            fine_bias = eta * torch.einsum('bnd,bd->bn', fine_relevance.float(), context.float())
            coarse_bias = eta * torch.einsum('bnd,bd->bn', coarse_relevance.float(), context.float())
        return (fine_bias / math.sqrt(fine.shape[-1]), coarse_bias / math.sqrt(coarse.shape[-1]))

    def forward(self, fine: torch.Tensor, coarse: torch.Tensor):
        if fine.shape[1] != 14 * 14 or coarse.shape[1] != 7 * 7:
            raise ValueError('SpA expects native Swin-T Stage-3/4 token grids')
        relation_weights = F.softmax(self.relation_logits, dim=0)
        scale_weights = F.softmax(self.scale_logits, dim=0)
        fine_bias, coarse_bias = (None, None)
        if self.context_guided:
            fine_bias, coarse_bias = self.semantic_context_biases(fine, coarse)
        fine_spatial, fine_semantic = self.fine_relations(fine, 14, 14, self.semantic_k, fine_bias)
        coarse_spatial, coarse_semantic = self.coarse_relations(coarse, 7, 7, self.semantic_k, coarse_bias)
        fine = self.fine_same_norm(fine + relation_weights[0] * fine_spatial + relation_weights[1] * fine_semantic)
        coarse = self.coarse_same_norm(coarse + relation_weights[0] * coarse_spatial + relation_weights[1] * coarse_semantic)
        batch_size, _, embed_dim = fine.shape
        fine_grid = fine.reshape(batch_size, 14, 14, embed_dim)
        children = fine_grid.reshape(batch_size, 7, 2, 7, 2, embed_dim)
        children = children.permute(0, 1, 3, 2, 4, 5).reshape(batch_size * 49, 4, embed_dim)
        coarse_queries = coarse.reshape(batch_size * 49, 1, embed_dim)
        child_message, _ = self.bottom_up(coarse_queries, children, children, need_weights=False)
        child_message = child_message.reshape(batch_size, 49, embed_dim)
        coarse = self.bottom_up_norm(coarse + relation_weights[2] * self.dropout(child_message))
        parent_grid = coarse.transpose(1, 2).reshape(batch_size, embed_dim, 7, 7)
        parent_for_child = parent_grid.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)
        parent_for_child = parent_for_child.flatten(2).transpose(1, 2)
        fine = self.top_down_norm(fine + relation_weights[2] * self.dropout(self.top_down(parent_for_child)))
        fine = fine + self.fine_ffn(self.fine_ffn_norm(fine))
        coarse = coarse + self.coarse_ffn(self.coarse_ffn_norm(coarse))
        pool_query = self.pool_query(coarse.mean(dim=1))
        fine_pool_logits = torch.einsum('bd,bnd->bn', pool_query, self.fine_pool_key(fine)) / math.sqrt(embed_dim)
        coarse_pool_logits = torch.einsum('bd,bnd->bn', pool_query, self.coarse_pool_key(coarse)) / math.sqrt(embed_dim)
        fine_pool_weights = F.softmax(fine_pool_logits, dim=1)
        coarse_pool_weights = F.softmax(coarse_pool_logits, dim=1)
        fine_pooled = torch.einsum('bn,bnd->bd', fine_pool_weights, fine)
        coarse_pooled = torch.einsum('bn,bnd->bd', coarse_pool_weights, coarse)
        structural = scale_weights[0] * fine_pooled + scale_weights[1] * coarse_pooled
        structural = F.normalize(self.output_norm(structural), dim=-1)
        relation_entropy = -(relation_weights * relation_weights.clamp_min(1e-08).log()).sum()
        fine_pool_entropy = -(fine_pool_weights * fine_pool_weights.clamp_min(1e-08).log()).sum(dim=1)
        coarse_pool_entropy = -(coarse_pool_weights * coarse_pool_weights.clamp_min(1e-08).log()).sum(dim=1)
        return (structural, {'hsrp_relation_weights': relation_weights, 'hsrp_scale_weights': scale_weights, 'hsrp_relation_entropy': relation_entropy, 'hsrp_pool_entropy': scale_weights[0] * fine_pool_entropy + scale_weights[1] * coarse_pool_entropy})

class CyclicWordReactivationCell(nn.Module):

    def __init__(self, embed_dim: int, num_heads: int, dropout: float, ffn_expansion: int=2):
        super().__init__()
        self.state_norm = nn.LayerNorm(embed_dim)
        self.evidence_norm = nn.LayerNorm(embed_dim)
        self.reactivation = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.gate_norm = nn.LayerNorm(embed_dim * 2)
        self.update_gate = nn.Linear(embed_dim * 2, embed_dim)
        self.update_norm = nn.LayerNorm(embed_dim)
        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, dropout, ffn_expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(self, state: torch.Tensor, word_evidence: torch.Tensor, sequence_mask: torch.Tensor):
        state_normalized = self.state_norm(state)
        evidence_normalized = self.evidence_norm(word_evidence)
        message, _ = self.reactivation(query=state_normalized, key=evidence_normalized, value=evidence_normalized, key_padding_mask=~sequence_mask, need_weights=False)
        gate = torch.sigmoid(self.update_gate(self.gate_norm(torch.cat([state_normalized, message], dim=-1))))
        state = self.update_norm(state + gate * self.dropout(message))
        state = state + self.ffn(self.ffn_norm(state))
        state = state.masked_fill(~sequence_mask.unsqueeze(-1), 0.0)
        return (state, gate)

class TemporalAwareCyclicEvolution(nn.Module):

    def __init__(self, embed_dim: int, time_span: int, num_heads: int, dropout: float, ffn_expansion: int=2):
        super().__init__()
        self.time_span = int(time_span)
        if self.time_span < 1:
            raise ValueError('TeA time_span must contain at least one cycle')
        self.cycle = CyclicWordReactivationCell(embed_dim, num_heads, dropout, ffn_expansion)
        self.anchor_query = nn.Linear(embed_dim, embed_dim)
        self.token_key = nn.Linear(embed_dim, embed_dim)
        self.token_norm = nn.LayerNorm(embed_dim)
        self.evidence_norm = nn.LayerNorm(embed_dim)

    @staticmethod
    def _masked_score_std(scores: torch.Tensor, mask: torch.Tensor):
        valid = mask.to(scores.dtype)
        count = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean = (scores * valid).sum(dim=1, keepdim=True) / count
        variance = ((scores - mean).square() * valid).sum(dim=1) / count.squeeze(1)
        return torch.sqrt(variance + 1e-08)

    def forward(self, word_features, content_mask):
        anchor = word_features[:, 0]
        mask = content_mask[:, 1:]
        evidence_tokens = word_features[:, 1:].masked_fill(~mask.unsqueeze(-1), 0.0)
        if not mask.any(1).all():
            raise ValueError('Each description needs a content token')
        state = evidence_tokens
        for _ in range(self.time_span):
            state, gate = self.cycle(state, evidence_tokens, mask)
        refined = self.token_norm(state)
        scores = torch.einsum('bd,bnd->bn', self.anchor_query(anchor), self.token_key(refined)) / math.sqrt(anchor.shape[-1])
        weights = F.softmax(scores.masked_fill(~mask, -10000.0), dim=1)
        evidence = self.evidence_norm(torch.einsum('bn,bnd->bd', weights, refined))
        return (anchor, evidence, {})


class TSSA(nn.Module):
    """Spatial and temporal evidence encoders with adaptive semantic fusion."""
    def __init__(self, config):
        super().__init__()
        m = config['model']
        d, heads, dropout = m['embed_dim'], m['num_heads'], m['dropout']
        expansion = m['ffn_expansion']
        self.vision_encoder = SwinHierarchicalEncoder(m['swin_checkpoint'])
        self.text_encoder = AutoModel.from_pretrained(m['bert_model'])
        if m['bert_gradient_checkpointing']:
            self.text_encoder.gradient_checkpointing_enable()
        self.fine_projection = nn.Linear(384, d)
        self.coarse_projection = nn.Linear(768, d)
        self.fine_projection_norm = nn.LayerNorm(384)
        self.hsrp = HierarchicalSpatialRelationPropagation(d, m['semantic_k'], heads, dropout,
                    expansion, context_guided=True, context_eta_initial=m['context_eta_initial'])
        self.hsrp_fusion = AdaptiveFeatureFusion(d, m['fusion_gate_initial'])
        self.text_projection = nn.Linear(self.text_encoder.config.hidden_size, d)
        self.tea = TemporalAwareCyclicEvolution(d, m['refinement_steps'], heads, dropout, expansion)
        self.tea_fusion = AdaptiveFeatureFusion(d, m['fusion_gate_initial'])
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / config['loss']['temperature_initial'])))
        self.fine_projection = ResidualFeatureProjection(self.fine_projection, dropout, expansion)
        self.coarse_projection = ResidualFeatureProjection(self.coarse_projection, dropout, expansion)
        self.text_projection = ResidualFeatureProjection(self.text_projection, dropout, expansion)
        self.hsrp_fusion.enable_nonlinear_gate(dropout)
        self.tea_fusion.enable_nonlinear_gate(dropout)
        nn.init.zeros_(self.hsrp.context_projection.weight)
        nn.init.zeros_(self.hsrp.context_projection.bias)

    def encode_image(self, images, return_gates=False):
        fine, coarse = self.vision_encoder(images)
        coarse = self.coarse_projection(_map_to_tokens(coarse))
        fine = self.fine_projection(self.fine_projection_norm(_map_to_tokens(fine)))
        evidence, _ = self.hsrp(fine, coarse)
        embedding, gate = self.hsrp_fusion(coarse.mean(1), evidence)
        return (embedding, gate) if return_gates else embedding

    @staticmethod
    def content_mask(attention_mask):
        mask = attention_mask.bool().clone()
        mask[:, 0] = False
        mask.scatter_(1, attention_mask.sum(1).sub(1).clamp_min(0)[:, None], False)
        if not mask.any(1).all():
            raise ValueError('Every description must contain at least one content token')
        return mask

    def encode_text(self, input_ids, attention_mask, return_gates=False):
        hidden = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask,
                                   return_dict=True).last_hidden_state
        tokens = self.text_projection(hidden)
        anchor, evidence, _ = self.tea(tokens, self.content_mask(attention_mask))
        embedding, gate = self.tea_fusion(anchor, evidence)
        return (embedding, gate) if return_gates else embedding

    def forward(self, images, input_ids, attention_mask):
        return self.encode_image(images), self.encode_text(input_ids, attention_mask)

    def contrastive_scale(self):
        return self.logit_scale.exp().clamp(max=100.0)
