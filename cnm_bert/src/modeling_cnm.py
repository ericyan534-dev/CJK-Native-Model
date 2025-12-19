"""CNM-BERT model components."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import BertConfig, BertModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions, MaskedLMOutput
from transformers.models.bert.modeling_bert import (
    BertEmbeddings,
    BertEncoder,
    BertForMaskedLM,
    BertPreTrainedModel,
)


class TreeMLPEncoder(nn.Module):
    """Recursive Tree-MLP encoder for IDS trees with per-batch caching."""

    def __init__(self, tree_map: Dict[str, Dict], struct_index_to_char: List[str], struct_dim: int = 256):
        super().__init__()
        self.struct_dim = struct_dim
        self.struct_index_to_char = struct_index_to_char
        self.tree_map = tree_map
        component_vocab, operator_vocab = self._collect_vocab()
        self.component_vocab = component_vocab
        self.operator_vocab = operator_vocab
        self.component_index = {c: i for i, c in enumerate(component_vocab)}
        self.operator_index = {o: i for i, o in enumerate(operator_vocab)}
        self.component_embeddings = nn.Embedding(len(component_vocab), struct_dim)
        self.operator_embeddings = nn.Embedding(len(operator_vocab), struct_dim)
        self.binary_mlp = nn.Linear(struct_dim * 3, struct_dim)
        self.ternary_mlp = nn.Linear(struct_dim * 4, struct_dim)
        self.layer_norm = nn.LayerNorm(struct_dim)
        # Store flattened instructions as a buffer for checkpoint portability.
        buffer_bytes = json.dumps(self.tree_map, ensure_ascii=False).encode("utf-8")
        self.register_buffer("tree_buffer", torch.tensor(list(buffer_bytes), dtype=torch.uint8))

    def _collect_vocab(self) -> Tuple[List[str], List[str]]:
        components: Set[str] = set()
        operators: Set[str] = set()

        def visit(node: Dict):
            if "leaf" in node:
                components.add(node["leaf"])
                return
            operators.add(node.get("op", ""))
            for child in node.get("children", []):
                visit(child)

        for tree in self.tree_map.values():
            visit(tree)
        return sorted(components | {"[UNK_COMP]"}), sorted(operators | {"[UNK_OP]"})

    def forward(self, struct_indices: torch.LongTensor) -> torch.Tensor:
        # struct_indices: (batch, seq)
        flat = struct_indices.reshape(-1)
        unique, inverse = torch.unique(flat, sorted=False, return_inverse=True)
        embeddings: List[torch.Tensor] = []
        for sid in unique.tolist():
            embeddings.append(self._encode_single(sid))
        stacked = torch.stack(embeddings, dim=0)
        gathered = stacked[inverse].view(*struct_indices.shape, self.struct_dim)
        return gathered

    def _encode_single(self, struct_id: int) -> torch.Tensor:
        if struct_id <= 1:
            return torch.zeros(self.struct_dim, device=self.component_embeddings.weight.device)
        char = self.struct_index_to_char[struct_id]
        tree = self.tree_map.get(char)
        if tree is None:
            return torch.zeros(self.struct_dim, device=self.component_embeddings.weight.device)
        return self._encode_tree(tree)

    def _encode_tree(self, node: Dict) -> torch.Tensor:
        if "leaf" in node:
            comp_idx = self.component_index.get(node["leaf"], self.component_index["[UNK_COMP]"])
            return self.layer_norm(self.component_embeddings.weight[comp_idx])
        op = node.get("op", "[UNK_OP]")
        op_idx = self.operator_index.get(op, self.operator_index["[UNK_OP]"])
        children = node.get("children", [])
        child_states = [self._encode_tree(child) for child in children]
        if len(child_states) == 2:
            cat = torch.cat([self.operator_embeddings.weight[op_idx], child_states[0], child_states[1]], dim=-1)
            return self.layer_norm(self.binary_mlp(cat))
        if len(child_states) == 3:
            cat = torch.cat(
                [self.operator_embeddings.weight[op_idx], child_states[0], child_states[1], child_states[2]], dim=-1
            )
            return self.layer_norm(self.ternary_mlp(cat))
        # Fallback: average children if arity unexpected.
        if child_states:
            avg = torch.mean(torch.stack(child_states), dim=0)
            cat = torch.cat([self.operator_embeddings.weight[op_idx], avg, avg], dim=-1)
            return self.layer_norm(self.binary_mlp(cat))
        return torch.zeros(self.struct_dim, device=self.component_embeddings.weight.device)


class CNMBertEmbeddings(nn.Module):
    def __init__(self, config: BertConfig, tree_encoder: TreeMLPEncoder):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.tree_encoder = tree_encoder
        self.struct_dim = tree_encoder.struct_dim
        self.fusion = nn.Linear(config.hidden_size + self.struct_dim, config.hidden_size)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
        struct_indices: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        if input_ids is not None:
            input_shape = input_ids.size()
            device = input_ids.device
        else:
            input_shape = inputs_embeds.size()[:-1]
            device = inputs_embeds.device
        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : past_key_values_length + seq_length]
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        if struct_indices is None:
            struct_embed = torch.zeros_like(inputs_embeds)
        else:
            struct_embed = self.tree_encoder(struct_indices)
        concat = torch.cat([inputs_embeds, struct_embed], dim=-1)
        fused = self.LayerNorm(self.fusion(concat))
        position = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = fused + position + token_type_embeddings
        return self.dropout(embeddings)


class CNMBertModel(BertModel):
    def __init__(self, config: BertConfig, tree_encoder: TreeMLPEncoder):
        super().__init__(config)
        self.embeddings = CNMBertEmbeddings(config, tree_encoder)
        self.encoder = BertEncoder(config)
        self.pooler = None
        self.post_init()

    def forward(self, struct_indices: Optional[torch.LongTensor] = None, **kwargs):
        kwargs.pop("struct_ids", None)

        output_attentions = kwargs.get("output_attentions", None)
        output_hidden_states = kwargs.get("output_hidden_states", None)
        return_dict = kwargs.get("return_dict", None)
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_ids = kwargs.get("input_ids", None)
        attention_mask = kwargs.get("attention_mask", None)
        token_type_ids = kwargs.get("token_type_ids", None)
        position_ids = kwargs.get("position_ids", None)
        head_mask = kwargs.get("head_mask", None)
        inputs_embeds = kwargs.get("inputs_embeds", None)
        encoder_hidden_states = kwargs.get("encoder_hidden_states", None)
        encoder_attention_mask = kwargs.get("encoder_attention_mask", None)
        past_key_values = kwargs.get("past_key_values", None)
        use_cache = kwargs.get("use_cache", None)

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is not None:
            input_shape = input_ids.size()
            device = input_ids.device
        else:
            input_shape = inputs_embeds.size()[:-1]
            device = inputs_embeds.device

        batch_size, seq_length = input_shape
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        past_key_values_length = 0
        if past_key_values is not None:
            try:
                past_key_values_length = past_key_values[0][0].shape[2]
            except Exception:
                past_key_values_length = 0

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
            struct_indices=struct_indices,
        )

        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, device)

        encoder_extended_attention_mask = None
        if encoder_hidden_states is not None:
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_states.shape[:-1], device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = encoder_outputs[0]
        pooled_output = None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=getattr(encoder_outputs, "past_key_values", None),
            hidden_states=getattr(encoder_outputs, "hidden_states", None),
            attentions=getattr(encoder_outputs, "attentions", None),
            cross_attentions=getattr(encoder_outputs, "cross_attentions", None),
        )


class CNMForMaskedLM(BertForMaskedLM):
    def __init__(self, config: BertConfig, tree_encoder: Optional[TreeMLPEncoder] = None):
        super().__init__(config)
        self.tree_encoder = tree_encoder
        if self.tree_encoder is None:
            self.tree_encoder = TreeMLPEncoder({}, ["[NONE]", "[UNK_STRUCT]"], struct_dim=256)
        self.bert = CNMBertModel(config, self.tree_encoder)
        self.post_init()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        tree_path: Path,
        tokenizer_struct_vocab: List[str],
        struct_dim: int = 256,
        **kwargs,
    ):
        config = BertConfig.from_pretrained(pretrained_model_name_or_path)
        if tree_path.exists():
            with tree_path.open("r", encoding="utf-8") as f:
                tree_map = json.load(f)
        else:
            tree_map = {}
        tree_encoder = TreeMLPEncoder(
            tree_map=tree_map, struct_index_to_char=tokenizer_struct_vocab, struct_dim=struct_dim
        )
        model = super().from_pretrained(pretrained_model_name_or_path, config=config, tree_encoder=tree_encoder, **kwargs)
        return model

    def forward(self, struct_indices: Optional[torch.LongTensor] = None, **kwargs):
        kwargs.pop("struct_ids", None)

        labels = kwargs.pop("labels", None)
        return_dict = kwargs.get("return_dict", None)
        output_attentions = kwargs.get("output_attentions", None)
        output_hidden_states = kwargs.get("output_hidden_states", None)

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids=kwargs.get("input_ids", None),
            attention_mask=kwargs.get("attention_mask", None),
            token_type_ids=kwargs.get("token_type_ids", None),
            position_ids=kwargs.get("position_ids", None),
            head_mask=kwargs.get("head_mask", None),
            inputs_embeds=kwargs.get("inputs_embeds", None),
            encoder_hidden_states=kwargs.get("encoder_hidden_states", None),
            encoder_attention_mask=kwargs.get("encoder_attention_mask", None),
            past_key_values=kwargs.get("past_key_values", None),
            use_cache=kwargs.get("use_cache", None),
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            struct_indices=struct_indices,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=getattr(outputs, "hidden_states", None),
            attentions=getattr(outputs, "attentions", None),
        )


__all__ = ["TreeMLPEncoder", "CNMBertEmbeddings", "CNMBertModel", "CNMForMaskedLM"]
