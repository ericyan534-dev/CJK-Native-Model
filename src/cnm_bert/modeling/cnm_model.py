"""CNM-BERT model classes compatible with HuggingFace Transformers."""

import json
import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from pathlib import Path

from transformers import BertPreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions, MaskedLMOutput
from transformers.models.bert.modeling_bert import BertEncoder, BertPooler, BertOnlyMLMHead

from .configuration_cnm import CNMConfig
from .tree_encoder import TreeMLPEncoder
from .cnm_embeddings import CNMEmbeddings


class CNMModel(BertPreTrainedModel):
    """CNM-BERT base model (without any task-specific head).

    This model extends BERT by integrating structural information from IDS trees.
    It can be used as a drop-in replacement for BertModel.

    Args:
        config: CNM configuration
        add_pooling_layer: Whether to add pooling layer (default: True)
    """

    config_class = CNMConfig
    base_model_prefix = "cnm"

    def __init__(self, config: CNMConfig, add_pooling_layer: bool = True):
        super().__init__(config)
        self.config = config

        # Load IDS tree map
        if config.tree_path:
            tree_path = Path(config.tree_path)
            if tree_path.exists():
                with open(tree_path, "r", encoding="utf-8") as f:
                    tree_map = json.load(f)
            else:
                raise ValueError(f"Tree path not found: {config.tree_path}")
        else:
            # Empty tree map (will need to be loaded later)
            tree_map = {}

        # Build struct_index_to_char from tree_map
        # Index 0: [NONE], Index 1: [UNK_STRUCT], Index 2+: characters
        struct_vocab = ["[NONE]", "[UNK_STRUCT]"] + sorted(tree_map.keys())
        self.struct_index_to_char = struct_vocab

        # Initialize tree encoder
        self.tree_encoder = TreeMLPEncoder(
            tree_map=tree_map,
            struct_index_to_char=struct_vocab,
            struct_dim=config.struct_dim
        )

        # CNM embeddings (combines BERT + structural)
        self.embeddings = CNMEmbeddings(config, self.tree_encoder)

        # BERT encoder (12 or 24 layers)
        self.encoder = BertEncoder(config)

        # Optional pooler
        self.pooler = BertPooler(config) if add_pooling_layer else None

        # Initialize weights
        self.post_init()

        # Optionally freeze BERT encoder
        if config.freeze_bert_encoder:
            self._freeze_bert_encoder()

    def _freeze_bert_encoder(self):
        """Freeze BERT encoder parameters (for curriculum learning)."""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_bert_encoder(self):
        """Unfreeze BERT encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = True

    def get_input_embeddings(self):
        """Get input embeddings (for HF compatibility)."""
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        """Set input embeddings (for HF compatibility)."""
        self.embeddings.word_embeddings = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        struct_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        """Forward pass of CNM-BERT model.

        Args:
            input_ids: Token IDs (batch_size, seq_len)
            struct_ids: Structural indices (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            token_type_ids: Segment IDs (batch_size, seq_len)
            position_ids: Position IDs (batch_size, seq_len)
            head_mask: Head mask for attention
            inputs_embeds: Optional pre-computed embeddings
            encoder_hidden_states: For cross-attention (not used)
            encoder_attention_mask: For cross-attention (not used)
            past_key_values: For generation (not used in pre-training)
            use_cache: Whether to use cache (not used in pre-training)
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output hidden states
            return_dict: Whether to return dict (default: True)

        Returns:
            BaseModelOutputWithPoolingAndCrossAttentions or tuple of tensors
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # Prepare attention mask
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # Get embeddings (BERT + structural fusion)
        if inputs_embeds is None:
            embedding_output = self.embeddings(
                input_ids=input_ids,
                struct_ids=struct_ids,
                position_ids=position_ids,
                token_type_ids=token_type_ids,
            )
        else:
            embedding_output = inputs_embeds

        # Pass through BERT encoder
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=None,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class CNMForMaskedLM(BertPreTrainedModel):
    """CNM-BERT model with masked language modeling head.

    This is the main model used for pre-training with MLM objective.

    Args:
        config: CNM configuration
    """

    config_class = CNMConfig
    base_model_prefix = "cnm"

    def __init__(self, config: CNMConfig):
        super().__init__(config)
        self.config = config

        # Base model
        self.cnm = CNMModel(config, add_pooling_layer=False)

        # MLM head (same as BERT)
        self.cls = BertOnlyMLMHead(config)

        # Initialize weights
        self.post_init()

    def get_output_embeddings(self):
        """Get output embeddings (for HF compatibility)."""
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        """Set output embeddings (for HF compatibility)."""
        self.cls.predictions.decoder = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        struct_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        """Forward pass with MLM loss computation.

        Args:
            input_ids: Token IDs (batch_size, seq_len)
            struct_ids: Structural indices (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            token_type_ids: Segment IDs (batch_size, seq_len)
            position_ids: Position IDs (batch_size, seq_len)
            head_mask: Head mask for attention
            inputs_embeds: Optional pre-computed embeddings
            labels: Labels for MLM (batch_size, seq_len), -100 for non-masked
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output hidden states
            return_dict: Whether to return dict (default: True)

        Returns:
            MaskedLMOutput with loss, logits, hidden_states, attentions
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Forward through base model
        outputs = self.cnm(
            input_ids=input_ids,
            struct_ids=struct_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @classmethod
    def from_bert_pretrained(
        cls,
        bert_model_name: str = "bert-base-chinese",
        tree_path: Optional[Path] = None,
        struct_dim: int = 256,
        **kwargs
    ) -> "CNMForMaskedLM":
        """Initialize CNM-BERT from pre-trained BERT weights.

        This is useful for warm-starting training with BERT weights.

        Args:
            bert_model_name: HuggingFace BERT model name
            tree_path: Path to IDS tree JSON
            struct_dim: Structural embedding dimension
            **kwargs: Additional config arguments

        Returns:
            CNMForMaskedLM with BERT weights loaded
        """
        from transformers import BertForMaskedLM

        # Load BERT model
        bert_model = BertForMaskedLM.from_pretrained(bert_model_name)

        # Create CNM config from BERT config
        config = CNMConfig.from_pretrained(bert_model_name)
        config.struct_dim = struct_dim
        config.tree_path = str(tree_path) if tree_path else None
        for key, value in kwargs.items():
            setattr(config, key, value)

        # Create CNM model
        cnm_model = cls(config)

        # Copy BERT weights
        cnm_model.cnm.encoder.load_state_dict(bert_model.bert.encoder.state_dict())
        cnm_model.cnm.embeddings.word_embeddings.load_state_dict(bert_model.bert.embeddings.word_embeddings.state_dict())
        cnm_model.cnm.embeddings.position_embeddings.load_state_dict(bert_model.bert.embeddings.position_embeddings.state_dict())
        cnm_model.cnm.embeddings.token_type_embeddings.load_state_dict(bert_model.bert.embeddings.token_type_embeddings.state_dict())
        cnm_model.cls.load_state_dict(bert_model.cls.state_dict())

        return cnm_model
