import torch
import torch.nn as nn
from transformers.modeling_outputs import MaskedLMOutput
from typing import List, Optional, Tuple, Union
from transformers import AutoConfig, AutoModel, LayoutLMv3Model, AutoModelForTokenClassification, AutoModelForQuestionAnswering
from transformers.activations import gelu  # ACT2FN
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig

class SpatialLMConfig(PretrainedConfig):
    model_type = "spatial_lm"

    def __init__(
        self,
        vocab_size=50265,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        max_2d_position_embeddings=1024,
        coordinate_size=128,
        shape_size=128,
        has_relative_attention_bias=False,
        rel_pos_bins=32,
        max_rel_pos=128,
        rel_2d_pos_bins=64,
        max_rel_2d_pos=256,
        has_spatial_attention_bias=True,
        text_embed=True,
        visual_embed=True,
        input_size=224,
        num_channels=3,
        patch_size=16,
        classifier_dropout=None,
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
        self.max_2d_position_embeddings = max_2d_position_embeddings
        self.coordinate_size = coordinate_size
        self.shape_size = shape_size
        self.has_relative_attention_bias = has_relative_attention_bias
        self.rel_pos_bins = rel_pos_bins
        self.max_rel_pos = max_rel_pos
        self.has_spatial_attention_bias = has_spatial_attention_bias
        self.rel_2d_pos_bins = rel_2d_pos_bins
        self.max_rel_2d_pos = max_rel_2d_pos
        self.text_embed = text_embed
        self.visual_embed = visual_embed
        self.input_size = input_size
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.classifier_dropout = classifier_dropout


class SpatialLMPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = AutoConfig
    base_model_prefix = "spatial_lm"

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

class SpatialLMHead(nn.Module):
    """copied from Roberta Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)

        return x

    def _tie_weights(self):
        # To tie those two weights if they get disconnected (on TPU or when the bias is resized)
        # For accelerate compatibility and to not break backward compatibility
        if self.decoder.bias.device.type == "meta":
            self.decoder.bias = self.bias
        else:
            self.bias = self.decoder.bias


class SpatialLMForMaskedLM(SpatialLMPreTrainedModel):
    def __init__(self, config, start_dir_path=None):
        super(SpatialLMForMaskedLM, self).__init__(config)
        config.has_relative_attention_bias = False

        # if the model is loaded the first time, we take advantage of the layoutlm
        if start_dir_path:
            self.spatial_lm = LayoutLMv3Model.from_pretrained(start_dir_path, config = config)
        else:
            self.spatial_lm = LayoutLMv3Model(config)

        self.lm_head = SpatialLMHead(config)

        # The LM head weights require special treatment only when they are tied with the word embeddings
        # self.update_keys_to_ignore(config, ["lm_head.decoder.weight"])
        # self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        bbox: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
        pixel_values: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:

        outputs = self.spatial_lm(
            input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            pixel_values=pixel_values,
        )

        # get [B, S, D] (not pooled), and predict complete sequence
        # sequence_output = outputs[0]
        # We need to remove the vision part
        input_shape = input_ids.size()
        seq_length = input_shape[1]
        # only take the text part of the output representations, i.e., [B, S-text, D]
        sequence_output = outputs[0][:, :seq_length]

        prediction_scores = self.lm_head(sequence_output)

        # calculate the loss btw predicted sequence and true sequence
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size),
                labels.view(-1),
            )
        else:
            print('==== why there is no sequencial labels?=======')

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        return outputs


class SpatialLMClassificationHead(nn.Module):
    """
    Head for sentence-level classification tasks. Reference: RobertaClassificationHead
    """

    def __init__(self, config, pool_feature=False):
        super().__init__()
        self.pool_feature = pool_feature
        if pool_feature:
            self.dense = nn.Linear(config.hidden_size * 3, config.hidden_size)
        else:
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class SpatialLMForTokenclassifier(SpatialLMPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.spatial_lm = LayoutLMv3Model(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if config.num_labels < 10:
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        else:
            self.classifier = SpatialLMClassificationHead(config, pool_feature = False)
        
        self.init_weights()


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        bbox: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
        pixel_values: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:

        outputs = self.spatial_lm_token_classifier(
            input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            pixel_values=pixel_values,
        )

        input_shape = input_ids.size()
        seq_length = input_shape[1]

        sequence_output = outputs[0][:, :seq_length]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        return outputs



class SpatialLMForDocVQA(SpatialLMPreTrainedModel):
    def __init__(self, opt, freeze_bert=False):
        super(SpatialLMForDocVQA, self).__init__()
        self.opt = opt
        # self.config = AutoConfig.from_pretrained(opt.spatial_lm_dir, num_labels=xxx)
        # self.spatial_lm = LayoutLMv3Model(config=self.config)
        self.spatial_lm_token_classifier = AutoModelForQuestionAnswering.from_pretrained(opt.spatial_lm_dir)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        bbox: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
        pixel_values: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:

        outputs = self.spatial_lm_token_classifier(
            input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            pixel_values=pixel_values,
            start_positions = start_positions,
            end_positions = end_positions
        )

        return outputs


if __name__=='__main__':
    spatial_mlm = SpatialLMForMaskedLM(None)
    print(spatial_mlm)

