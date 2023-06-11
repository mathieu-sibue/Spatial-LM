# -*- coding: utf-8 -*-
import torch.nn as nn

from transformers import RobertaModel, RobertaConfig
from transformers import RobertaForTokenClassification, AutoModelForQuestionAnswering

import torch
from transformers.utils import ModelOutput
from torch.nn import CrossEntropyLoss
from typing import Optional, Tuple, Union


class GraphRobertaTokenClassifier(nn.Module):
    def __init__(self,opt):
        super(GraphRobertaTokenClassifier, self).__init__()
        self.opt = opt
        self.num_labels = opt.num_labels
        self.config = RobertaConfig.from_pretrained(opt.roberta_dir)
        # self.config.num_labels = opt.num_labels
        self.roberta = RobertaModel.from_pretrained(opt.roberta_dir,config=self.config, add_pooling_layer=False)
        classifier_dropout = (
            self.config.classifier_dropout if self.config.classifier_dropout is not None else self.config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(self.config.hidden_size + opt.hidden_dim_2, self.num_labels)

        # self.model = RobertaForTokenClassification.from_pretrained("deepset/roberta-base-squad2")
    
    def forward(self,input_ids, attention_mask, labels, gvect, **args):
        outputs =  self.roberta(input_ids, attention_mask)
        hidden_state = outputs[0]   # sequence output
        fully_sequence = torch.cat((hidden_state,gvect),-1)
        sequence_output = self.dropout(fully_sequence)
        logits = self.classifier(sequence_output)   # shape = (num_bach, dim, num_label), label_shape = (num_batch, dim)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1,self.num_labels),labels.view(-1))
        
        return ModelOutput(
            loss = loss,
            logits = logits
        )


class RobertaTokenClassifier(nn.Module):
    def __init__(self,opt):
        super(RobertaTokenClassifier, self).__init__()
        self.opt = opt
        self.num_labels = opt.num_labels
        self.config = RobertaConfig.from_pretrained(opt.roberta_dir)
        self.roberta = RobertaModel.from_pretrained(opt.roberta_dir,config=self.config, add_pooling_layer=False)
        classifier_dropout = (
            self.config.classifier_dropout if self.config.classifier_dropout is not None else self.config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(self.config.hidden_size, self.num_labels)

        # self.model = RobertaForTokenClassification.from_pretrained("deepset/roberta-base-squad2")
    
    def forward(self,input_ids, attention_mask, labels, **args):
        outputs =  self.roberta(input_ids, attention_mask)
        hidden_state = outputs[0]   # sequence output
        sequence_output = self.dropout(hidden_state)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1,self.num_labels),labels.view(-1))
        
        return ModelOutput(
            loss = loss,
            logits = logits
        )


class RobertaForQA(nn.Module):
    def __init__(self, opt, freeze_bert=False):
        super(RobertaForQA, self).__init__()
        self.opt = opt
        self.config = RobertaConfig.from_pretrained(opt.roberta_dir)
        # self.roberta = RobertaModel(self.config)
        # self.roberta = RobertaModel.from_pretrained(opt.roberta_dir, config=self.config)
        self.roberta = AutoModelForQuestionAnswering.from_pretrained(opt.roberta_dir,config=self.config)

    def forward(self,input_ids, attention_mask,start_positions=None, end_positions=None, **args):
        outputs = self.roberta(
            input_ids = input_ids,
            attention_mask = attention_mask,
            start_positions = start_positions,
            end_positions = end_positions
        )
        return outputs



class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class RobertaForBinaryQA(nn.Module):
    def __init__(self, opt, has_visual_segment_embedding=True):
        super(RobertaForBinaryQA, self).__init__()
        self.opt = opt
        self.config = RobertaConfig.from_pretrained(opt.roberta_dir)
        self.config.num_labels = opt.num_labels
        # self.roberta = RobertaModel(self.config)
        self.roberta = RobertaModel.from_pretrained(opt.roberta_dir, config=self.config)

        self.classifier = RobertaClassificationHead(self.config)
        # self.classifier = nn.Sequential(
        #     nn.Linear(self.opt.input_dim,self.opt.input_dim),   # hidden dim
        #     nn.ReLU(),
        #     nn.Dropout(self.opt.dropout),
        #     nn.Linear(self.opt.input_dim, self.opt.num_labels),
        #     nn.Sigmoid()
        # )

    def forward(self,
                input_ids: Optional[torch.LongTensor] = None, 
                attention_mask: Optional[torch.FloatTensor] = None,
                token_type_ids: Optional[torch.LongTensor] = None,
                labels: Optional[torch.LongTensor] = None,
            ):

        outputs = self.roberta(input_ids = input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids)
        # use the first token CLS state for sentence-level prediction
        # last_hidden_state_cls = outputs[0][:,0,:] # (batch_size, seq_len, dim)  => (batch_size, 1, dim)
        hidden_state = outputs[0]  # (batch_size, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (batch_size, dim) for first token CLS state
        # feet input to classifier to compute logits
        logits = self.classifier(pooled_output)

        # calculate loss
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.opt.num_labels), labels.view(-1))

        return ModelOutput(
            loss=loss,
            logits = logits
        )
