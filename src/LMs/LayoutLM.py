# -*- coding: utf-8 -*-

import torch.nn as nn

from transformers import AutoConfig, LayoutLMModel
from transformers import LayoutLMForTokenClassification, AutoModelForTokenClassification
from transformers import AutoModelForQuestionAnswering
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import ModelOutput,MaskedLMOutput,TokenClassifierOutput, SequenceClassifierOutput,QuestionAnsweringModelOutput



class LayoutLM4DocVQA(nn.Module):
    def __init__(self,opt,freeze_bert=False):
        super(LayoutLM4DocVQA, self).__init__()
        self.opt = opt

        self.layoutlm = AutoModelForQuestionAnswering.from_pretrained(opt.layoutlm_dir)

    # dif: token_type_ids; start and end positions -> labels; image -> no pixel_values
    def forward(self,input_ids, token_type_ids, bbox, attention_mask, pixel_values, start_positions=None, end_positions=None,**args):
        # print('input_ids:',input_ids.shape)
        # print('token_type_ids:',token_type_ids.shape)
        # print('bbox:',bbox.shape)
        # print('attention_mask:',attention_mask.shape)
        # print('pixel_values:',pixel_values.shape)

        outputs = self.layoutlm(
            input_ids = input_ids,
            token_type_ids = token_type_ids,
            bbox = bbox,
            attention_mask = attention_mask,
            pixel_values = pixel_values, 
            start_positions = start_positions,
            end_positions = end_positions
        )
        return outputs


class LayoutLMTokenclassifier(nn.Module):
    def __init__(self, opt, freeze_bert=False):
        super(LayoutLMTokenclassifier, self).__init__()
        self.opt = opt
        self.num_labels = opt.num_labels
        self.config = AutoConfig.from_pretrained(opt.layoutlm_dir)
        # self.roberta = RobertaModel(self.config)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)

        self.layoutlm = LayoutLMModel.from_pretrained(opt.layoutlm_dir, num_labels=opt.num_labels, label2id=opt.label2id, id2label=opt.id2label)
        self.classifier = nn.Linear(self.config.hidden_size, self.num_labels)
        # freeze the bert model
        if freeze_bert:
            for param in self.roberta.parameters():
                param.requires_grad = False
        

    # def forward(self,input_ids, bbox, attention_mask, pixel_values, labels, **args):
    def forward(self,input_ids, bbox, attention_mask, token_type_ids,labels, **args):
        # print('input id:',input_ids.size())
        outputs = self.layoutlm(input_ids = input_ids, bbox = bbox, attention_mask = attention_mask,token_type_ids=token_type_ids)

        input_shape = input_ids.size()
        seq_length = input_shape[1]

        hidden_state = outputs[0]

        sequence_output = hidden_state[:, :seq_length]   # take (batch_size, 192, dim)
        sequence_output = self.dropout(sequence_output)
        
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.opt.num_labels), labels.view(-1))

        return ModelOutput(
            loss=loss,
            logits = logits
        )

        # outputs = self.layoutlm(input_ids = input_ids, bbox = None, attention_mask = attention_mask, pixel_values = None, labels = labels)
        # return outputs

