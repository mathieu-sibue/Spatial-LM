from datasets import Dataset, DatasetDict,load_dataset ,Features, Sequence, Value, Array2D, Array3D
from datasets.features import ClassLabel
from transformers import AutoTokenizer,AutoConfig
import transformers
import os
import json
import pandas as pd
import pickle
import numpy as np
# from transformers import RobertaTokenizer

# disable caching!!
import datasets
datasets.disable_caching()

class DocVQA:
    def __init__(self,opt,start_chunk=1) -> None:    
        self.opt = opt
        '''
        {train, val, test}
        Dataset({
            features: ['input_ids', 'bbox', 'attention_mask', 'token_type_ids', 'image', 'start_positions', 'end_positions'],
            num_rows: 50
        })
        three steps: 1 prepare pickle, 2 load dataset, and 3 map dataset features
        '''
        self.config = AutoConfig.from_pretrained(opt.layoutlm_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(opt.layoutlm_dir)
        assert isinstance(self.tokenizer, transformers.PreTrainedTokenizerFast) # get sub

        # 1. load raw train_test pickles
        # for each one_doc = {'tokens':[],'bboxes':[],'widths':[],'heights':[], 'seg_ids':[],'image':None}
        
        if bool(opt.inference_only):
            test_id2qa, test_id2doc = self._load_pickle(self.opt.docvqa_pickles + self.test_split+'.pickle')
            raw_test = self.get_raw_ds('test',test_id2qa, test_id2doc)
            self.trainable_test_ds = self.get_trainable_dataset('test')
        else:
            train_id2qa, train_id2doc = self._load_pickle(self.opt.docvqa_pickles + 'train.pickle')
            val_id2qa, val_id2doc = self._load_pickle(self.opt.docvqa_pickles + 'val.pickle')
            raw_train = self.get_raw_ds('train',train_id2qa, train_id2doc)
            raw_val = self.get_raw_ds('val',val_id2qa, val_id2doc)
            self.trainable_train_ds = self.get_trainable_dataset('train')
            self.trainable_val_ds = self.get_trainable_dataset('val')
            # concatenate 


    # 1.1. raw dataset wrap
    def get_raw_ds(self,split, id2qa,id2doc):
        return Dataset.from_dict(self._load_qa_pairs(split,id2qa, id2doc))

    # 1.2. raw dataset generator
    def _load_qa_pairs(self,split, id2qa,id2doc):
        qIDs = list(id2qa.keys())

        res_dict = {"qID": [],'question':[], 'answers': [], "words": [], "boxes": [],
                         "image_pixel_values": [], 
                         'ans_start':[], 'ans_end':[]}
        for qID in qIDs:
            docID_page, question, answers, ans_word_idx_start, ans_word_idx_end = id2qa[qID]
            # remove those that could not have found answers
            if bool(self.opt.filter_no_answer) and (split != 'test') and ans_word_idx_end == 0: continue

            # get the corresponding doc:
            # for each doc, keys = {tokens,bboxes,seg_ids,widths,heights,image }
            doc = id2doc[docID_page]

            res_dict['qID'].append(qID)
            res_dict['question'].append(question)
            res_dict['answers'].append(answers)
            res_dict['words'].append(doc['tokens'])
            res_dict['boxes'].append(doc['bboxes'])
            res_dict['image_pixel_values'].append(doc['image'])
            res_dict['ans_start'].append(ans_word_idx_start)
            res_dict['ans_end'].append(ans_word_idx_end)
        return res_dict


    # 2.1 wrap mapped features
    def get_trainable_dataset(self,split='train',shuffle=True):
        '''
        return sth like:
        Dataset({
            features: ['input_ids', 'bbox', 'attention_mask', 'token_type_ids', 'image', 'ans_start_positions', 'ans_end_positions'],
            num_rows: 50
        })
        '''
        # 1. data split return self.prepare_one_doc(self.dataset[split])
        dataset = self.train_test_dataset[split]   # get split
        # 2. feature definition (final shape to model)
        features = Features({
                'questionId': Value(dtype='int64'),
                'input_ids': Sequence(feature=Value(dtype='int64')),
                'bbox': Array2D(dtype="int64", shape=(512, 4)),
                'attention_mask': Sequence(Value(dtype='int64')),
                'token_type_ids': Sequence(Value(dtype='int64')),
                'pixel_values': Array3D(dtype="float32", shape=(3, 224, 224)),
                'ans_start_positions': Value(dtype='int64'),
                'ans_end_positions': Value(dtype='int64'),
                'gvect': Array2D(dtype="float32", shape=(512,32)),
            })
        print('start mapping for', split)
        # 3. produce dataset
        trainable_dataset = dataset.map(
            self._prepare_one_batch,
            batched=True, 
            batch_size = self.opt.batch_size,
            remove_columns=dataset.column_names,    # remove original features
            features = features,
            writer_batch_size=3_000,
            # num_proc=1, # cpu number, or set to =psutil.cpu_count(),
            # load_from_cache=False,
        ).with_format("torch")
        # trainable_samples = trainable_samples.set_format("torch")  # with_format is important
        # 4. return dataset
        # before return, we can delete data for space saving
        del dataset # !!!!!!! not sure!!!!!
        return trainable_dataset


    # 2.2. map function definition
    def _prepare_one_batch(self,examples):
        # 1. take a batch 
        question = examples['question']
        answers = examples['answers']
        words = examples['words']
        boxes = examples['boxes']
        ans_starts = examples['ans_start']
        ans_ends = examples['ans_end']

        # 2. encode it
        encoding = self.tokenizer(question, words, boxes, max_length=self.opt.max_seq_len, padding="max_length", truncation=True, return_token_type_ids=True)

        # 3. next, add start_positions and end_positions
        ans_start_positions = []
        ans_end_positions = []
        
        # for every example in the batch:
        for batch_index in range(len(answers)):
            ans_word_idx_start, ans_word_idx_end = ans_starts[batch_index],ans_ends[batch_index]
            # for test without answers, we just skip them!
            if (not answers[batch_index]) or (ans_word_idx_end==0):
                cls_index = encoding.input_ids[batch_index].index(self.tokenizer.cls_token_id)
                ans_start_positions.append(cls_index)
                ans_end_positions.append(cls_index)
                        
            # 3.2 step2: match answer range in the sequence e.g., [None, 0,1,2,2,2,3,3,4,5,6,7,7,7, None] to get index range
            else:
                answer_start_index, answer_end_index = self._ans_index_range(encoding,batch_index,ans_word_idx_start, ans_word_idx_end)
                ans_start_positions.append(answer_start_index)
                ans_end_positions.append(answer_end_index)
                # print("Verifying start position and end position:===")
                # print("True answer:", answers[batch_index])
                # reconstructed_answer = self.tokenizer.decode(encoding.input_ids[batch_index][answer_start_index:answer_end_index+1])
                # print("Reconstructed answer:", reconstructed_answer)
                # print("-----------")

        # 3.3 append the ans_start, ans_end_index
        encoding['ans_start_positions'] = ans_start_positions
        encoding['ans_end_positions'] = ans_end_positions
        # 4 append the rest features
        encoding['pixel_values'] = examples['image_pixel_values']   # sometimes, it needs to change it into open Image !!!!!
        encoding['questionId'] = examples['qID']

        return encoding


    def _ans_index_range(self, batch_encoding,batch_index, answer_word_idx_start, answer_word_idx_end):
        sequence_ids = batch_encoding.sequence_ids(batch_index) # types 0s and 1s
        # Start token index of the current span in the text.
        left = 0
        while sequence_ids[left] != 1:
            left += 1
        # End token index of the current span in the text.
        right = len(batch_encoding.input_ids[batch_index]) - 1
        while sequence_ids[right] != 1:
            right -= 1

        sub_word_ids = batch_encoding.word_ids(batch_index)[left:right+1]
        for id in sub_word_ids:
            if id == answer_word_idx_start:
                break
            else:
                left += 1
        for id in sub_word_ids[::-1]:
            if id == answer_word_idx_end:
                break
            else:
                right -= 1
        # return the result (ans_index_start, ans_index_end)
        return [left,right]
        

    def _load_pickle(self,picke_path):
        with open(picke_path,'rb') as fr:
            res = pickle.load(fr)
        return res



if __name__ == '__main__':
    # Section 1, parse parameters
    mydata = CORD(None)
    test_dataset = mydata.get_data(split='test')
    print(test_dataset)
    doc1 = test_dataset[0]
    print(doc1['input_ids'])

    '''
    Dataset({
        features: ['input_ids', 'attention_mask', 'bbox', 'labels', 'pixel_values'],
        num_rows: 100
    })
    '''

