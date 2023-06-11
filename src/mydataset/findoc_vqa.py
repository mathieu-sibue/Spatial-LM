from datasets import load_dataset ,Features, Sequence, Value, Array2D, Array3D, Dataset, DatasetDict
from datasets.features import ClassLabel
from transformers import AutoProcessor, AutoTokenizer, AutoConfig
import os
import json
import transformers
from PIL import Image


class FinDoc:
    def __init__(self,opt) -> None:    
        self.opt = opt

        # step1: create config, tokenizer, and processor
        self.config = AutoConfig.from_pretrained(opt.layoutlm_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(opt.layoutlm_dir)
        assert isinstance(self.tokenizer, transformers.PreTrainedTokenizerFast)  # get sub
        self.processor = AutoProcessor.from_pretrained(opt.layoutlm_dir,tokenizer=self.tokenizer, apply_ocr=False)    # wrap of featureExtract & tokenizer

        # self.label_col_name = "doc_class"

        # step 2.1: get raw ds (already normalized bbox, img object)
        raw_train, raw_test = self.get_raw_ds()
        # step 2.2, get  label_list and label2id dict
        # opt.id2label, opt.label2id, opt.label_list = self._get_label_map(raw_train)
        opt.num_labels = 2
        # print('label list:',opt.label_list)
        # load img and norm bbox with max=1000
        normed_train, normed_test = self.get_normed_ds(raw_train), self.get_normed_ds(raw_test)

        # step 3: prepare for getting trainable data (encode and define features)
        trainable_train_ds, trainable_test_ds = self.get_preprocessed_ds(normed_train),self.get_preprocessed_ds(normed_test)

        # print('=====',opt.label_list)
        
        self.trainable_ds = DatasetDict({
            "train" : trainable_train_ds.shuffle(seed=88), 
            "test" : trainable_test_ds 
        })

    def get_normed_ds(self,raw_ds):
        def _load_imgs_obj(sample):
            # 1) load img
            sample['image'],_ = self._load_image(sample['image_path'])
            # 2) norm boxes
            tbox = sample['tboxes']
            size = sample['size']
            if self.opt.bbox_type == 'tbox':
                boxes =  sample['tboxes']
            else:
                boxes = sample['bboxes']
            # norm to 1000
            sample['bboxes'] = [self._normalize_bbox(bbox, size) for bbox in boxes]
            return sample
   
        ds = raw_ds.map(_load_imgs_obj, num_proc=os.cpu_count())
        return ds


    def get_raw_ds(self):
        train = Dataset.from_generator(self.ds_generator, gen_kwargs={'base_dir':self.opt.findoc_dir,'split':'train'})
        test = Dataset.from_generator(self.ds_generator, gen_kwargs={'base_dir':self.opt.findoc_dir,'split':'test'})
        # train = Dataset.from_dict(self.ds_generator(self.opt.findoc_dir,'val'))
        return train, test

    # split = train, test, val
    def ds_generator(self, base_dir, split='val'):
        # logger.info("⏳ Generating examples from = %s", filepath)

        json_path = os.path.join(base_dir,split+'.json')
        with open(json_path, "r", encoding="utf8") as f:
            data = json.load(f)

        for doc in data:
            # print('---doc id:---',doc_idx)
            doc_idx = doc['doc_id']
            size = (doc['metadata']['width'],doc['metadata']['height'])
            img_path = doc['metadata']['image_path']
            img_path = os.path.join(base_dir, img_path)

            doc_class_id = doc['metadata']['doc_class_id']  # dont use for now


            # 1) get from entities (page tokens)
            tokID_text = {}
            tokID_tbox = {}
            tokID_NER = {}

            for token in doc['tokens']:
                tokID = int(token['token_id'])
                x1,y1 = token['x'],token['y']
                tbox = [x1,y1,x1+token['width'],y1+token['height']]
                tok_text = token['text'].strip()
                if not tok_text: continue   # remove empty text
                tok_ner = token['class_id']

                # add
                tokID_text[tokID] = tok_text
                tokID_tbox[tokID] = tbox
                if tok_ner<0:
                    ner = 'O'
                else:
                    ner = 'I-'+str(tok_ner)
                tokID_NER[tokID] = ner
            
            # 2) get from annotations (entity ners and bbox)
            tokID_bbox = {}

            for entity in doc['annotations']:
                bbox = entity['bbox']
                bbox = [bbox[0],bbox[1],bbox[0]+bbox[2], bbox[1]+bbox[3]]
                # add B-
                tok_ner = entity['class_id']
                if tok_ner>=0:
                    ordered_tokIDs = sorted(entity['token_ids'], key=lambda tok_id: (tokID_tbox[tok_id][0], tokID_tbox[tok_id][1]))
                    first_tokID = ordered_tokIDs[0]
                    # add B-
                    tokID_NER[first_tokID] = 'B-'+str(tok_ner)

                # add bbox
                for tokID in entity['token_ids']:
                    tokID = int(tokID)
                    tokID_bbox[tokID] = bbox

            # 3) combine above two bulletpoints
            tokens = []
            tboxes = []
            bboxes = []
        
            for tokID in sorted(tokID_text.keys()):
                tokens.append(tokID_text[tokID])
                tboxes.append(tokID_tbox[tokID])
                if tokID in tokID_bbox.keys():
                    bboxes.append(tokID_bbox[tokID])
                else:
                    bboxes.append(tokID_tbox[tokID])

            # 4) add QA pairs and share the use of doc 
            for vqa in doc['qa']:
                if vqa['type'] == 'span':
                    q,a  = vqa['question'], vqa['answer']
                    sorted_ans = sorted(a[0])
                    yield {
                        "id": doc_idx, "tokens": tokens,"tboxes":tboxes, "bboxes": bboxes, 
                        "image_path":img_path, "size":size,
                        'question': q, 'ans_start': sorted_ans[0], 'ans_end': sorted_ans[-1]
                    }

            # yield {"id": doc_idx, "tokens": tokens,"tboxes":tboxes, "bboxes": bboxes, "doc_class": doc_class_id, 
            #     "image_path": img_path, "size":size}
            # print(item)


    def get_processed_for_bertx(self,ds):
        def _preprocess(sample):
            token_boxes = []
            for word, box in zip(sample['tokens'], sample['bboxes']):
                word_tokens = self.tokenizer.tokenize(word)
                token_boxes.extend([box] * len(word_tokens))
            # add bounding boxes of cls + sep tokens
            token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]*(self.opt.max_seq_len-len(token_boxes)-1)
            # == wrap the sample into: {input_ids, token_type_ids, attention_mask, bbox, labels} ==
            # 1) tokenize, here, you must join the 'tokens' as one 'text' in a conventional way
            encodings = self.tokenizer(sample['question'], " ".join(sample['tokens']), return_token_type_ids = True,
                max_length=self.opt.max_seq_len, padding="max_length", truncation=True)
            # 2) add extended bbox and labels
            encodings['bbox'] = token_boxes[:self.opt.max_seq_len]
            # 3) add start and end positions
            ans_starts = sample['ans_start']
            ans_ends = sample['ans_end']

            # print(encodings)
            # print(ans_starts)
            # print(ans_ends)
            # print(sample['question'])
            # print([sample['tokens'][i] for i in range(ans_starts,ans_ends+1)])
 
            answer_start_index, answer_end_index = self._ans_index_range(encodings,-1,ans_starts, ans_ends)

            encodings['start_positions'] = answer_start_index
            encodings['end_positions'] = answer_end_index

            return encodings

        features = Features({
            'input_ids': Sequence(feature=Value(dtype='int64')),
            # 'position_ids': Sequence(feature=Value(dtype='int64')),
            'token_type_ids':Sequence(feature=Value(dtype='int64')),
            'attention_mask': Sequence(Value(dtype='int64')),
            'bbox': Array2D(dtype="int64", shape=(512, 4)),
            'start_positions': Value(dtype='int64'),
            'end_positions': Value(dtype='int64'),
        })
        processed_ds = ds.map(_preprocess, num_proc=os.cpu_count(), 
            remove_columns=ds.column_names, features=features).with_format("torch") 
        return processed_ds


    def get_preprocessed_ds(self,ds):
        # old dataset style
        if self.opt.network_type in ['bert','layoutlmv1']:
            return self.get_processed_for_bertx(ds)

        # otherwise
        def _preprocess(batch):
            if self.opt.network_type == 'layoutlmv2': 
                return_type = True
            else:
                return_type = False
            # no idea why layoutlmv2 use token_type_ids, maybe just default with no reason
            encodings = self.processor(batch['image'],batch['question'], batch['tokens'], boxes=batch['bboxes'],
                            return_token_type_ids = return_type,
                            truncation=True, padding='max_length', max_length=self.opt.max_seq_len)
            
            # 2) add labels separately for sequence classification
            ans_starts = batch['ans_start']
            ans_ends = batch['ans_end']
            ans_start_positions = []
            ans_end_positions = []
            for batch_index in range(len(ans_starts)):
                ans_word_idx_start, ans_word_idx_end = ans_starts[batch_index],ans_ends[batch_index]

                answer_start_index, answer_end_index = self._ans_index_range(encodings,batch_index,ans_word_idx_start, ans_word_idx_end)
                ans_start_positions.append(answer_start_index)
                ans_end_positions.append(answer_end_index)
                # print("Verifying start position and end position:===")
                # print("True answer:", answers[batch_index])
                # reconstructed_answer = self.tokenizer.decode(encoding.input_ids[batch_index][answer_start_index:answer_end_index+1])
                # print("Reconstructed answer:", reconstructed_answer)
                # print("-----------")

            # 3.3 append the ans_start, ans_end_index
            encodings['start_positions'] = ans_start_positions
            encodings['end_positions'] = ans_end_positions

            return encodings


        if self.opt.network_type == 'layoutlmv2':
            features = Features({
                'image' : Array3D(dtype="float32", shape=(3, 224, 224)),
                'input_ids': Sequence(feature=Value(dtype='int64')),
                'token_type_ids':Sequence(feature=Value(dtype='int64')),
                'attention_mask': Sequence(Value(dtype='int64')),
                'bbox': Array2D(dtype="int64", shape=(512, 4)),
                'labels': Value(dtype='int64'),
            })
        else:
            features = Features({
                'pixel_values' : Array3D(dtype="float32", shape=(3, 224, 224)),
                'input_ids': Sequence(feature=Value(dtype='int64')),
                'attention_mask': Sequence(Value(dtype='int64')),
                'bbox': Array2D(dtype="int64", shape=(512, 4)),
                'labels': Value(dtype='int64'),
            })
        # processed_ds = ds.map(_preprocess, batched=True, num_proc=self.cpu_num, 
        #     remove_columns=['id','tokens', 'bboxes','ner_tags','block_ids','image'], features=features).with_format("torch")
        processed_ds = ds.map(_preprocess, batched=True, num_proc=os.cpu_count(), 
            remove_columns=ds.column_names, features=features,batch_size=200).with_format("torch")
    
        # process to: 'input_ids', 'position_ids','attention_mask', 'bbox', 'labels', 'pixel_values']
        return processed_ds


    # get label list
    def _get_label_map(self,ds):
        label_list = list(set(ds[self.label_col_name]))   # this invokes another function
        label_list.sort()
        id2label = {k: v for k,v in enumerate(label_list)}
        label2id = {v: k for k,v in enumerate(label_list)}

        return id2label, label2id, label_list

    def _load_image(self,image_path):
        image = Image.open(image_path).convert("RGB")
        w, h = image.size
        return image, (w, h)
    def _normalize_bbox(self,bbox, size):
        return [
            int(1000 * bbox[0] / size[0]),
            int(1000 * bbox[1] / size[1]),
            int(1000 * bbox[2] / size[0]),
            int(1000 * bbox[3] / size[1]),
        ]
    def _get_line_bbox(self,bboxs):
        x = [bboxs[i][j] for i in range(len(bboxs)) for j in range(0, len(bboxs[i]), 2)]
        y = [bboxs[i][j] for i in range(len(bboxs)) for j in range(1, len(bboxs[i]), 2)]
        x0, y0, x1, y1 = min(x), min(y), max(x), max(y)
        assert x1 >= x0 and y1 >= y0
        bbox = [[x0, y0, x1, y1] for _ in range(len(bboxs))]
        return bbox

    def _get_rel_pos(self,word_ids, block_ids):   # [None, 0, 1, 2, 2, 3, None]; [1,1,2,2] which is dict {word_idx: block_num}
        res = []
        rel_cnt = self.config.pad_token_id+1
        prev_block = 1
        for word_id in word_ids:
            if word_id is None:
                res.append(self.config.pad_token_id)
                continue
            else:
                curr_block = block_ids[word_id]   # word_id is the 0,1,2,3,.. word index;
                if curr_block != prev_block:
                    # set back to 0; 
                    rel_cnt = self.config.pad_token_id+1
                    res.append(rel_cnt) # operate
                    # reset prev_block
                    prev_block = curr_block
                else:
                    res.append(rel_cnt)
            rel_cnt+=1
        return res
    
    # get range
    def _ans_index_range(self, batch_encoding,batch_index, answer_word_idx_start, answer_word_idx_end):
        if batch_index == -1:
            sequence_ids = batch_encoding['token_type_ids']
            input_ids = batch_encoding['input_ids']
            word_ids = batch_encoding.word_ids()
        else:
            sequence_ids = batch_encoding.sequence_ids(batch_index) # types 0s and 1s
            input_ids = batch_encoding.input_ids[batch_index]
            word_ids = batch_encoding.word_ids(batch_index)

        # Start token index of the current span in the text.
        left = 0
        while sequence_ids[left] != 1:
            left += 1
        # End token index of the current span in the text.
        right = len(input_ids) - 1
        while sequence_ids[right] != 1:
            right -= 1

        sub_word_ids = word_ids[left:right+1]
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



if __name__ == '__main__':
    from params import Params

    # Section 1, parse parameters
    opt = Params()
    opt.layoutlm_dir = '/home/ubuntu/air/vrdu/models/layoutlmv2.base'
    opt.findoc_dir='/home/ubuntu/air/vrdu/datasets/no_rare'
    opt.network_type = 'layoutlmv2'
    opt.bbox_type = 'tbox'
    opt.max_seq_len = 512

    mydata = FinDoc(opt)
    print(mydata.trainable_ds)
    print('finished')

    '''
    DatasetDict({
        train: Dataset({
            features: ['image', 'input_ids', 'token_type_ids', 'attention_mask', 'bbox', 'labels'],
            num_rows: 1220
        })
        test: Dataset({
            features: ['image', 'input_ids', 'token_type_ids', 'attention_mask', 'bbox', 'labels'],
            num_rows: 350.
        })
    })
    '''

