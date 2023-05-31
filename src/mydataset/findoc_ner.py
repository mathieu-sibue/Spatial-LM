from datasets import load_dataset ,Features, Sequence, Value, Array2D, Array3D, Dataset, DatasetDict
from datasets.features import ClassLabel
from transformers import AutoProcessor, AutoTokenizer, AutoConfig
import os
import json
import transformers
from PIL import Image
from mydataset import myds_util

class FinDoc:
    def __init__(self,opt) -> None:    
        self.opt = opt
        '''
        DatasetDict({
            train: Dataset({
                features: ['id', 'words', 'bboxes', 'ner_tags', 'image'],
                num_rows: 800
            })
        })
        '''
        # step1: create config, tokenizer, and processor
        self.config = AutoConfig.from_pretrained(opt.layoutlm_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(opt.layoutlm_dir)
        assert isinstance(self.tokenizer, transformers.PreTrainedTokenizerFast)  # get sub
        self.processor = AutoProcessor.from_pretrained(opt.layoutlm_dir,tokenizer=self.tokenizer, apply_ocr=False)    # wrap of featureExtract & tokenizer

        self.label_col_name = "ner_tags"

        # step 2.1: get raw ds (already normalized bbox, img object)
        raw_train, raw_test = self.get_raw_ds()
        # step 2.2, get  labeled ds (get label list, and map label dataset)
        opt.id2label, opt.label2id, opt.label_list = self._get_label_map(raw_train)
        opt.num_labels = len(opt.label_list)
        # map label ds
        train_label_ds, test_label_ds = self.get_label_ds(raw_train), self.get_label_ds(raw_test)

        normed_train, normed_test = self.get_normed_ds(train_label_ds), self.get_normed_ds(test_label_ds)

        # step 3: prepare for getting trainable data (encode and define features)
        trainable_train_ds, trainable_test_ds = self.get_preprocessed_ds(normed_train),self.get_preprocessed_ds(normed_test)
        print(trainable_train_ds)
        print(trainable_test_ds)

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
                bbox =  sample['tboxes']
            else:
                bbox = sample['bboxes']
            sample['bboxes'] = [self._normalize_bbox(bbox, size) for bbox in sample['bboxes']]
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

            # get from entities
            tokID_text = {}
            tokID_tbox = {}
            tokID_NER = {}

            # get from annotations
            tokID_bbox = {}

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
                tokID_NER[tokID] = tok_ner
            
            for entity in doc['annotations']:
                bbox = entity['bbox']
                bbox = [bbox[0],bbox[1],bbox[0]+bbox[2], bbox[1]+bbox[3]]
                for tokID in entity['token_ids']:
                    tokID = int(tokID)
                    tokID_bbox[tokID] = bbox

            # combine 
            tokens = []
            tboxes = []
            bboxes = []
            ner_tags = []
        
            for tokID in sorted(tokID_text.keys()):
                tokens.append(tokID_text[tokID])
                tboxes.append(tokID_tbox[tokID])
                if tokID in tokID_bbox.keys():
                    bboxes.append(tokID_bbox[tokID])
                else:
                    bboxes.append(tokID_tbox[tokID])
                ner_tags.append(tokID_NER[tokID])

            yield {"id": doc_idx, "tokens": tokens,"tboxes":tboxes, "bboxes": bboxes, "ner_tags": ner_tags, 
                "image_path": img_path, "size":size}
            # print(item)


    def get_preprocessed_ds(self,ds):
        def _preprocess(batch):
            # 1) encode words and imgs
            encodings = self.processor(images=batch['image'],text=batch['tokens'], boxes=batch['bboxes'],
                                       word_labels=batch['ner_tags'], truncation=True, padding='max_length', max_length=self.opt.max_seq_len)
            # 2) add position_ids
            # position_ids = []
            # for i, block_ids in enumerate(batch['block_ids']):
            #     word_ids = encodings.word_ids(i)
            #     rel_pos = self._get_rel_pos(word_ids, block_ids)
            #     position_ids.append(rel_pos)
            # encodings['position_ids'] = position_ids

            return encodings


        features = Features({
            'pixel_values': Array3D(dtype="float32", shape=(3, 224, 224)),
            'input_ids': Sequence(feature=Value(dtype='int64')),
            # 'position_ids': Sequence(feature=Value(dtype='int64')),
            'attention_mask': Sequence(Value(dtype='int64')),
            'bbox': Array2D(dtype="int64", shape=(512, 4)),
            'labels': Sequence(feature=Value(dtype='int64')),
        })
        # processed_ds = ds.map(_preprocess, batched=True, num_proc=self.cpu_num, 
        #     remove_columns=['id','tokens', 'bboxes','ner_tags','block_ids','image'], features=features).with_format("torch")
        processed_ds = ds.map(_preprocess, batched=True, num_proc=os.cpu_count(), 
            remove_columns=ds.column_names, features=features,batch_size=200).with_format("torch")
    
        # process to: 'input_ids', 'position_ids','attention_mask', 'bbox', 'labels', 'pixel_values']
        return processed_ds


    def get_label_ds(self,ds):
        def map_label2id(sample):
            # sample['ner_tags'] = [self.opt.label2id[ner_label] for ner_label in sample['ner_tags'] if ner_label>=0 else -100]
            new_tags = []
            for ner_label in sample['ner_tags']:
                if ner_label>=0:
                    new_tags.append(self.opt.label2id[ner_label])
                else: 
                    new_tags.append(-100)
            sample['ner_tags'] = new_tags
            return sample
        label_ds = ds.map(map_label2id, num_proc=os.cpu_count())
        return label_ds


    def _get_label_list(self,labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        label_list = list(unique_labels)
        if label_list[0]<0:
            label_list.pop(0)
        label_list.sort()
        return label_list
    # get label list
    def _get_label_map(self,ds):
        features = ds.features
        column_names = ds.column_names
        if isinstance(features[self.label_col_name].feature, ClassLabel):
            label_list = features[self.label_col_name].feature.names
            # No need to convert the labels since they are already ints.
            id2label = {k: v for k,v in enumerate(label_list)}
            label2id = {v: k for k,v in enumerate(label_list)}
        else:
            label_list = self._get_label_list(ds[self.label_col_name])   # this invokes another function
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

if __name__ == '__main__':
    # Section 1, parse parameters
    mydata = FUNSD(None)
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

