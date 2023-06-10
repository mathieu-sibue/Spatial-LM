from datasets import load_dataset ,load_from_disk, concatenate_datasets,Features, Sequence, Value, Array2D, Array3D, Dataset, DatasetDict
from datasets.features import ClassLabel
from transformers import AutoProcessor, AutoTokenizer, AutoConfig
import os
import json
import transformers
from PIL import Image
from mydataset import myds_util
import datasets

class FUNSDPLUS:
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
        # raw_train = self.get_raw_ds(opt.funsdplus_train)
        raw_train = load_from_disk('/home/ubuntu/air/vrdu/datasets/funsd_plus/funsd+train.h')
        raw_val = self.get_raw_ds(opt.funsdplus_val)
        # raw_test = self.get_raw_ds(opt.funsdplus_test)
        raw_test = load_from_disk('/home/ubuntu/air/vrdu/datasets/funsd_plus/funsd+test.h')

        # raw_train.save_to_disk('funsd+train.h')
        # raw_test.save_to_disk('funsd+test.h')

        # step 2.2, get  labeled ds (get label list, and map label dataset)
        _,_, opt.label_list = self._get_label_map(raw_train)
        opt.num_labels = len(opt.label_list)
        self.class_label = ClassLabel(num_classes=opt.num_labels, names = opt.label_list)
        print(self.class_label)

        # map label ds
        label_train_ds, label_val_ds, label_test_ds = self.get_label_ds(raw_train), self.get_label_ds(raw_val), self.get_label_ds(raw_test)

        # step 3: prepare for getting trainable data (encode and define features)
        trainable_train_ds, trainable_val_ds, trainable_test_ds = \
            self.get_preprocessed_ds(label_train_ds),self.get_preprocessed_ds(label_val_ds), self.get_preprocessed_ds(label_test_ds)

        # trainable_train_ds = concatenate_datasets(trainable_train_ds,trainable_val_ds)
        self.trainable_ds = DatasetDict({
            "train" : trainable_train_ds , 
            "test" : trainable_test_ds 
        })

    def get_raw_ds(self, base_dir):
        raw_ds = Dataset.from_generator(self.ds_generator, gen_kwargs={'base_dir':base_dir})
        # raw_ds = Dataset.from_dict(self.ds_generator, gen_kwargs={'base_dir':base_dir})
        # raw_ds = self.ds_generator(base_dir)
        return raw_ds

    def ds_generator(self, base_dir):
        # logger.info("⏳ Generating examples from = %s", filepath)
        ann_dir = os.path.join(base_dir, "annotations")
        img_dir = os.path.join(base_dir, "images")
        for doc_idx, file in enumerate(sorted(os.listdir(ann_dir))):
            # print('---doc id:---',doc_idx)
            tokens = []
            bboxes = []
            tboxes = []
            ner_tags = []
            block_ids = []

            file_path = os.path.join(ann_dir, file)
            with open(file_path, "r", encoding="utf8") as f:
                data = json.load(f)
            image_path = os.path.join(img_dir, file)
            image_path = image_path.replace("json", "png")
            image, _ = self._load_image(image_path)
            block_idx = 1

            size = (data['bbox'][2],data['bbox'][3])
            for block in data["children"]:
                for field in block['children']:
                    label = field["label"]
                    text = field['value']['value']
                    bbox = field['value']['bbox']   # b-box
                    norm_tbox = self._normalize_bbox(bbox, size)

                    if text.strip() == '': continue
                    words = [w for w in text.split() if w.strip() != ""]
                    if len(words) == 0:
                        continue
                    if label == "other":
                        for w in words:
                            tokens.append(w)
                            block_ids.append(block_idx)
                            ner_tags.append("O")
                            bboxes.append(norm_tbox)
                    else:
                        tokens.append(words[0])
                        block_ids.append(block_idx)
                        ner_tags.append("B-" + label.upper())
                        bboxes.append(norm_tbox)
                        for w in words[1:]:
                            tokens.append(w)
                            block_ids.append(block_idx)
                            ner_tags.append("I-" + label.upper())
                            bboxes.append(norm_tbox)
                    block_idx += 1

                yield {"id": doc_idx, "tokens": tokens,"bboxes": bboxes, "ner_tags": ner_tags,
                    "block_ids": block_ids, "image": image}
            # one_page_info = {'tokens': [], 'tboxes': [], 'bboxes': [], 'block_ids':[], 'image': image_path}


    def get_processed_for_bertx(self,ds):
        def _preprocess(sample):
            token_boxes = []
            token_labels = []
            for word, box, ner in zip(sample['tokens'], sample['bboxes'], sample['ner_tags']):
                word_tokens = self.tokenizer.tokenize(word)
                token_boxes.extend([box] * len(word_tokens))
                token_labels.extend([ner]+ [-100]*(len(word_tokens) - 1))
            # add bounding boxes of cls + sep tokens
            token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]*(self.opt.max_seq_len-len(token_boxes)-1)
            token_labels = [-100] + token_labels + [-100] * (self.opt.max_seq_len-len(token_labels)-1)
            # == wrap the sample into: {input_ids, token_type_ids, attention_mask, bbox, labels} ==
            # 1) tokenize, here, you must join the 'tokens' as one 'text' in a conventional way
            encodings = self.tokenizer(text=" ".join(sample['tokens']), return_token_type_ids = True,
                max_length=self.opt.max_seq_len, padding="max_length", truncation=True)
            # 2) add extended bbox and labels
            encodings['bbox'] = token_boxes[:self.opt.max_seq_len]
            encodings['labels'] = token_labels[:self.opt.max_seq_len]
            return encodings

        features = Features({
            'input_ids': Sequence(feature=Value(dtype='int64')),
            # 'position_ids': Sequence(feature=Value(dtype='int64')),
            'token_type_ids':Sequence(feature=Value(dtype='int64')),
            'attention_mask': Sequence(Value(dtype='int64')),
            'bbox': Array2D(dtype="int64", shape=(512, 4)),
            'labels': Sequence(feature=Value(dtype='int64')),
        })
        processed_ds = ds.map(_preprocess, num_proc=os.cpu_count(), 
            remove_columns=ds.column_names, features=features).with_format("torch")
        return processed_ds


    def get_preprocessed_ds(self,ds):
        # old BERT style (different vocab and tokenization)
        if self.opt.network_type in ['bert','layoutlmv1']:
            return self.get_processed_for_bertx(ds)

        def _preprocess(batch):
            # 1) encode words and imgs
            if self.opt.network_type == 'layoutlmv2': 
                return_type = True
            else:
                return_type = False
            encodings = self.processor(images=batch['image'],text=batch['tokens'], boxes=batch['bboxes'],return_token_type_ids = return_type,
                                       word_labels=batch['ner_tags'], truncation=True, padding='max_length', max_length=self.opt.max_seq_len)
            # 2) add relative position_ids
            if self.opt.network_type != 'layoutlmv2':
                position_ids = []
                for i, block_ids in enumerate(batch['block_ids']):
                    word_ids = encodings.word_ids(i)
                    rel_pos = self._get_rel_pos(word_ids, block_ids)
                    position_ids.append(rel_pos)
                encodings['position_ids'] = position_ids

            # 3) add spatial attention
            # spatial_matrix = []
            # for i, bb in enumerate(encodings['bbox']):
            #     word_ids = encodings.word_ids(i)
            #     sm = myds_util._fully_spatial_matrix(bb, word_ids)
            #     spatial_matrix.append(sm)
            # encodings['spatial_matrix'] = spatial_matrix

            return encodings

        if self.opt.network_type == 'layoutlmv2':
            features = Features({
                'image' : Array3D(dtype="float32", shape=(3, 224, 224)),
                'input_ids': Sequence(feature=Value(dtype='int64')),
                'token_type_ids':Sequence(feature=Value(dtype='int64')),
                'attention_mask': Sequence(Value(dtype='int64')),
                'bbox': Array2D(dtype="int64", shape=(512, 4)),
                'labels': Sequence(feature=Value(dtype='int64')),
            })
        else:
            features = Features({
                'pixel_values': Array3D(dtype="float32", shape=(3, 224, 224)),
                'input_ids': Sequence(feature=Value(dtype='int64')),
                'position_ids': Sequence(feature=Value(dtype='int64')),
                'attention_mask': Sequence(Value(dtype='int64')),
                'bbox': Array2D(dtype="int64", shape=(512, 4)),
                # 'spatial_matrix': Array3D(dtype='float32', shape=(512, 512, 11)),     # 
                'labels': Sequence(feature=Value(dtype='int64')),
            })
        # processed_ds = ds.map(_preprocess, batched=True, num_proc=self.cpu_num, 
        #     remove_columns=['id','tokens', 'bboxes','ner_tags','block_ids','image'], features=features).with_format("torch")
        processed_ds = ds.map(_preprocess, batched=True, num_proc=os.cpu_count(), 
            remove_columns=ds.column_names, features=features,batch_size=100).with_format("torch")
    
        # process to: 'input_ids', 'position_ids','attention_mask', 'bbox', 'labels', 'pixel_values']
        return processed_ds


    def get_label_ds(self,ds):
        def map_label2id(sample):
            sample['ner_tags'] = [self.class_label.str2int(ner_label) for ner_label in sample['ner_tags']]
            bboxes = sample['bboxes']
            # for bbox in bboxes:
            #     if bbox[0]>bbox[2] or bbox[1]>bbox[3]:
            #         print('=====neg=====')
            #         print(bbox)
            #         print('==========')
            #         # input(bbox)
            #         sample['bboxes'] = [bbox[0],bbox[1],bbox[0]+1,bbox[1]+1]
            #     else:
            #         if bbox[2]-bbox[0]>127 or bbox[3]-bbox[1]>127:
            #             print('=====BIG=====')
            #             print(bbox)
            #             print('=====s=====')
            return sample
        label_ds = ds.map(map_label2id, num_proc=os.cpu_count())
        return label_ds


    def _get_label_list(self,labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        label_list = list(unique_labels)
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

