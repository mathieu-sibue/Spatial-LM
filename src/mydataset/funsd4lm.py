from datasets import load_dataset ,Features, Sequence, Value, Array2D, Array3D, Dataset
from datasets.features import ClassLabel
from transformers import AutoProcessor, AutoTokenizer
import os
import json
import transformers

class FUNSD:
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
        self.tokenizer = AutoTokenizer.from_pretrained(opt.layoutlm_dir)
        assert isinstance(self.tokenizer, transformers.PreTrainedTokenizerFast)  # get sub
        self.processor = AutoProcessor.from_pretrained(opt.layoutlm_dir,tokenizer=self.tokenizer, apply_ocr=False)    # wrap of featureExtract & tokenizer

        self.image_col_name = "image"
        self.text_col_name = "tokens"
        self.boxes_col_name = "bboxes"
        self.label_col_name = "ner_tags"
        # load data
        raw_train, raw_test = self.get_raw_ds()
        opt.id2label, opt.label2id, opt.label_list = self._get_label_map(raw_train)
        opt.num_labels = len(opt.label_list)

        print(raw_train)
        print(opt.id2label)
        print(opt.label2id)
        print(opt.label_list)

        # prepare for getting trainable data
        # 6 labels: {0: 'O', 1: 'B-HEADER', 2: 'I-HEADER', 3: 'B-QUESTION', 4: 'I-QUESTION', 5: 'B-ANSWER', 6: 'I-ANSWER'}
        # processed_train_ds, processed_test_ds = self.get_preprocessed_ds(raw_train),self.get_preprocessed_ds(raw_test)
        # trainable_train_ds, trainable_test_ds = self.get_defined_ds(processed_train_ds), self.get_defined_ds(processed_test_ds)

    def get_raw_ds(self):
        train = Dataset.from_generator(self.ds_generator, gen_kwargs={'base_dir':self.opt.funsd_train})
        test = Dataset.from_generator(self.ds_generator, gen_kwargs={'base_dir':self.opt.funsd_test})
        return train, test

    def ds_generator(self, base_dir):
        # logger.info("⏳ Generating examples from = %s", filepath)
        ann_dir = os.path.join(base_dir, "adjusted_annotations")
        img_dir = os.path.join(base_dir, "images")
        for doc_idx, file in enumerate(sorted(os.listdir(ann_dir))):
            # print('---doc id:---',doc_idx)
            tokens = []
            bboxes = []
            ner_tags = []
            block_ids = []

            file_path = os.path.join(ann_dir, file)
            with open(file_path, "r", encoding="utf8") as f:
                data = json.load(f)
            image_path = os.path.join(img_dir, file)
            image_path = image_path.replace("json", "png")
            image, size = self._load_image(image_path)
            block_id = 1
            for block in data["form"]:
                cur_line_bboxes = []
                words, label = block["words"], block["label"]
                text = block['text']
                if text.strip() == '': continue
                words = [w for w in words if w["text"].strip() != ""]
                if len(words) == 0:
                    continue
                if label == "other":
                    for w in words:
                        tokens.append(w["text"])
                        block_ids.append(block_id)
                        ner_tags.append("O")
                        cur_line_bboxes.append(self._normalize_bbox(w["box"], size))
                else:
                    tokens.append(words[0]["text"])
                    block_ids.append(block_id)
                    ner_tags.append("B-" + label.upper())
                    cur_line_bboxes.append(self._normalize_bbox(words[0]["box"], size))
                    for w in words[1:]:
                        tokens.append(w["text"])
                        block_ids.append(block_id)
                        ner_tags.append("I-" + label.upper())
                        cur_line_bboxes.append(self._normalize_bbox(w["box"], size))
                cur_line_bboxes = self._get_line_bbox(cur_line_bboxes)  # shared boxes, token-wise
                bboxes.extend(cur_line_bboxes)
                block_id += 1

            yield {"id": doc_idx, "tokens": tokens, "bboxes": bboxes, "ner_tags": ner_tags,
                   "block_ids": block_id, "image": image}
        # one_page_info = {'tokens': [], 'tboxes': [], 'bboxes': [], 'block_ids':[], 'image': image_path}


    def get_preprocessed_ds(self,ds):
        def _preprocess(batch):
            # 1) encode words and imgs
            encodings = self.processor(images=batch['image'],text=batch['tokens'], boxes=batch['bboxes'],
                                       word_labels=batch['ner_tags'], truncation=True, padding='max_length', max_length=self.opt.max_seq_len)
            # 2) add position_ids
            position_ids = []
            for i, block_ids in enumerate(batch['block_ids']):
                word_ids = encodings.word_ids(i)
                rel_pos = self._get_rel_pos(word_ids, block_ids)
                position_ids.append(rel_pos)
            encodings['position_ids'] = position_ids
            return encodings

        processed_ds = ds.map(_preprocess,
            batched=True, num_proc=self.cpu_num, remove_columns=['tokens', 'bboxes','block_ids','image'])
        # process to: 'input_ids', 'position_ids','attention_mask', 'bbox', 'pixel_values']
        return processed_ds


    def get_defined_ds(self,ds):
        # return self.prepare_one_doc(self.dataset[split])
        features = Features({
            'pixel_values': Array3D(dtype="float32", shape=(3, 224, 224)),
            'input_ids': Sequence(feature=Value(dtype='int64')),
            'position_ids': Sequence(feature=Value(dtype='int64')),
            'attention_mask': Sequence(Value(dtype='int64')),
            'bbox': Array2D(dtype="int64", shape=(512, 4)),
            'labels': Sequence(feature=Value(dtype='int64')),
        })
        trainable_ds = ds.map(num_proc=self.cpu_num,
                              batched=True, features=features).with_format("torch")
        # trainable_samples = trainable_samples.set_format("torch")  # with_format is important
        return trainable_ds


    def get_label_list(self,labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        label_list = list(unique_labels)
        label_list.sort()
        return label_list

    def get_label_map(self,ds):
        features = ds.features
        column_names = ds.column_names
        
        if isinstance(features[self.label_col_name].feature, ClassLabel):
            label_list = features[self.label_col_name].feature.names
            # No need to convert the labels since they are already ints.
            id2label = {k: v for k,v in enumerate(label_list)}
            label2id = {v: k for k,v in enumerate(label_list)}
        else:
            label_list = self.get_label_list(ds[self.label_col_name])
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

