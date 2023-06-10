# coding=utf-8
import json
import os
from pathlib import Path
import datasets
from PIL import Image
# import torch
# from detectron2.data.transforms import ResizeTransform, TransformList
from datasets import load_dataset ,Features, Sequence, Value, Array2D, Array3D, Dataset, DatasetDict
from transformers import AutoProcessor, AutoTokenizer, AutoConfig
import transformers
from datasets.features import ClassLabel
from mydataset import myds_util

class SROIE:

    def __init__(self,opt) -> None:    
        self.opt = opt

        # step1: create config, tokenizer, and processor
        self.config = AutoConfig.from_pretrained(opt.layoutlm_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(opt.layoutlm_dir)
        assert isinstance(self.tokenizer, transformers.PreTrainedTokenizerFast)  # get sub
        self.processor = AutoProcessor.from_pretrained(opt.layoutlm_dir,tokenizer=self.tokenizer, apply_ocr=False)    # wrap of featureExtract & tokenizer
        self.label_col_name = "ner_tags"
        raw_train, raw_test = self.get_raw_ds('/home/ubuntu/air/vrdu/datasets/sroie/')
        opt.id2label, opt.label2id, opt.label_list = self._get_label_map(raw_train)
        opt.num_labels = len(opt.label_list)

        train_label_ds, test_label_ds = self.get_label_ds(raw_train), self.get_label_ds(raw_test)
        trainable_train_ds, trainable_test_ds = self.get_preprocessed_ds(train_label_ds),self.get_preprocessed_ds(test_label_ds)
        self.trainable_ds = DatasetDict({
            "train" : trainable_train_ds.shuffle(seed=88) , 
            "test" : trainable_test_ds 
        })

    def get_raw_ds(self, base_dir):
        train_ds = Dataset.from_generator(self._generate_examples, gen_kwargs={'base_dir':os.path.join(base_dir,'train')})
        test_ds = Dataset.from_generator(self._generate_examples, gen_kwargs={'base_dir':os.path.join(base_dir,'test')})
        # test_ds = Dataset.from_generator(self._generate_examples(os.path.join(base_dir,'test')))
        # self._generate_examples(os.path.join(base_dir,'train'))
        return train_ds, test_ds

    def _generate_examples(self, base_dir):
        # logger.info("⏳ Generating examples from = %s", filepath)
        ann_dir = os.path.join(base_dir, "tagged")
        img_dir = os.path.join(base_dir, "images")
        for guid, fname in enumerate(sorted(os.listdir(img_dir))):
            name, ext = os.path.splitext(fname)
            file_path = os.path.join(ann_dir, name + ".json")
            with open(file_path, "r", encoding="utf8") as f:
                data = json.load(f)
            image_path = os.path.join(img_dir, fname)
            image, size = self.load_image(image_path)
            boxes = [self.normalize_bbox(box, size) for box in data["bbox"]]

            yield {"id": str(guid), "tokens": data["words"], "bboxes": boxes, "ner_tags": data["labels"], "image": image}

    def get_label_ds(self,ds):
        def map_label2id(sample):
            sample['ner_tags'] = [self.opt.label2id[ner_label] for ner_label in sample['ner_tags']]
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

            # 3) add spatial attention
            # spatial_matrix = []
            # for i, bb in enumerate(encodings['bbox']):
            #     word_ids = encodings.word_ids(i)
            #     sm = myds_util._fully_spatial_matrix(bb, word_ids)
            #     spatial_matrix.append(sm)
            # encodings['spatial_matrix'] = spatial_matrix

            return encodings


        features = Features({
            'pixel_values': Array3D(dtype="float32", shape=(3, 224, 224)),
            'input_ids': Sequence(feature=Value(dtype='int64')),
            # 'position_ids': Sequence(feature=Value(dtype='int64')),
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

    def load_image(self,image_path):
        image = Image.open(image_path).convert("RGB")
        w, h = image.size
        return image, (w, h)
    def normalize_bbox(self,bbox, size):
        return [
            int(1000 * bbox[0] / size[0]),
            int(1000 * bbox[1] / size[1]),
            int(1000 * bbox[2] / size[0]),
            int(1000 * bbox[3] / size[1]),
        ]
