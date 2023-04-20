
from datasets import load_from_disk, Features, Sequence, Value, Array2D, Array3D
from transformers import LayoutLMv3TokenizerFast, AutoTokenizer, AutoProcessor, AutoConfig, LayoutLMv3FeatureExtractor
from PIL import Image
from datasets import Dataset, concatenate_datasets
import transformers
from mydataset import myds_util


class RVLCDIP:
    def __init__(self,opt):
        self.opt = opt
        self.config = AutoConfig.from_pretrained(opt.layoutlm_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(opt.layoutlm_dir)
        assert isinstance(self.tokenizer, transformers.PreTrainedTokenizerFast) # get sub
        self.processor = AutoProcessor.from_pretrained(opt.layoutlm_dir,tokenizer=self.tokenizer, apply_ocr=False) 
        self.cpu_num = opt.num_cpu
        # four maps
        dataset_list = []
        for i in range(1):
            ds_path = '/home/ubuntu/air/vrdu/datasets/rvl_HF_datasets/weighted_rvl'+str(i)+'_dataset.hf'
            self.raw_ds = self.get_raw_ds(ds_path)
            self.processed_ds = self.get_preprocessed_ds(self.raw_ds)
            temp_ds = self.get_label_define_features(self.processed_ds)
            dataset_list.append(temp_ds)
        self.trainable_ds = concatenate_datasets(dataset_list)

    # load raw dataset (including image object)
    def get_raw_ds(self, ds_path):
        def _load_imgs_obj(sample):
            # 1) load img obj
            sample['images'],size = self._load_image(sample['image'])
            # 2) normalize bboxes using the img size 
            sample['bboxes'] = [self._normalize_bbox(bbox, size) for bbox in sample['bboxes']]
            return sample

        # 1 load raw data
        raw_ds = load_from_disk(ds_path) # {'tokens': [], 'tboxes': [], 'bboxes': [], 'block_ids':[], 'image': image_path}
        # raw_ds = Dataset.from_dict(raw_ds[:500])    # obtain subset for experiment/debugging use
        # 2 load img obj and norm bboxes 
        ds = raw_ds.map(_load_imgs_obj, num_proc=self.cpu_num, remove_columns=['tboxes']) # load image objects

        return ds

    # overall preprocessing
    def get_preprocessed_ds(self,ds):
        def _preprocess(batch):
            # 1) encode words and imgs
            encodings = self.processor(images=batch['images'],text=batch['tokens'], boxes=batch['bboxes'],truncation=True, padding='max_length', max_length=self.opt.max_seq_len)
            # 2) add position_ids
            position_ids = []
            for i, block_ids in enumerate(batch['block_ids']):
                word_ids = encodings.word_ids(i)
                rel_pos = self._get_rel_pos(word_ids, block_ids)
                position_ids.append(rel_pos)
            encodings['position_ids'] = position_ids
            # 3) add spatial attention
            spatial_matrix = []
            for i, bb in enumerate(encodings['bbox']):
                word_ids = encodings.word_ids(i)
                sm = myds_util._fully_spatial_matrix(bb, word_ids)
                spatial_matrix.append(sm)
            encodings['spatial_matrix'] = spatial_matrix

            return encodings

        processed_ds = ds.map(_preprocess,
            batched=True, num_proc=self.cpu_num, remove_columns=['tokens', 'bboxes','block_ids','images','image'])
        # process to: 'input_ids', 'position_ids','attention_mask', 'bbox', 'pixel_values']
        return processed_ds


    def get_label_define_features(self, ds):
        features = Features({
                'pixel_values': Array3D(dtype="float32", shape=(3, 224, 224)),
                'input_ids': Sequence(feature=Value(dtype='int64')),
                'position_ids': Sequence(feature=Value(dtype='int64')),
                'attention_mask': Sequence(Value(dtype='int64')),
                'bbox': Array2D(dtype="int64", shape=(512, 4)),
                'spatial_matrix': Array3D(dtype='float32', shape=(512, 512, 11)),     # 
                'labels': Sequence(feature=Value(dtype='int64')),
                })
        trainable_ds = ds.map(lambda example: {"labels": example['input_ids'].copy()}, num_proc=self.cpu_num,
            features = features).with_format("torch")

        return trainable_ds

    def _load_image(self,image_path):
        image = Image.open(image_path).convert("RGB")
        w, h = image.size
        return image, (w, h)

    def _normalize_bbox(self, bbox, size):
        return [
            int(1000 * bbox[0] / size[0]),
            int(1000 * bbox[1] / size[1]),
            int(1000 * bbox[2] / size[0]),
            int(1000 * bbox[3] / size[1]),
        ]
    
    # mimic the way of automating position_ids in layoutlmv3
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
                    # set back to 2; 
                    rel_cnt = self.config.pad_token_id+1
                    res.append(rel_cnt) # operate
                    # reset prev_block
                    prev_block = curr_block
                else:
                    res.append(rel_cnt)
            rel_cnt+=1
        return res
