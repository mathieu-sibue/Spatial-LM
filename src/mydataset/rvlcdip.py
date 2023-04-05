
from datasets import load_from_disk, Features, Sequence, Value, Array2D, Array3D
from transformers import LayoutLMv3TokenizerFast, AutoTokenizer, AutoProcessor, LayoutLMv3FeatureExtractor
from PIL import Image
from datasets import Dataset
import transformers

class RVLCDIP:
    def __init__(self,opt):
        self.opt = opt

        self.tokenizer = AutoTokenizer.from_pretrained(opt.spatial_lm_dir)
        assert isinstance(self.tokenizer, transformers.PreTrainedTokenizerFast) # get sub
        self.processor = AutoProcessor.from_pretrained(opt.spatial_lm_dir, tokenizer=self.tokenizer, apply_ocr=False) 

        # four maps
        self.raw_ds = self.get_raw_ds(opt)
        self.processed_ds = self.get_preprocessed_ds(self.raw_ds)
        self.trainable_ds = self.get_label_define_features(self.processed_ds)


    # load raw dataset (including image object)
    def get_raw_ds(self, opt):
        def _load_imgs_obj(sample):
            sample['images'],_ = self._load_image(sample['image'])
            return sample
        # 1) load raw data
        raw_ds = load_from_disk(opt.rvl_cdip_ds) # {'tokens': [], 'tboxes': [], 'bboxes': [], 'block_ids':[], 'image': image_path}
        # raw_ds = Dataset.from_dict(raw_ds[:100])    # obtain subset for experiment/debugging use
        # 2) load img obj
        ds = raw_ds.map(_load_imgs_obj, num_proc=8, remove_columns=['tboxes']) # load image objects
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
            return encodings

        processed_ds = ds.map(_preprocess,
            batched=True, num_proc=8, remove_columns=['tokens', 'bboxes','block_ids','images','image'])
        # process to: 'input_ids', 'position_ids','attention_mask', 'bbox', 'pixel_values']
        return processed_ds

    def _add_relative_position(self,ds):
        # add position_ids
        position_ids = []
        for i, block_ids in enumerate(ds['block_ids']):
            word_ids = ds.word_ids(i)
            rel_pos = get_rel_pos(word_ids, block_ids)
            position_ids.append(rel_pos)
        ds['position_ids'] = position_ids
        return ds


    def get_label_define_features(self, ds):
        features = Features({
                'pixel_values': Array3D(dtype="float32", shape=(3, 224, 224)),
                'input_ids': Sequence(feature=Value(dtype='int64')),
                'position_ids': Sequence(feature=Value(dtype='int64')),
                'attention_mask': Sequence(Value(dtype='int64')),
                'bbox': Array2D(dtype="int64", shape=(512, 4)),
                'labels': Sequence(feature=Value(dtype='int64')),}
        )
        trainable_ds = ds.map(lambda example: {"labels": example['input_ids'].copy()}, num_proc=8,
            features = features).with_format("torch")
        return trainable_ds

    def _load_image(self,image_path):
        image = Image.open(image_path).convert("RGB")
        w, h = image.size
        return image, (w, h)

    def _get_rel_pos(self,word_ids, block_ids):   # [None, 0, 1, 2, 2, 3, None]; [1,1,2,2] which is dict {word_idx: block_num}
        res = []
        rel_cnt = 0
        prev_block = 1
        for word_id in word_ids:
            if word_id is None:
                res.append(0)
            else:
                curr_block = block_ids[word_id]   # word_id is the 0,1,2,3,.. word index;
                if curr_block != prev_block:
                    # set back to 0; 
                    rel_cnt = 0
                    res.append(rel_cnt) # operate
                    # reset prev_block
                    prev_block = curr_block
                else:
                    res.append(rel_cnt)
            rel_cnt+=1
        return res