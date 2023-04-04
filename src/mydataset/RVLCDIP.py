
from datasets import load_from_disk
from transformers import LayoutLMv3TokenizerFast, AutoTokenizer, LayoutLMv3FeatureExtractor
from PIL import Image

class RVLCDIP:
    def __init__(self,opt):
        self.opt = opt

        self.tokenizer = LayoutLMv3TokenizerFast.from_pretrained(opt.spatial_lm_dir)
        self.processor = AutoProcessor.from_pretrained(opt.layoutlm_dir, tokenizer=self.tokenizer, apply_ocr=False) 

        # four maps
        self.raw_ds = self.get_raw_ds(opt)
        self.processed_ds = self.get_preprocessed_ds(self.raw_ds)
        self.rp_ds = self.get_rp(self.processed_ds)
        self.trainable_ds = self.get_label_define_features(self.rp_ds)

    # load raw dataset (including image object)
    def get_raw_ds(opt):
        def _load_imgs_obj(self,sample):
            sample['image'],_ = self.load_image(sample['image_path'])
            return sample
        # 1) load raw data
        raw_ds = load_from_disk(opt.rvl_cdip_ds) # {'tokens': [], 'tboxes': [], 'bboxes': [], 'block_ids':[], 'image': image_path}
        # 2) load img obj
        ds = raw_ds.map(_load_imgs_obj, num_proc=4, remove_columns=['image_path'])
        return ds


    def get_preprocessed_ds(self,ds):
        processed_ds = ds.map(lambda batch: self.processor(image=batch['image'],text=batch['tokens'], boxes=batch['bboxes'],truncation=True, padding='max_length'),
            batched=True, num_proc=4, remove_columns=['tokens', 'bboxes'])
        # process to: 'input_ids', 'attention_mask', 'bbox', 'pixel_values']
        return processed_ds

    def _add_relative_position(self,ds):
        def get_rel_pos(word_ids, block_ids):   # [None, 0, 1, 2, 2, 3, None]; [1,1,2,2] which is dict {word_idx: block_num}
            res = []
            rel_cnt = 0
            prev_block = 1
            for word_id in word_ids:
                if word_id is None:
                    res.append(rel_cnt)
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
        # add position_ids
        position_ids = []
        for i, block_ids in enumerate(ds['block_ids']):
            word_ids = ds.word_ids(i)
            rel_pos = get_rel_pos(word_ids, block_ids)
            position_ids.append(rel_pos)
        ds['position_ids'] = position_ids
        return ds

    # get relative dataset, and make it trainable
    def get_RP_ds(self,ds):
        rel_pos_ds = ds.map(self._add_relative_position, 
            num_proc=4, batched=True, remove_columns=['block_ids'], 
        return res_pos_ds

    def get_label_define_features(self, ds):
        features = Features({
                'pixel_values': Array3D(dtype="float32", shape=(3, 224, 224)),
                'input_ids': Sequence(feature=Value(dtype='int64')),
                'position_ids': Sequence(feature=Value(dtype='int64')),
                'attention_mask': Sequence(Value(dtype='int64')),
                'bbox': Array2D(dtype="int64", shape=(512, 4)),
                'labels': Sequence(feature=Value(dtype='int64')),}
        )
        trainable_ds = ds.map(lambda example: {'labels': example['input_ids'].copy()), num_proc=4,
            features = features).with_format("torch")
        return trainable_ds

    def _load_image(self,image_path):
        image = Image.open(image_path).convert("RGB")
        w, h = image.size
        return image, (w, h)

