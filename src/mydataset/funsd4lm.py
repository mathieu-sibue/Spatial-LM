from datasets import load_dataset ,Features, Sequence, Value, Array2D, Array3D
from datasets.features import ClassLabel
from transformers import AutoProcessor



class FUNSD:
    def __init__(self,opt) -> None:    
        self.opt = opt
        '''
        DatasetDict({
            train: Dataset({
                features: ['id', 'words', 'bboxes', 'ner_tags', 'image'],
                num_rows: 800
            })
            validation: Dataset({
                features: ['id', 'words', 'bboxes', 'ner_tags', 'image'],
                num_rows: 100
            })
            test: Dataset({
                features: ['id', 'words', 'bboxes', 'ner_tags', 'image'],
                num_rows: 100
            })
        })
        '''
        self.image_col_name = "image"
        self.text_col_name = "tokens"
        self.boxes_col_name = "bboxes"
        self.label_col_name = "ner_tags"
        # load data
        self.dataset = load_dataset("nielsr/funsd-layoutlmv3")
        print('--dataset:--',self.dataset)

        # prepare for getting trainable data
        # 6 labels: {0: 'O', 1: 'B-HEADER', 2: 'I-HEADER', 3: 'B-QUESTION', 4: 'I-QUESTION', 5: 'B-ANSWER', 6: 'I-ANSWER'}
        self.processor = AutoProcessor.from_pretrained('/home/ubuntu/air/vrdu/models/layoutlmv1.base', apply_ocr=False)    # wrap of featureExtract & tokenizer
        opt.id2label, opt.label2id, opt.label_list = self.get_label_map(self.dataset)
        opt.num_labels = len(opt.label_list)
        # process to: 'input_ids', 'attention_mask', 'bbox', 'labels', 'pixel_values']
        self.train_dataset, self.test_dataset = self.get_data('train'), self.get_data('test')



    def prepare_one_doc(self,doc):
        images = doc[self.image_col_name] ##if you use an image path, this will need to be updated to read the image in
        words = doc[self.text_col_name]
        boxes = doc[self.boxes_col_name]
        word_labels = doc[self.label_col_name]
        encoding = self.processor(images, words, boxes=boxes, word_labels=word_labels,
                            truncation=True, padding="max_length") # must put return tensor
        return encoding

    def get_data(self,split='train',shuffle=True):
        # return self.prepare_one_doc(self.dataset[split])
        samples = self.dataset[split]   # get split

        trainable_samples = samples.map(
            self.prepare_one_doc,
            batched=True,
            remove_columns=samples.column_names,    # remove original features
            features = Features({
                'pixel_values': Array3D(dtype="float32", shape=(3, 224, 224)),
                'input_ids': Sequence(feature=Value(dtype='int64')),
                'attention_mask': Sequence(Value(dtype='int64')),
                'bbox': Array2D(dtype="int64", shape=(512, 4)),
                'labels': Sequence(feature=Value(dtype='int64')),
            })
        ).with_format("torch")
        # trainable_samples = trainable_samples.set_format("torch")  # with_format is important
        return trainable_samples


    def get_label_list(self,labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        label_list = list(unique_labels)
        label_list.sort()
        return label_list

    def get_label_map(self,dataset,split='train'):
        features = dataset[split].features
        column_names = dataset[split].column_names
        
        if isinstance(features[self.label_col_name].feature, ClassLabel):
            label_list = features[self.label_col_name].feature.names
            # No need to convert the labels since they are already ints.
            id2label = {k: v for k,v in enumerate(label_list)}
            label2id = {v: k for k,v in enumerate(label_list)}
        else:
            label_list = self.get_label_list(dataset[split][self.label_col_name])
            id2label = {k: v for k,v in enumerate(label_list)}
            label2id = {v: k for k,v in enumerate(label_list)}
        return id2label, label2id, label_list



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

