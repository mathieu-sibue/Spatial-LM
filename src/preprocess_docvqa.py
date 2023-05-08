import os


def get_imgs_dfs(dir, suffix = 'png'):
    res = []
    files = os.listdir(dir)
    for file in files:
        file_path = os.path.join(dir,file)

        if os.path.isdir(file_path):
            sub_res = get_imgs_dfs(file_path, suffix)
            res += sub_res
        elif file.endswith(suffix):
            res.append(file_path)
    return res


def get_question_pairs(base,split='val'):
    # from json of questions and answers
    file_path = os.path.join(base, split+'_v1.0.json')
    
    with open(file_path) as fr:
        data = json.load(fr)
    id2trip = {}
    for sample in data['data']:
        qID = sample['questionId']  # numeric e.g., 8366
        question = sample['question']
        # for test set, there is no answersr
        answers = []
        if 'answers' in sample.keys():
            answers = sample['answers']

        ucsf_doc_id = sample['ucsf_document_id']   # e.g.,: txpp0227
        ucsf_doc_page = sample['ucsf_document_page_no'] # e.g.,: 10
        docID_page = ucsf_doc_id + '_' + ucsf_doc_page
        trip_object = (docID_page, question, answers)
        id2trip[qID] = trip_object
    return id2trip

def generate_docvqa_ds():
    dir = '/home/ubuntu/air/vrdu/datasets/docvqa'

    for split in ['train','val','test']:
        files,labels = get_img_label_pairs(split)
        print(split, ' to be generated file num:',len(files))

        img_dir = os.path.join(dir,split, 'documents')
        img_paths = get_img_dfs(img_dir)

        mydataset = tesseract4img.imgs_to_dataset_generator(sub_files,sub_labels)
        saveto = '/home/ubuntu/air/vrdu/datasets/rvl_HF_datasets/full_rvl_'+split+str(i)+'_dataset.hf'
        mydataset.save_to_disk(saveto)
        print(mydataset)


if __name__=='__main__':
    # load qa pairs 
    split = 'val'

    # 1 load all QA pairs 
    id2trip = get_question_pairs(base,split)

    # 2 parse imgs and generate ds

    # 3 merge the ds into datasets; -> merge train and val

