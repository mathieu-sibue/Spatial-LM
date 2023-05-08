import os
import json
from OCRs import tesseract4img

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
    file_path = os.path.join(base, split, split+'_v1.0.json')
    
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
    split = 'val'   # train, test, val
    base = '/home/ubuntu/air/vrdu/datasets/docvqa'
    # 1 load all QA pairs 
    id2trip = get_question_pairs(base,split)    

    cnt = 0

    img_paths = []
    ans_list = []
    q_list = []
    QA_pair_list=[]

    for k,val in id2trip.items():
        docID_page, question, answers = val
        img_path = os.path.join(base, split, 'documents', docID_page +'.png')
        
        ans_list.append(answers)
        q_list.append(question)
        img_paths.append(img_path)
        QA_pair_list.append((question,answers))
        # print(question,answers)

        if not question: continue

    # 2 parse imgs and generate ds
    ds = tesseract4img.imgs_to_dataset_generator(img_paths,labels=None, tesseract_wait=True, questions = q_list, answers = ans_list)
    print(ds)

    # 3 output
    ds.save_to_disk('val.hf')

