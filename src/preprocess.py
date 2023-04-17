import datasets
from datasets import Dataset, Features, Sequence, Value, Array2D, Array3D
import os
import random
from OCRs import tesseract4img



'''
what I wanna do:
    1 Load all images 
    2 Generate dataset dict as dataset file
    3 load them for pretraining 
'''


# step1: get all images from a folder
images = '/home/ubuntu/air/vrdu/datasets/images'


'''

SROIE dataset (receipt);


0 letter; 15
1 form; 10
2 email; 2
3 handwritten; 1
4 advertisement; 2.5
5, scientific report; 4.5
6 scientific publication; 2
7 specification; 5
8 file folder; 1
9 news article; 1.5
10 budget; 7
11 invoice, 9.5
12 presentation (slide): 2
13 questionaire; 7
14 resume; 3
15 memo; 1.5


'''
tgt_prob = {
    '0':0.15,
    '1':1.0,
    '2':0.15,
    '3':0.15,
    '4':0.25,
    '5':0.45,
    '6':0.2,
    '7':0.55,
    '8':0.1,
    '9':0.15,
    '10':0.7,
    '11':0.95,
    '12':0.15,
    '13':0.7,
    '14':0.3,
    '15':0.15,
}

# sampling some data proportionally to the the ratio above;
def get_ratioly_sampled(imgs,labels):
    sub_imgs, sub_labels = [],[]
    for img,label in zip(imgs,labels):
        prob = random.random()
        if prob > tgt_prob[label]: continue
        sub_imgs.append(img)
        sub_labels.append(label)
    return sub_imgs, sub_labels


# Step1. get image path list
def get_img_label_pairs(split='train'):  # .tif files
    imgs = []
    labels = []
    if split=='train':
        split_file = '/home/ubuntu/air/vrdu/datasets/labels/train.txt'
    elif split=='val':
        split_file = '/home/ubuntu/air/vrdu/datasets/labels/val.txt'
    elif split=='test':
        split_file = '/home/ubuntu/air/vrdu/datasets/labels/test.txt'
    else:
        print('wrong split:',split)
        return

    with open(split_file, 'r', encoding='utf8') as fr:
        data = fr.readlines()
    for row in data:
        path, label = row.split(' ')
        label = label.strip()
        labels.append(label)
        image_path = os.path.join(images, path)
        imgs.append(image_path)
    return imgs,labels


# step1. get image path list from a folder
funsd_plus_val = '/home/ubuntu/air/vrdu/datasets/funsd_plus/val_data/images'
funsd_plus_train = '/home/ubuntu/air/vrdu/datasets/funsd_plus/train_data/images'
funsd_plus_test = '/home/ubuntu/air/vrdu/datasets/funsd_plus/test_data/images'

cord_train = '/home/ubuntu/air/vrdu/datasets/CORD/train/image'
cord_test = '/home/ubuntu/air/vrdu/datasets/CORD/test/image'
cord_dev = '/home/ubuntu/air/vrdu/datasets/CORD/dev/image'

sorie_train = '/home/ubuntu/air/vrdu/datasets/sorie2019/train/img'
sorie_test = '/home/ubuntu/air/vrdu/datasets/sorie2019/test/img'


def get_imgs_list(dir):
    files = os.listdir(dir)
    files = [os.path.join(dir,file) for file in files]
    return files

# step2: shuffle and split into 5 subsets
def _split(input_list, n):
    k, m = divmod(len(input_list), n)
    return (input_list[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


# load file list
# if __name__ == "__main__":
#     # 1 get file_list
#     all_files = []
#     for dir in [funsd_plus_val,funsd_plus_train,funsd_plus_test, cord_train,
#         cord_test, cord_dev, sorie_train, sorie_test]:
#
#         files = get_imgs_list(dir)
#         all_files += files
#     print('total files:',len(all_files))
#     random.Random(88).shuffle(all_files)    # shuffle
#
#     # 2 generate dataset for file_list
#     mydataset = tesseract4img.imgs_to_dataset_generator(all_files)
#
#
#     print(mydataset)
#     # 3 save dataset
#     saveto = '/home/ubuntu/air/vrdu/datasets/rvl_HF_datasets/funsd_cord_sorie_dataset.hf'
#     mydataset.save_to_disk(saveto)



if __name__ == "__main__":
    for split in ['train','test','val']:
        files,labels = get_img_label_pairs(split)
        mydataset = tesseract4img.imgs_to_dataset_generator(files,labels)
        saveto = '/home/ubuntu/air/vrdu/datasets/rvl_HF_datasets/full_rvl_'+split+'_dataset.hf'
        mydataset.save_to_disk(saveto)

        print(mydataset)
