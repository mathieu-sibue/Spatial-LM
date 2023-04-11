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

# Step1. get image path list
def get_file_list():  # .tif files
    res = []
    train = '/home/ubuntu/air/vrdu/datasets/labels/train.txt'
    # val = '/home/ubuntu/air/vrdu/datasets/labels/val.txt'
    # train = '/home/ubuntu/air/vrdu/datasets/labels/test.txt'
    with open(train, 'r', encoding='utf8') as fr:
        data = fr.readlines()
    for row in data:
        path, label = row.split(' ')
        label = label.strip()
        prob = random.random()
        # prob < tgt prob, will process
        if prob > tgt_prob[label]: continue
        
        image_path = os.path.join(images, path)
        res.append(image_path)
    return res
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
def split(input_list, n):
    k, m = divmod(len(input_list), n)
    return (input_list[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


# load file list
if __name__ == "__main__":
    # 1 get file_list
    all_files = []
    for dir in [funsd_plus_val,funsd_plus_train,funsd_plus_test, cord_train,
        cord_test, cord_dev, sorie_train, sorie_test]:

        files = get_imgs_list(dir)
        all_files += files
    print('total files:',len(all_files))
    random.Random(88).shuffle(all_files)    # shuffle

    # 2 generate dataset for file_list
    mydataset = tesseract4img.imgs_to_dataset_generator(all_files)


    print(mydataset)
    # 3 save dataset
    saveto = '/home/ubuntu/air/vrdu/datasets/rvl_HF_datasets/funsd_cord_sorie_dataset.hf'
    mydataset.save_to_disk(saveto)



# if __name__ == "__main__":
#     files = get_file_list()

#     random.Random(88).shuffle(files)

#     subsets = list(split(files, 10))
#     print('dataset num: ',len(subsets))
#     # for i,subset in enumerate(subsets):
#     for i in range(0,1):
#         subset = subsets[i]
#         print('dataset ',i,', sample size:',len(subset))
#         mydataset = tesseract4img.imgs_to_dataset_generator(subset)

#         saveto = '/home/ubuntu/air/vrdu/datasets/rvl_HF_datasets/weighted_rvl' + str(i) + '_dataset.hf'
#         mydataset.save_to_disk(saveto)

#         print(mydataset)
    
    # for i in range(7, len(subsets)):
    #     subset = subsets[i]
    #     print(len(subset))
    #     saveto = generate_and_save(i, subset)
    #     # print('saved to:', saveto)