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


# Step1. get image path list
def get_file_list():  # .tif files
    res = []
    train = '/home/ubuntu/air/vrdu/datasets/labels/train.txt'
    # val = '/home/ubuntu/air/vrdu/datasets/labels/val.txt'
    # train = '/home/ubuntu/air/vrdu/datasets/labels/test.txt'
    with open(train, 'r', encoding='utf8') as fr:
        data = fr.readlines()
    for row in data:
        path = row.split(' ')[0]
        image_path = os.path.join(images, path)
        res.append(image_path)
    return res
# step1. get image path list from a folder
def get_imgs_list(dir):
    files = os.path.listdir(dir)
    return files

# step2: shuffle and split into 5 subsets
def split(input_list, n):
    k, m = divmod(len(input_list), n)
    return (input_list[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


if __name__ == "__main__":
    files = get_file_list()

    random.Random(88).shuffle(files)

    
    subsets = list(split(files, 10))
    print('dataset num: ',len(subsets))
    for i,subset in enumerate(subsets):
        print('dataset ',i,', sample size:',len(subset))
        mydataset = tesseract4img.imgs_to_dataset_generator(subset)

        saveto = '/home/ubuntu/air/vrdu/datasets/rvl_HF_datasets/rvl' + str(i) + '_dataset.hf'
        mydataset.save_to_disk(saveto)

        print(mydataset)
    
    # for i in range(7, len(subsets)):
    #     subset = subsets[i]
    #     print(len(subset))
    #     saveto = generate_and_save(i, subset)
    #     # print('saved to:', saveto)