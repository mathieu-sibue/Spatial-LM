# coding=utf-8
import json
import os
from pathlib import Path
import datasets
from PIL import Image
# import torch
# from detectron2.data.transforms import ResizeTransform, TransformList
logger = datasets.logging.get_logger(__name__)
_CITATION = """\
@article{2019,
   title={ICDAR2019 Competition on Scanned Receipt OCR and Information Extraction},
   url={http://dx.doi.org/10.1109/ICDAR.2019.00244},
   DOI={10.1109/icdar.2019.00244},
   journal={2019 International Conference on Document Analysis and Recognition (ICDAR)},
   publisher={IEEE},
   author={Huang, Zheng and Chen, Kai and He, Jianhua and Bai, Xiang and Karatzas, Dimosthenis and Lu, Shijian and Jawahar, C. V.},
   year={2019},
   month={Sep}
}
"""
_DESCRIPTION = """\
https://arxiv.org/abs/2103.10213
"""


def load_image(image_path):
    image = Image.open(image_path)
    w, h = image.size
    return image, (w, h)
def normalize_bbox(bbox, size):
    return [
        int(1000 * bbox[0] / size[0]),
        int(1000 * bbox[1] / size[1]),
        int(1000 * bbox[2] / size[0]),
        int(1000 * bbox[3] / size[1]),
    ]

def _get_drive_url(url):
    base_url = 'https://drive.google.com/uc?id='
    split_url = url.split('/')
    return base_url + split_url[5]
_URLS = [
    _get_drive_url("https://drive.google.com/file/d/1ZyxAw1d-9UvhgNLGRvsJK4gBCMf0VpGD/view?usp=sharing"),
]
class SroieConfig(datasets.BuilderConfig):
    """BuilderConfig for SROIE"""
    def __init__(self, **kwargs):
        """BuilderConfig for SROIE.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(SroieConfig, self).__init__(**kwargs)
class Sroie(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        SroieConfig(name="sroie", version=datasets.Version("1.0.0"), description="SROIE dataset"),
    ]
    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "words": datasets.Sequence(datasets.Value("string")),
                    "bboxes": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=["O","B-COMPANY", "I-COMPANY", "B-DATE", "I-DATE", "B-ADDRESS", "I-ADDRESS", "B-TOTAL", "I-TOTAL"]
                        )
                    ),
                    #"image": datasets.Array3D(shape=(3, 224, 224), dtype="uint8"),
                    "image_path": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
            citation=_CITATION,
            homepage="https://arxiv.org/abs/2103.10213",
        )
    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        """Uses local files located with data_dir"""
        downloaded_file = dl_manager.download_and_extract(_URLS)
        # move files from the second URL together with files from the first one.
        dest = Path(downloaded_file[0])/"sroie"
        
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"filepath": dest/"train"}
            ),            
            datasets.SplitGenerator(
                name=datasets.Split.TEST, gen_kwargs={"filepath": dest/"test"}
            ),
        ]
    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        ann_dir = os.path.join(filepath, "tagged")
        img_dir = os.path.join(filepath, "images")
        for guid, fname in enumerate(sorted(os.listdir(img_dir))):
            name, ext = os.path.splitext(fname)
            file_path = os.path.join(ann_dir, name + ".json")
            with open(file_path, "r", encoding="utf8") as f:
                data = json.load(f)
            image_path = os.path.join(img_dir, fname)
            
            image, size = load_image(image_path)
            
            boxes = [normalize_bbox(box, size) for box in data["bbox"]]


            yield guid, {"id": str(guid), "words": data["words"], "bboxes": boxes, "ner_tags": data["labels"], "image_path": image_path}

