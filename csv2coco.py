# python csv2coco.py --train=Sony_RX100m7_annotation_jpg_train.csv --test=Sony_RX100m7_annotation_jpg_test.csv --val=Sony_RX100m7_annotation_jpg_val.csv --camera=Sony_RX100m7 --directory=../
# python csv2coco.py --train=Sony_RX100m7_annotation_jpg_train.csv --test=Sony_RX100m7_annotation_jpg_test.csv --val=Sony_RX100m7_annotation_jpg_val.csv --camera=Sony_RX100m7 --directory=../
import argparse
import os
import os.path as op
from PIL import Image
import json

IMAGE_ID_BY_NAME = True

from dataset_info import CATEGORIES, CAT_NAME_TO_ID, ID_TO_CAT_NAME

# https://towardsdatascience.com/getting-started-with-coco-dataset-82def99fa0b8
COCO_JSON_CONTENT_TEMPLATE = '''
  "info": {info_dict},
  "licenses": {licenses_list},
  "images": {images_list},
  "categories": {categories_list},
  "annotations": {annotations_list}'''

IMAGE_DICT_CONTENT_TEMPLATE = '''
    "id": {id}, 
    "width": {width}, 
    "height": {height}, 
    "file_name": "{file_name}", 
    "license": {license}'''

BBOX_ANNOT_DICT_CONTENT_TEMPLATE = '''
    "image_id": {image_id},
    "bbox":
    [
        {x1},
        {y1},
        {width},
        {height}
    ],
    "category_id": {category_id},
    "id": {id}, 
    "iscrowd":0,
    "area": {area}
'''

info_dict = {
  "description": "[WARNING] area tag of the instances is calculated as width*height since we don't have annotations for segmentation!",
  "url": "TODO",
  "version": "TODO",
  "year": 2021,
  "contributor": "TODO",
  "date_created": "TODO"
}

licenses_list = [
  {
    "id": 1,
    "name": "Attribution-NonCommercial-ShareAlike License",
    "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",  
  }
]

categories_list = [
  {"supercategory": "person", "id": 1, "name": "person"},
  {"supercategory": "vehicle", "id": 2, "name": "bicycle"},
  {"supercategory": "vehicle", "id": 3, "name": "car"}
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--train', type=str)
    parser.add_argument('--test', type=str)
    parser.add_argument('--val', type=str)
    parser.add_argument('--camera', type=str)
    parser.add_argument('--directory', type=str, help="Directory to move files to, must exist")
    parser.add_argument('--lowlight_data', type=str, help="Directory to move files to, must exist")
    parser.add_argument('--lowlight_flag', action="store_true")
    args = parser.parse_args()

    assert len(set([args.train, args.test, args.val])) == 3
    
    file_sets = []
    contents = []
    for path in [args.train, args.test, args.val]:
        with open(path, "r") as f:
            content = f.readlines()
            contents.append(content)
        file_set = set()
        for line in content:
            chunks = str(line).split(",")
            assert len(chunks) > 1
            fp = chunks[0]
            file_set.add(str(fp))
        file_sets.append(file_set)
    train_set, test_set, val_set = file_sets
    assert train_set.intersection(test_set) == set()
    assert train_set.intersection(val_set) == set()
    assert test_set.intersection(val_set) == set()
    for file_set, set_name, content in zip([train_set, test_set, val_set], ['train', 'test', 'val'], contents):
        images_list = []
        annotations_list = []
        path_prefix = op.join(args.directory, "{}_{}".format(args.camera, set_name))
        if not op.exists(path_prefix):
            os.mkdir(path_prefix)
        assert op.exists(path_prefix)
        tmp_file_name_to_id = {}
        for idx, path in enumerate(file_set):
            new_path = op.join(path_prefix, op.split(path)[-1])
            if op.exists(path): 
                if op.exists(new_path): raise Exception(new_path)
                os.rename(path, new_path)
            else:
                if not op.exists(new_path): raise Exception(new_path)
            img = Image.open(new_path)
            img.verify()
            height = img.height
            width = img.width
            file_name = op.split(path)[-1]
            file_id = '"'+file_name.split(".")[0]+'"' if IMAGE_ID_BY_NAME else idx
            image_dict = json.loads('{'+IMAGE_DICT_CONTENT_TEMPLATE.format(id=file_id, width=width, height=height, file_name=file_name, license=1)+'}')
            tmp_file_name_to_id[file_name] = file_id
            images_list.append(image_dict)
        for annot_idx, line in enumerate(content):
            chunks = str(line).split(",")
            fp, x1, y1, x2, y2, cat_name = chunks
            file_name = op.split(fp)[-1]
            image_id = tmp_file_name_to_id[file_name]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            width, height = x2-x1, y2-y1
            assert width>0
            assert height>0
            area=int(width*height)
            assert len(cat_name.splitlines()) == 1
            cat_name = cat_name.splitlines()[0]
            assert cat_name in CATEGORIES
            cat_id = CAT_NAME_TO_ID[cat_name]
            bbox_annot_dict = json.loads('{'+BBOX_ANNOT_DICT_CONTENT_TEMPLATE.format(image_id=image_id, x1=x1,y1=y1,width=width,height=height,category_id=cat_id,id=annot_idx,area=area)+'}')
            annotations_list.append(bbox_annot_dict)
        coco_dict = {"info":info_dict, "licenses":licenses_list, "images":images_list, "categories":categories_list, "annotations":annotations_list}
        with open(op.join(args.directory, "{}_{}".format(args.camera, set_name)+".json"), 'w', encoding='utf-8') as f:
            json.dump(coco_dict, f, ensure_ascii=False)