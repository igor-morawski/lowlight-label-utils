import argparse
import os
import os.path as op
from PIL import Image
import json

IMAGE_ID_BY_NAME = False

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

# licenses_list = [
#   {
#     "id": 1,
#     "name": "Attribution-NonCommercial-ShareAlike License",
#     "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",  
#   }
# ]

# python csv2coco_sid.py --train=Sony_train_raw.csv --val=Sony_val_raw.csv --test=Sony_test_raw.csv --dataset_name=sid_raw --coco_anno_json=instances_val2017.json --height=1424 --width=2128
# python csv2coco_sid.py --train=Sony_train_rawpy.csv --val=Sony_val_rawpy.csv --test=Sony_test_rawpy.csv --dataset_name=sid_rawpy --coco_anno_json=instances_val2017.json --height=2848 --width=4256

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--train', type=str)
    parser.add_argument('--test', type=str)
    parser.add_argument('--val', type=str)
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--coco_anno_json', type=str, help="path to a coco json (to be used as a template), e.g. instances_val2017.json")
    parser.add_argument('--data_root', type=str, default=None)
    parser.add_argument('--height', type=int, default=None)
    parser.add_argument('--width', type=int, default=None)
    args = parser.parse_args()

    coco_example = json.load(open(args.coco_anno_json, "r"))
    categories_list = coco_example['categories']
    licenses_list = coco_example['licenses']
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
        tmp_file_name_to_id = {}
        for idx, path in enumerate(file_set):
            _, file_name = op.split(path)
            if args.data_root:
                full_path = op.join(args.data_root, file_name)
                img = Image.open(full_path)
                img.verify()
                height = img.height
                width = img.width
                file_name = op.split(path)[-1]
            else:
                assert args.height
                assert args.width
                height = args.height
                width = args.width
            file_id = '"'+file_name.split(".")[0]+'"' if IMAGE_ID_BY_NAME else idx
            image_dict = json.loads('{'+IMAGE_DICT_CONTENT_TEMPLATE.format(id=file_id, width=width, height=height, file_name=file_name, license=1)+'}')
            tmp_file_name_to_id[file_name] = file_id
            images_list.append(image_dict)
        for annot_idx, line in enumerate(content):
            chunks = str(line).split(",")
            fp, cat_id, x1, y1, x2, y2, score  = chunks
            file_name = op.split(fp)[-1]
            image_id = tmp_file_name_to_id[file_name]
            x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
            width, height = x2-x1, y2-y1
            if not width>=0: raise Exception("Width smaller than zero in {}".format(line))
            if not height>=0: raise Exception("Height smaller than zero in {}".format(line))
            area=float(width*height)
            bbox_annot_dict = json.loads('{'+BBOX_ANNOT_DICT_CONTENT_TEMPLATE.format(image_id=image_id, x1=x1,y1=y1,width=width,height=height,category_id=cat_id,id=annot_idx,area=area)+'}')
            annotations_list.append(bbox_annot_dict)
        coco_dict = {"info":info_dict, "licenses":licenses_list, "images":images_list, "categories":categories_list, "annotations":annotations_list}
        with open(op.join("{}_{}".format(args.dataset_name, set_name)+".json"), 'w', encoding='utf-8') as f:
            json.dump(coco_dict, f, ensure_ascii=False)