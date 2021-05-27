# python join_coco_jsons.py /home/igor/Datasets/LightSubset/COCO_SUB_S_train.json /home/igor/Datasets/new_Sony_RX100m7_train.json --output=/home/igor/Datasets/Mixed/Sony_5_Coco_5_train.json
# python join_coco_jsons.py /home/igor/Datasets/LightSubset/COCO_SUB_S_train.json /home/igor/Datasets/new_Sony_RX100m7_train.json --output=/home/igor/Datasets/Mixed/Sony_7_Coco_3_train.json --proportions=3:7
# python join_coco_jsons.py /home/igor/Datasets/LightSubset/COCO_SUB_S_train.json /home/igor/Datasets/new_Sony_RX100m7_train.json --output=/home/igor/Datasets/Mixed/Sony_3_Coco_7_train.json --proportions=7:3
import argparse
import os
import os.path as op
from PIL import Image
import json
from pycocotools.coco import COCO
import copy
import numpy as np 
import tqdm
import random

SEED = 1
random.seed(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('jsons', nargs=2, help="In order of importance. First json is the base, i.e. only images and annotations from the second json are merged to the first argument.")
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--proportions', type=str, help="E.g. 5:5, 7:3, 1:9, etc. ", required=True)
    parser.add_argument('--seeds', type=int, default=100)
    args = parser.parse_args()
    
    if ":" not in args.proportions:
        raise ValueError("Wrong proportion format.")
    proportions = args.proportions.split(":")
    if len(proportions) != 2:
        raise ValueError("Wrong proportion format.")
    p1, p2 = [int(p)/10 for p in proportions]
    if p1+p2 != 1:
        raise ValueError("Proportions should sum to 10.")

    
    json_f1, json_f2 = args.jsons
    with open(json_f1) as json_file:
        json1 = json.load(json_file)
    with open(json_f2) as json_file:
        json2 = json.load(json_file)
    cocoGt1 = COCO(json_f1)
    cocoGt2 = COCO(json_f2)
    
    cats = [c["name"] for c in json1['categories']]
    catId2Nm = {cocoGt1.getCatIds(catNms=cat)[0] : cat for cat in cats}
    # assert images ids do not repeat etc, modify if you don't have this assumption
    json1.keys() == json2.keys()
    assert json1['categories'] == json2['categories']
    ids1 = set(i['id'] for i in json1['images'])
    ids2 = set(i['id'] for i in json2['images'])
    assert not ids1.intersection(ids2)
    ids1 = set(i['id'] for i in json1['annotations'])
    ids2 = set(i['id'] for i in json2['annotations'])
    assert not ids1.intersection(ids2)

    # Sample
    print(f"Samplig in proportion {int(p1*10)}:{int(p2*10)}...")
    imgs_n1 = int(len(json1["images"])*p1)
    json1_ann_ids_image_ids = [(a['id'],a['image_id']) for a in json1['annotations']]
    json1_ImgId2AnnId = {}
    for id, image_id in json1_ann_ids_image_ids:
        if image_id in json1_ImgId2AnnId.keys():
            json1_ImgId2AnnId[image_id].append(id)
        else:
            json1_ImgId2AnnId[image_id] = [id]
    print(f"Sampling json1 from {len(json1['images'])} to {imgs_n1}") 
    imgs1 = json1['images']
    result = {cat : 0 for cat in cats}
    for img in imgs1:
        img_idx = img["id"]
        ann_ids = json1_ImgId2AnnId[img_idx]
        anns = cocoGt1.loadAnns(ids=ann_ids)
        for ann in anns:
            result[catId2Nm[ann['category_id']]] += 1
    orig_result = copy.deepcopy(result)
    orig_dist = np.array([orig_result[cat] for cat in cats])
    orig_dist = np.round(orig_dist/orig_dist.sum(), 2)
    scores = np.zeros(args.seeds)
    for seed in range(args.seeds):
        random.seed(seed)
        imgs1_s = random.sample(json1['images'].copy(), imgs_n1)
        result = {cat : 0 for cat in cats}
        for img in imgs1_s:
            img_idx = img["id"]
            ann_ids = json1_ImgId2AnnId[img_idx]
            anns = cocoGt1.loadAnns(ids=ann_ids)
            for ann in anns:
                result[catId2Nm[ann['category_id']]] += 1
        tmp_result = copy.deepcopy(result)
        tmp_dist = np.array([tmp_result[cat] for cat in cats])
        tmp_dist = np.round(tmp_dist/tmp_dist.sum(), 2)
        score = ((tmp_dist - orig_dist) ** 2).sum()
        scores[seed] = score
    best_seed = np.argmin(scores)
    random.seed(best_seed)
    imgs1_s = random.sample(json1['images'].copy(), imgs_n1)
    result = {cat : 0 for cat in cats}
    for img in imgs1_s:
        img_idx = img["id"]
        ann_ids = json1_ImgId2AnnId[img_idx]
        anns = cocoGt1.loadAnns(ids=ann_ids)
        for ann in anns:
            result[catId2Nm[ann['category_id']]] += 1
    tmp_result = copy.deepcopy(result)
    tmp_dist = np.array([tmp_result[cat] for cat in cats])
    tmp_dist = np.round(tmp_dist/tmp_dist.sum(), 2)
    print(f"Sampled from {orig_dist} to {tmp_dist}")
    random.seed(SEED)


    imgs_n2 = int(len(json2["images"])*p2)
    json2_ann_ids_image_ids = [(a['id'],a['image_id']) for a in json2['annotations']]
    json2_ImgId2AnnId = {}
    for id, image_id in json2_ann_ids_image_ids:
        if image_id in json2_ImgId2AnnId.keys():
            json2_ImgId2AnnId[image_id].append(id)
        else:
            json2_ImgId2AnnId[image_id] = [id]
    print(f"Sampling json2 from {len(json2['images'])} to {imgs_n2}") 
    imgs2 = json2['images']
    result = {cat : 0 for cat in cats}
    for img in imgs2:
        img_idx = img["id"]
        ann_ids = json2_ImgId2AnnId[img_idx]
        anns = cocoGt2.loadAnns(ids=ann_ids)
        for ann in anns:
            result[catId2Nm[ann['category_id']]] += 1
    orig_result = copy.deepcopy(result)
    orig_dist = np.array([orig_result[cat] for cat in cats])
    orig_dist = np.round(orig_dist/orig_dist.sum(), 2)
    scores = np.zeros(args.seeds)
    for seed in range(args.seeds):
        random.seed(seed)
        imgs2_s = random.sample(json2['images'].copy(), imgs_n2)
        result = {cat : 0 for cat in cats}
        for img in imgs2_s:
            img_idx = img["id"]
            ann_ids = json2_ImgId2AnnId[img_idx]
            anns = cocoGt2.loadAnns(ids=ann_ids)
            for ann in anns:
                result[catId2Nm[ann['category_id']]] += 1
        tmp_result = copy.deepcopy(result)
        tmp_dist = np.array([tmp_result[cat] for cat in cats])
        tmp_dist = np.round(tmp_dist/tmp_dist.sum(), 2)
        score = ((tmp_dist - orig_dist) ** 2).sum()
        scores[seed] = score
    best_seed = np.argmin(scores)
    random.seed(best_seed)
    imgs2_s = random.sample(json2['images'].copy(), imgs_n2)
    result = {cat : 0 for cat in cats}
    for img in imgs2_s:
        img_idx = img["id"]
        ann_ids = json2_ImgId2AnnId[img_idx]
        anns = cocoGt2.loadAnns(ids=ann_ids)
        for ann in anns:
            result[catId2Nm[ann['category_id']]] += 1
    tmp_result = copy.deepcopy(result)
    tmp_dist = np.array([tmp_result[cat] for cat in cats])
    tmp_dist = np.round(tmp_dist/tmp_dist.sum(), 2)
    print(f"Sampled from {orig_dist} to {tmp_dist}")
    random.seed(SEED)

    # merge
    json_data = copy.deepcopy(json1)
    img1_idxs = [img["id"] for img in imgs1_s]
    img2_idxs = [img["id"] for img in imgs2_s]
    images1 = list(filter(lambda x: x['id'] in img1_idxs, json1['images'].copy())) 
    images2 = list(filter(lambda x: x['id'] in img2_idxs, json2['images'].copy())) 
    annotations1 = list(filter(lambda x: x['image_id'] in img1_idxs, json1['annotations'].copy())) 
    annotations2 = list(filter(lambda x: x['image_id'] in img2_idxs, json2['annotations'].copy())) 
    json_data['images'] = images1 + images2
    json_data['annotations'] = annotations1 + annotations2

    with open(args.output, 'w') as outfile:
        json.dump(json_data, outfile)
        