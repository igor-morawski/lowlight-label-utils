# SONY: 
# python -i coco2subset.py --anno=/home/igor/Desktop/temp/instances_train2017.json --classes=person,bicycle,car --train=10069,3287,1626 --val=1283,403,210 --test=1232,398,197 --train_imgs=2570 --val_imgs=320 --test_imgs=320 --seeds=100
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

SUBSETS = ["train", "val", "test"]
JSON_OUT_TEMPLATE = "COCO_SUB_{}.json"
CSV_OUT_TEMPLATE = "COCO_SUB_{}.csv"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--anno', type=str)
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--classes', type=str, default="person,bicycle,car", help="Separated by comma ,")
    parser.add_argument('--train', type=str, help="Number of instances desired for each class [order as in args.classes] -- not guaranteed. Separated by comma ,")
    parser.add_argument('--val', type=str, help="Number of instances desired for each class [order as in args.classes] -- not guaranteed. Separated by comma ,")
    parser.add_argument('--test', type=str, help="Number of instances desired for each class [order as in args.classes] -- not guaranteed. Separated by comma ,")
    parser.add_argument('--train_imgs', type=int)
    parser.add_argument('--val_imgs', type=int)
    parser.add_argument('--test_imgs', type=int)
    parser.add_argument('--seeds', type=int, default=50000)
    args = parser.parse_args()

    cocoGt=COCO(args.anno)
    
    with open(args.anno) as json_file:
        json_data = json.load(json_file)
        json_data_img_ids = {i['id'] : idx for idx, i in enumerate(json_data['images'])}
        json_data_ann_ids = {a['id'] : idx for idx, a in enumerate(json_data['annotations'])}
    output_data = {"train" : {}, "val" : {}, "test" : {}}
    cats = args.classes.split(",")
    for cat_name in cats:
        if len(cocoGt.getCatIds(cat_name)) != 1:
            raise ValueError(f"{cat_name} not in {args.anno}")
    for arg in [args.train, args.val, args.test]:
        if len(arg.split(",")) != len(cats):
            raise ValueError()
    output_classes = list(filter(lambda x: x['name'] in cats, json_data['categories'].copy())) 
    
            
    if args.debug:
        JSON_OUT_TEMPLATE = "debug_" + JSON_OUT_TEMPLATE
        CSV_OUT_TEMPLATE = "debug_" + CSV_OUT_TEMPLATE
        args.classes = "person,bicycle,car"
        cats = args.classes.split(",")
        args.train = "7,8,9"
        args.val = "1,2,3"
        args.test = "4,5,6"
    train, val, test = [arg.split(",") for arg in [args.train, args.val, args.test]] 
    targets = {}
    for set_name, cats_instances in zip(["train", "val", "test"], [train, val, test]):
        targets[set_name] = {}
        for cat, instances in zip(cats, cats_instances):
            targets[set_name][cat] = int(instances)
            if int(instances) < 0:
                raise ValueError(f"Incorrect value {instances} for number of instances of {cat} in {set_name}")

    cat_ids = [cocoGt.getCatIds(catNms=cat)[0] for cat in cats]
    catId2catNm = {id : name for id, name in zip(cat_ids, cats)}

    imgs_n = {k : v for k, v in zip(["train", "val", "test"], [args.train_imgs, args.val_imgs, args.test_imgs])}
    print("Targets for subsets: " + "(debug)" if args.debug else "")
    print(targets)
        
    
    lines = {s : [] for s in SUBSETS}
    
    visited = set()
    img_idx = 0
    img_sets = {s : None for s in SUBSETS}
    distributions = []
    for subset in SUBSETS:
        img_set = set()
        print(f"Fetching {subset} subset...")
        print(targets[subset])
        print({k : np.round(targets[subset][k] / sum(list(targets[subset].values())), 2) 
            for k in targets[subset].keys()})
        cat_downcounter = copy.deepcopy(targets[subset])
        with tqdm.tqdm(total=np.array(list(targets[subset].values())).sum()) as pbar:
            while any(np.array(list(cat_downcounter.values())) > 0):
                img_idx += 1
                ann_ids = cocoGt.getAnnIds(imgIds = [img_idx])
                if not ann_ids or img_idx in visited:
                    continue
                img_info  = cocoGt.loadImgs(ids = [img_idx])
                all_anns = cocoGt.loadAnns(ann_ids)
                anns = []
                for ann in all_anns:
                    if ann['category_id'] in cat_ids: 
                        anns.append(ann)
                        if ann['iscrowd']:
                            anns = []
                            break
                if not anns:
                    continue
                # also continue if adding image is getting us further away from the
                # target by adding objects that there is enough of 
                # and not adding needed ones
                t = [cat_downcounter[cat] for cat in cats]
                gain = 0
                for ann in anns:
                    if cat_downcounter[catId2catNm[ann['category_id']]] <= 0:
                        continue
                    gain += 1
                if not gain:
                    continue
                img_set.add(img_idx)
                visited.add(img_idx)
                for ann in anns:
                    cat_downcounter[catId2catNm[ann['category_id']]] -= 1
                # print(img_info)
                p = (np.array(list(targets[subset].values())) -
                    np.array(list(cat_downcounter.values()))).sum()
                p = max(0, p)
                pbar.update(p)
                # XXX I didn't have to but depedning on your targets
                # you might run out images in your dataset
                # add a condition yourself if that happens  
        # not necessary if sampling randomly anyway!
        print("Sampling...")
        # balance sets
        img_set0 = img_set.copy()
        dists = []
        scores = []
        for seed in range(args.seeds):
            random.seed(seed)
            img_set = set(random.sample(img_set0.copy(), imgs_n[subset]))
            result = {k : 0 for k in targets[subset].keys()}
            for img_idx in img_set:
                ann_ids = cocoGt.getAnnIds(imgIds = [img_idx])
                all_anns = cocoGt.loadAnns(ann_ids)
                anns = []
                for ann in all_anns:
                    if ann['category_id'] in cat_ids: 
                        anns.append(ann)
                        assert not ann['iscrowd']
                for ann in anns:
                    result[catId2catNm[ann['category_id']]] += 1
            dist = np.array([result[cat] for cat in cats])
            dist = dist/dist.sum()
            if not distributions:
                distributions.append(dist)
                best_seed = seed
                break
            score = 0
            dist2 = distributions[-1]
            score += np.abs((dist-dist2)**2).sum()
            scores.append(score)
            dists.append(dist)

        # # or mse from tgt distribution
        # img_set0 = img_set.copy()
        # dists = []
        # scores = []
        # for seed in range(args.seeds):
        #     random.seed(seed)
        #     img_set = set(random.sample(img_set0.copy(), imgs_n[subset]))
        #     result = {k : 0 for k in targets[subset].keys()}
        #     for img_idx in img_set:
        #         ann_ids = cocoGt.getAnnIds(imgIds = [img_idx])
        #         all_anns = cocoGt.loadAnns(ann_ids)
        #         anns = []
        #         for ann in all_anns:
        #             if ann['category_id'] in cat_ids: 
        #                 anns.append(ann)
        #                 assert not ann['iscrowd']
        #         for ann in anns:
        #             result[catId2catNm[ann['category_id']]] += 1
        #     dist = np.array([result[cat] for cat in cats])
        #     dist = dist/dist.sum()
        #     tgt = np.array([targets[subset][cat] for cat in cats])
        #     tgt = tgt/tgt.sum()
        #     score = np.abs((dist-tgt)**2).sum()
        #     scores.append(score)
        #     dists.append(dist)
        if scores:
            best_seed = np.argmin(scores)
            distributions.append(dists[best_seed])
            print(dists[best_seed])
            # print(scores[best_seed], dists[best_seed])
        random.seed(best_seed)
        img_set = set(random.sample(img_set0.copy(), imgs_n[subset]))
        
        print(f"Resulting {subset} set")
        result = {k : 0 for k in targets[subset].keys()}
        for img_idx in img_set:
            ann_ids = cocoGt.getAnnIds(imgIds = [img_idx])
            all_anns = cocoGt.loadAnns(ann_ids)
            anns = []
            for ann in all_anns:
                if ann['category_id'] in cat_ids: 
                    anns.append(ann)
                    if ann['iscrowd']:
                        continue
            for ann in anns:
                result[catId2catNm[ann['category_id']]] += 1
        # result = {k : targets[subset][k] - cat_downcounter[k] for k in targets[subset].keys()}
        img_sets[subset] = img_set
        print(result)
        print({k : np.round(result[k] / sum(list(result.values())), 2) 
            for k in result.keys()})
        print(f"{len(img_set)} images in {subset}")
    
    
    print("Validating sets (asserting there's no overlap)... ")
    img_sets_list = list(img_sets.values())
    while img_sets_list:
        s1 = img_sets_list.pop()
        for s2 in img_sets_list:
            assert not s1.intersection(s2)
    print("OK.")

    for subset in SUBSETS:
        output_data[subset] = copy.deepcopy(json_data)
        output_data[subset]['images'] = []
        output_data[subset]['annotations'] = []
        output_data[subset]['categories'] = output_classes
        img_set = img_sets[subset]
        for img_idx in img_set:
            if img_idx == 196610: continue
            ann_ids = cocoGt.getAnnIds(imgIds = [img_idx])
            all_anns = cocoGt.loadAnns(ann_ids)
            img_info = json_data['images'][json_data_img_ids[img_idx]]
            output_data[subset]['images'].append(img_info) 
            for ann_id in ann_ids:
                # XXX filter
                ann = json_data['annotations'][json_data_ann_ids[ann_id]]
                # sanity check
                assert ann['image_id'] == img_idx
                if ann['category_id'] in cat_ids: 
                    assert not ann['iscrowd']
                    output_data[subset]['annotations'].append(ann)
    
    for subset in SUBSETS:
        with open(JSON_OUT_TEMPLATE.format(subset), 'w') as outfile:
            json.dump(output_data[subset], outfile)
        lines = []
        tmp_output_id2idx = {i['id'] : idx for idx, i in enumerate(output_data[subset]['images'])}
        for ann in output_data[subset]['annotations']:
            img_name = output_data[subset]['images'][tmp_output_id2idx[ann['image_id']]]['file_name']
            bbox = ann['bbox']
            TLx, TLy, W, H = bbox
            x1, y1, x2, y2 = TLx, TLy, TLx+W, TLy+H
            obj_class = catId2catNm[ann['category_id']]
            l = [str(e) for e in [img_name, x1, y1, x2, y2, obj_class]]
            l = ",".join(l)
            lines.append(l)
        with open(CSV_OUT_TEMPLATE.format(subset), 'w') as outfile:
            text = "\n".join(lines)+"\n"
            outfile.write(text)
        