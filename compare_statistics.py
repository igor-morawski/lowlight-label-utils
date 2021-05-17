import argparse
import os
import os.path as op
from PIL import Image
import json
import glob

import cv2
from matplotlib import pyplot as plt
import numpy as np
import tqdm

CSV_TEMPLATE = "{}_annotation_jpg_{}.csv"
CAT_NAMES = ['person', 'bicycle', 'car']
AREA_NAMES = ['all', 'small', 'medium', 'large']
SONY_AREA_RNG = [[0, 1e5 ** 2], [0, 10**4.8], [10**4.8, 10**5.8], [10**5.8, 1e8]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dir', type=str, default="/home/igor/Desktop/TWCC/Dataset/all_annotation/seeded")
    parser.add_argument('--camera', type=str, default="Sony_RX100m7")
    parser.add_argument('--train', action="store_true")
    args = parser.parse_args()
    assert len(AREA_NAMES) == len(SONY_AREA_RNG)
    CSV_TEMPLATE = CSV_TEMPLATE.format(args.camera, "{}")
    seed_dict = {}
    for dir in glob.glob(op.join(args.dir, "*")):
        csvs = [op.join(dir, CSV_TEMPLATE.format(s)) for s in ("val", "test")]
        if args.train:
            csvs = [op.join(dir, CSV_TEMPLATE.format(s)) for s in ("train", "val", "test")]
        results = [[], []] if not args.train else [[], [], []]
        results2 = [[], []] if not args.train else [[], [], []]
        for csv_idx, csv in enumerate(csvs):
            cat_count = {c:0 for c in CAT_NAMES}
            with open(csv, "r") as f:
                content = f.readlines()
            area_counts = {a:{c:0 for c in CAT_NAMES} for a in AREA_NAMES}
            cat_area_counts = {c:{a:0 for a in AREA_NAMES[1:]} for c in CAT_NAMES}
            for annot_idx, line in enumerate(content):
                chunks = str(line).split(",")
                fp, x1, y1, x2, y2, cat_name = chunks
                cat_name = cat_name.rstrip()
                x1, y1, x2, y2 = [int(v) for v in (x1, y1, x2, y2)]
                assert cat_name in cat_count.keys()
                cat_count[cat_name] += 1
                for area_idx, area_name in enumerate(AREA_NAMES):
                    m, M = SONY_AREA_RNG[area_idx]
                    if m <= np.abs((x2-x1)*(y2-y1)) <= M:
                        area_counts[area_name][cat_name] += 1
                        if area_name == "all":
                            continue
                        cat_area_counts[cat_name][area_name] += 1
            dicts = [cat_count] + list(d for d in list(area_counts.values())+list(cat_area_counts.values()))
            for d in dicts:
                d_sum = sum(d.values())
                for k in d.keys():
                    d[k] = np.round(d[k]/d_sum*100, 2)
            results[csv_idx] = [cat_count[k] for k in sorted(list(cat_count.keys()))]
            results2[csv_idx] = []
            for k in sorted(list(cat_area_counts.keys())):
                d = cat_area_counts[k]
                results2[csv_idx].extend(d[dk] for dk in sorted(list(d.keys())))
        assert len(results) == 3 if args.train else 2
        score = 0
        score2 = 0
        for v_i in range(len(results[0])):
            score += (results[0][v_i] - results[1][v_i])**2
            if args.train:
                score += (results[0][v_i] - results[2][v_i])**2
                score += (results[1][v_i] - results[2][v_i])**2
        score = np.sqrt(score)
        # for v_i in range(len(results2[0])):
        #     score += (results2[0][v_i] - results2[1][v_i])**2
        #     if (results2[0][v_i] - results2[1][v_i])**2 > 4:
        #         score += 10000
        #     if args.train:
        #         score += (results2[0][v_i] - results2[2][v_i])**2
        #         score += (results2[1][v_i] - results2[2][v_i])**2
        #         if (results2[0][v_i] - results2[2][v_i])**2 > 4:
        #             score += 1
        #         if (results2[1][v_i] - results2[2][v_i])**2 > 4:
        #             score += 10
        # score2 = np.sqrt(score2)
        # score += 1/8 * score2
        score *= -1
        seed_dict[op.split(dir)[-1]] = score
    max_seed = max(int(k) for k in seed_dict.keys())
    assert len(seed_dict.keys()) == max_seed
    scores = np.zeros(max_seed)
    for k, v in seed_dict.items():
        scores[int(k)-1] = v
    ind = np.argpartition(scores, -5)[-5:]
    print("Top 5 scores: ")
    for i in ind:
        print(i+1, scores[i])



