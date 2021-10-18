import argparse
import os
import os.path as op
from PIL import Image
import json

import cv2
from matplotlib import pyplot as plt
import numpy as np
import tqdm


CAT_NAMES = ['person', 'bicycle', 'car']
AREA_NAMES = ['all', 'small', 'medium', 'large']
SONY_AREA_RNG = [[0, 1e5 ** 2], [0, 10**4.8], [10**4.8, 10**5.8], [10**5.8, 1e8]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('csv', type=str)
    parser.add_argument('--histogram', action="store_true")
    parser.add_argument('--dataset_dir', type=str)
    parser.add_argument('--percentage', action="store_true")
    args = parser.parse_args()
    assert len(AREA_NAMES) == len(SONY_AREA_RNG)
    if args.histogram:
        assert op.exists(args.dataset_dir)
    cat_count = {c:0 for c in CAT_NAMES}
    with open(args.csv, "r") as f:
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
    if args.percentage:
        dicts = [cat_count] + list(d for d in list(area_counts.values())+list(cat_area_counts.values()))
        for d in dicts:
            d_sum = sum(d.values())
            for k in d.keys():
                d[k] = np.round(d[k]/d_sum*100, 2)
    print(cat_count)
    # print("***************")
    # for k in area_counts.keys():
    #     print(f"{k}: {area_counts[k]}")
    print("***************")
    for k in cat_area_counts.keys():
        print(f"{k}: {cat_area_counts[k]}")
    print("***************")
    # histograms
    if not args.histogram:
        exit()
    counted = set()
    cum = np.zeros_like(np.arange(0,256))
    for annot_idx, line in tqdm.tqdm(enumerate(content)):
        chunks = str(line).split(",")
        fp, x1, y1, x2, y2, cat_name = chunks
        fn = op.split(fp)[-1]
        if fn in counted:
            continue
        img_fp = op.join(args.dataset_dir, fn)
        assert op.exists(img_fp)
        img = cv2.imread(img_fp)/255
        x,bins = np.histogram(img, bins=256, range=(0,1))
        cum += x
        counted.add(fn)
    print("Counted from {} imgs".format(len(counted)))
    cum = cum/cum.sum()
    plt.title("Normalized histogram")
    plt.step(bins[:-1], cum, label=op.split(args.csv)[-1])
    plt.ylabel("Probability")
    plt.xlabel("Normalized pixel level")
    plt.legend()
    plt.show()
    plt.clf()
