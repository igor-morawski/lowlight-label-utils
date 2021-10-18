import argparse
import os
import os.path as op
from PIL import Image
import json

CAT_NAMES = ['person', 'bicycle', 'car']
AREA_NAMES = ['all', 'small', 'medium', 'large']
SONY_AREA_RNG = [[0, 1e5 ** 2], [0, 10**4.8], [10**4.8, 10**5.8], [10**5.8, 1e8]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('csv', type=str)
    args = parser.parse_args()
    assert len(AREA_NAMES) == len(SONY_AREA_RNG)
    cat_count = {c:0 for c in CAT_NAMES}
    with open(args.csv, "r") as f:
        content = f.readlines()
    for val_to_filter_out in [-1, 1, 0]:
        result = {k:0 for k in AREA_NAMES}
        for annot_idx, line in enumerate(content):
            chunks = str(line).split(",")
            fp, x1, y1, x2, y2, cat_name, islowlight_flag = chunks
            assert cat_name in cat_count.keys()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            width, height = x2-x1, y2-y1
            assert width>0
            assert height>0
            area=int(width*height)
            assert len(cat_name.splitlines()) == 1
            cat_name = cat_name.splitlines()[0]
            assert islowlight_flag.startswith("islowlight=")
            ill_chunks = islowlight_flag.split("islowlight=")
            assert len(ill_chunks) == 2
            assert ill_chunks[0] == ""
            islowlight_val = int(ill_chunks[1])
            assert (islowlight_val == 0) or (islowlight_val == 1)
            if islowlight_val == val_to_filter_out:
                continue
            for idx, area_type in enumerate(AREA_NAMES):
                m, M = SONY_AREA_RNG[idx]
                if m <= area <= M:
                    result[area_type] += 1
        single_loop_finished = True
        light_type = None
        if val_to_filter_out == -1:
            light_type = "all"
        elif val_to_filter_out == 1:
            light_type = "high"
        elif val_to_filter_out == 0:
            light_type = "low"    
        print(light_type, result)