import argparse
import os
import os.path as op
from PIL import Image
import json
import glob

FIELDS = ["img_name", "obj_class", "light_type", "env", "split", "l", "t", "w", "h"]
CLASSES = ["Bicycle", "Boat", "Bottle", "Bus", "Car", "Cat", "Chair", "Cup", "Dog", "Motorbike", "People", "Table"]
INT2CLASS = {idx: obj_class for idx, obj_class in enumerate(CLASSES)}
SPLITS = ['train', 'val', 'test']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--anno', type=str)
    parser.add_argument('--imageclasslist', type=str)
    args = parser.parse_args()

    assert op.exists(args.anno)
    assert op.exists(args.imageclasslist)

    csvs_full = {k:[] for k in SPLITS}
    csvs_short = {k:[] for k in SPLITS}

    with open(args.imageclasslist, 'r') as f:
        content = f.readlines()[1:]
        l = []
        imageclasslist_dict = {}
        for line in content:
            line = line.rstrip()
            img_name, obj_class, light_type, env, split = line.split(" ") 
            l += [img_name]
            imageclasslist_dict[img_name] = {
                "obj_class" : obj_class,
                "light_type" : light_type, 
                "env" : env, 
                "split" : split
            }
        assert len(l) == len(set(l))
    
    for path in glob.glob(op.join(args.anno, "*", "*.jpg.txt")):
        img_name = op.split(path)[-1].split(".txt")[0]
        try:
            d = imageclasslist_dict[img_name]
        except KeyError:
            img_name_no_ext = img_name.split(".")[0]
            for k in imageclasslist_dict.keys():
                if k.split(".")[0] ==  img_name_no_ext:
                    break
            d = imageclasslist_dict[k]
        light_type = int(d["light_type"])
        env = int(d["env"])
        split = int(d["split"]) 
        assert split in [1, 2, 3]
        with open(path, 'r') as f:
            content = f.readlines()[1:]
            for line in content:
                line = line.rstrip()
                obj_class, l, t, w, h = line.split(" ")[:5]
                l, t, w, h = int(l), int(t), int(w), int(h)
                # convert to csv
                x1 = l
                y1 = t
                x2 = x1 + w
                y2 = y1 + h
                l = [str(e) for e in [img_name, x1, y1, x2, y2, obj_class]]
                csvs_short[SPLITS[split - 1]].append(",".join(l))
                l += [f"lighting_type={light_type}"]
                outdoor = -1
                if env == 1:
                    outdoor = 0
                elif env == 2:
                    outdoor = 1
                if not outdoor >= 0: raise Exception(line, img_name, d, outdoor)
                assert outdoor >= 0 
                l += [f"outdoor={outdoor}"]
                csvs_full[SPLITS[split - 1]].append(",".join(l))

    for split in SPLITS:
        with open(f"ExDark_{split}.csv", 'w') as o:
            text = "\n".join([l.splitlines()[0] for l in csvs_full[split]])+"\n"
            o.write(text)
        with open(f"ExDark_{split}_full.csv", 'w') as o:
            text = "\n".join([l.splitlines()[0] for l in csvs_short[split]])+"\n"
            o.write(text)
                    
    

