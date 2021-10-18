# python lowlight_labeler.py /home/igor/Desktop/temp/lowlight-label-utils/Sony_RX100m7_annotation_jpg_test.csv --jpg_directory=/home/igor/Desktop/temp/Sony_RX100m7_test --output_file=Lowlight_Sony_RX100m7_annotation_jpg_test.csv

import argparse
import os
import os.path as op
import cv2
import shutil

# from dataset_info import CATEGORIES, CAT_NAME_TO_ID, ID_TO_CAT_NAME

WINDOW_NAME="LOWLIGHT LABELER"
KEYS_TO_INTSTRUCTIONS = {ord("q"): "quit", ord("s"):"switch", ord("a"):"label_as_lowlit", ord("d"):"label_as_welllit"}
SUPPORTED_KEYS = list(KEYS_TO_INTSTRUCTIONS.keys())

ISLOWLIGHT_TEMPLATE = "islowlight={}"

def fxyxyc2key(file_name, x1, y1, x2, y2, cat_name):
    return ",".join([str(a) for a in [file_name, x1, y1, x2, y2, cat_name]])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('dataset_csv', type=str, help='Every line in format: file_name, x1, y1, x2, y2, class_name')
    parser.add_argument('--jpg_directory', type=str, help="Directory with images from the data set")
    parser.add_argument('--max_width', default=800, type=int, help="Max width in pixels")
    parser.add_argument('--max_height', default=600, type=int, help="Max width in pixels")
    parser.add_argument('--output_file', type=str)
    args = parser.parse_args()
    assert args.jpg_directory
    assert args.output_file

    lines_by_fxyxyc_dict = {}
    labeled_lines = []
    if not op.exists(args.output_file):
        shutil.copyfile(args.dataset_csv, args.output_file)
    assert op.exists(args.output_file)

    with open(args.output_file, "r") as f:
        output_content = f.readlines()
        for line_idx, line in enumerate(output_content):
            chunks = str(line).split(",")
            if len(chunks) == 6:
                fp, x1, y1, x2, y2, cat_name = chunks
                assert len(cat_name.splitlines()) == 1
                cat_name = cat_name.splitlines()[0]
                lowlight_flag = None
            elif len(chunks) == 7:
                fp, x1, y1, x2, y2, cat_name, lowlight_flag = chunks
                assert len(lowlight_flag.splitlines()) == 1
                lowlight_flag = lowlight_flag.splitlines()[0]
                assert (lowlight_flag == ISLOWLIGHT_TEMPLATE.format(1) or lowlight_flag == ISLOWLIGHT_TEMPLATE.format(0))
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            file_name = op.split(fp)[-1]
            lines_by_fxyxyc_dict[fxyxyc2key(file_name, x1, y1, x2, y2, cat_name)] = line_idx
            if lowlight_flag == ISLOWLIGHT_TEMPLATE.format(1) or lowlight_flag == ISLOWLIGHT_TEMPLATE.format(0):
                labeled_lines.append(line_idx)

    print("{}/{} lines labeled".format(len(labeled_lines), len(lines_by_fxyxyc_dict.keys())))

    with open(args.dataset_csv, "r") as f:
        content = f.readlines()
        mem_objects = {"img": None, "obj": None, "img_name":None}
        for line in content:
            chunks = str(line).split(",")
            fp, x1, y1, x2, y2, cat_name = chunks
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            file_name = op.split(fp)[-1]
            assert len(cat_name.splitlines()) == 1
            cat_name = cat_name.splitlines()[0]
            line_idx = lines_by_fxyxyc_dict[fxyxyc2key(file_name, x1, y1, x2, y2, cat_name)]
            if line_idx in labeled_lines:
                content[line_idx] = output_content[line_idx]
                continue
            if not op.exists(op.join(args.jpg_directory, file_name)):
                raise Exception("{} does not exist".format(op.join(args.jpg_directory, file_name)))
            if file_name != mem_objects['img_name']:
                mem_objects['img0'] = cv2.imread(op.join(args.jpg_directory, file_name))
                mem_objects['img'] = cv2.imread(op.join(args.jpg_directory, file_name))
                mem_objects['img_name'] = file_name
            mem_objects['obj'] = mem_objects['img0'][y1:y2, x1:x2]
            for n in ["img", "obj"]:
                array = mem_objects[n]
                height, width, ch = array.shape
                if (height > args.max_height) or (width > args.max_width):
                    scale_h = args.max_height / height
                    scale_w = args.max_width / height
                    scale = scale_h if scale_h * width <= args.max_width else scale_w
                    new_h, new_w = int(scale*height), int(scale*width)
                    mem_objects[n] = cv2.resize(mem_objects[n], (new_w, new_h))
            currently_displayed = "obj"
            next_sample = False
            while not next_sample:
                current_display = mem_objects[currently_displayed]
                cv2.imshow(WINDOW_NAME+" {} {}".format(cat_name, file_name), current_display)
                pressed_key = cv2.waitKey(0) & 0xFF
                if pressed_key not in SUPPORTED_KEYS:
                    continue
                instruction = KEYS_TO_INTSTRUCTIONS[pressed_key]
                if instruction == 'quit':
                    cv2.destroyAllWindows()
                    exit() 
                elif instruction == 'switch':
                    currently_displayed = "obj" if currently_displayed == "img" else "img"
                    continue
                elif instruction == "label_as_lowlit" or instruction == "label_as_welllit":
                    value = 1 if instruction == "label_as_lowlit" else 0
                    islowlight_flag=ISLOWLIGHT_TEMPLATE.format(value)
                    assert len(content[line_idx].splitlines())==1
                    content[line_idx]=content[line_idx].splitlines()[0]+","+islowlight_flag+"\n"
                    labeled_lines.append(line_idx)
                    with open(args.output_file, 'w') as o:
                        text = "\n".join([l.splitlines()[0] for l in content])+"\n"
                        o.write(text)
                    cv2.destroyAllWindows()
                    next_sample = True






