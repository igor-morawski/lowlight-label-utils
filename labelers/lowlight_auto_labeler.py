# python lowlight_auto_labeler.py /home/igor/Desktop/temp/lowlight-label-utils/Sony_RX100m7_annotation_jpg_test.csv --jpg_directory=/home/igor/Desktop/temp/Sony_RX100m7_test --output_file=Lowlight_Sony_RX100m7_annotation_jpg_test_hist.csv --threshold=0.15 --save

import argparse
import os
import os.path as op
import cv2
import shutil
import tqdm
from skimage.exposure import is_low_contrast
from iou import get_iou

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
    parser.add_argument('--threshold', type=float)
    parser.add_argument('--save', action="store_true")
    parser.add_argument('--output_directory', default="auto_labeler", type=str)
    parser.add_argument('--no_overlap', action="store_true") # ensure no overlap between different categories
    parser.add_argument('--iou_thrsh', default=0.1, type=float)
    args = parser.parse_args()
    assert args.jpg_directory
    assert args.output_file

    lines_by_fxyxyc_dict = {}
    anno_by_filename = {}
    skip_lines_indices = []
    labeled_lines = []
    shutil.copyfile(args.dataset_csv, args.output_file)
    assert op.exists(args.output_file)
    if not op.exists(args.output_directory):
        os.mkdir(args.output_directory)
    if not op.exists(op.join(args.output_directory, "0")):
        os.mkdir(op.join(args.output_directory, "0"))
    if not op.exists(op.join(args.output_directory, "1")):
        os.mkdir(op.join(args.output_directory, "1"))
    assert op.exists(op.join(args.output_directory, "0"))
    assert op.exists(op.join(args.output_directory, "1"))
    assert op.exists(args.output_directory)


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
            if args.no_overlap:
                assert x1 < x2
                assert y1 < y2
                if file_name not in anno_by_filename.keys():
                    anno_by_filename[file_name] = [[x1, y1, x2, y2, cat_name]]
                else:
                    indices2keep = []
                    for ann_idx, ann in enumerate(anno_by_filename[file_name]):
                        ann_x1, ann_y1, ann_x2, ann_y2, ann_cat_name = ann
                        # if cat_name != ann_cat_name:
                        if True: # XXX
                            b1 = {'x1':x1, 'y1':y1, 'x2':x2, 'y2':y2}
                            b2 = {'x1':ann_x1, 'y1':ann_y1, 'x2':ann_x2, 'y2':ann_y2}
                            iou = get_iou(b1, b2)
                            if iou > args.iou_thrsh:
                                skip_lines_indices.append(lines_by_fxyxyc_dict[fxyxyc2key(file_name, x1, y1, x2, y2, cat_name)])
                                skip_lines_indices.append(lines_by_fxyxyc_dict[fxyxyc2key(file_name, ann_x1, ann_y1, ann_x2, ann_y2, ann_cat_name)])
                                continue
                        indices2keep.append(ann_idx)
                    if len(indices2keep) != len(anno_by_filename[file_name]):
                        selected_elements = [anno_by_filename[file_name][i] for i in indices2keep]
                        anno_by_filename[file_name] = selected_elements
                    else: # no overlap with the same cat
                        anno_by_filename[file_name] = anno_by_filename[file_name] + [[x1, y1, x2, y2, cat_name]]
                    
    print("{}/{} lines labeled".format(len(labeled_lines), len(lines_by_fxyxyc_dict.keys())))

    with open(args.dataset_csv, "r") as f:
        content = f.readlines()
        low_count = 0
        for idx, line in tqdm.tqdm(enumerate(content)):
            chunks = str(line).split(",")
            fp, x1, y1, x2, y2, cat_name = chunks
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            file_name = op.split(fp)[-1]
            assert len(cat_name.splitlines()) == 1
            cat_name = cat_name.splitlines()[0]
            line_idx = lines_by_fxyxyc_dict[fxyxyc2key(file_name, x1, y1, x2, y2, cat_name)]
            img = cv2.imread(op.join(args.jpg_directory, file_name))[y1:y2, x1:x2]
            islowlight_flag = int(is_low_contrast(img, args.threshold))
            low_count+=islowlight_flag
            content[line_idx]=content[line_idx].splitlines()[0]+","+ISLOWLIGHT_TEMPLATE.format(islowlight_flag)+"\n"
            labeled_lines.append(line_idx)
            if args.save:
                if args.no_overlap and line_idx in skip_lines_indices:
                    continue
                cv2.imwrite(op.join(args.output_directory, str(islowlight_flag), "{}_{}_{}.png".format(cat_name, "low" if islowlight_flag else "high", idx)), img)

    with open(args.output_file, 'w') as o:
        text = "\n".join([l.splitlines()[0] for l in content])+"\n"
        o.write(text)
    print(low_count)

    # raise NotImplementedError
    # with open(args.dataset_csv, "r") as f:
    #     content = f.readlines()
    #     mem_objects = {"img": None, "obj": None, "img_name":None}
    #     for line in content:
    #         chunks = str(line).split(",")
    #         fp, x1, y1, x2, y2, cat_name = chunks
    #         x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    #         file_name = op.split(fp)[-1]
    #         assert len(cat_name.splitlines()) == 1
    #         cat_name = cat_name.splitlines()[0]
    #         line_idx = lines_by_fxyxyc_dict[fxyxyc2key(file_name, x1, y1, x2, y2, cat_name)]
    #         if line_idx in labeled_lines:
    #             content[line_idx] = output_content[line_idx]
    #             continue
    #         if not op.exists(op.join(args.jpg_directory, file_name)):
    #             raise Exception("{} does not exist".format(op.join(args.jpg_directory, file_name)))
    #         if file_name != mem_objects['img_name']:
    #             mem_objects['img0'] = cv2.imread(op.join(args.jpg_directory, file_name))
    #             mem_objects['img'] = cv2.imread(op.join(args.jpg_directory, file_name))
    #             mem_objects['img_name'] = file_name
    #         mem_objects['obj'] = mem_objects['img0'][y1:y2, x1:x2]
    #         for n in ["img", "obj"]:
    #             array = mem_objects[n]
    #             height, width, ch = array.shape
    #             if (height > args.max_height) or (width > args.max_width):
    #                 scale_h = args.max_height / height
    #                 scale_w = args.max_width / height
    #                 scale = scale_h if scale_h * width <= args.max_width else scale_w
    #                 new_h, new_w = int(scale*height), int(scale*width)
    #                 mem_objects[n] = cv2.resize(mem_objects[n], (new_w, new_h))
    #         currently_displayed = "obj"
    #         next_sample = False
    #         while not next_sample:
    #             current_display = mem_objects[currently_displayed]
    #             cv2.imshow(WINDOW_NAME+" {} {}".format(cat_name, file_name), current_display)
    #             pressed_key = cv2.waitKey(0) & 0xFF
    #             if pressed_key not in SUPPORTED_KEYS:
    #                 continue
    #             instruction = KEYS_TO_INTSTRUCTIONS[pressed_key]
    #             if instruction == 'quit':
    #                 cv2.destroyAllWindows()
    #                 exit() 
    #             elif instruction == 'switch':
    #                 currently_displayed = "obj" if currently_displayed == "img" else "img"
    #                 continue
    #             elif instruction == "label_as_lowlit" or instruction == "label_as_welllit":
    #                 value = 1 if instruction == "label_as_lowlit" else 0
    #                 islowlight_flag=ISLOWLIGHT_TEMPLATE.format(value)
    #                 assert len(content[line_idx].splitlines())==1
    #                 content[line_idx]=content[line_idx].splitlines()[0]+","+islowlight_flag+"\n"
    #                 labeled_lines.append(line_idx)
    #                 with open(args.output_file, 'w') as o:
    #                     text = "\n".join([l.splitlines()[0] for l in content])+"\n"
    #                     o.write(text)
    #                 cv2.destroyAllWindows()
    #                 next_sample = True