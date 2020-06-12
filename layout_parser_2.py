import glob
import re

import lxml
from lxml import etree
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
import numpy as np
import math
from bs4 import BeautifulSoup
import warnings
import run_sentence_predict

warnings.filterwarnings("ignore")

estimator = run_sentence_predict.build_model()


def get_bbox(node):
    bbox = []
    if "bbox" not in node.keys():
        return bbox
    for i in node.attrib["bbox"].split(','):
        bbox.append(float(i))
    return bbox


def get_ascii(character: str):
    try:
        return ord(character)
    except:
        return 10000


def get_word_spaces(tree):
    spaces = {}
    lines = []
    count = 0
    for line in tree.findall(".//textline"):
        bbox = get_bbox(line)
        sizes = []
        max_char_width = 0.0
        text = ''
        for c in line:
            if c.text is None:
                c.text = r"!0"
            if (get_ascii(c.text) > 8000):
                c.text = ''
            if not c.text.isspace():
                c_box = get_bbox(c)
                if not c_box:
                    continue
                char_width = c_box[2] - c_box[0]
                if max_char_width < char_width:
                    max_char_width = char_width
                sizes.append(int(round(float(c.attrib["size"]), 0)))
            text += c.text
        if not sizes:
            continue
        lines.append({
            "node": line,
            "bbox": bbox,
            "max_char_width": max_char_width,
            "wrong_spaces": [],
            "text": text
        })
        size = max(set(sizes), key=sizes.count)
        left = 0
        if size not in spaces.keys():
            spaces[size] = []
        for i in range(len(line)):
            if line[i].text is None or not line[i].text.isspace():
                if i == left:
                    continue
                elif i == left + 1:
                    left = i
                else:
                    width = get_bbox(line[i])[0] - get_bbox(line[left])[2]
                    spaces[size].append({
                        "line": count,
                        "position": i,
                        "width": width
                    })
                    left = i
            else:
                continue
        count += 1

    return spaces, lines


def split_lines(tree):
    count = 0
    spaces, lines = get_word_spaces(tree)
    for size in spaces:
        distances = []
        wrong_distance = 0
        for space in spaces[size]:
            distances.append([space["width"]])
            max_space = lines[space["line"]]["max_char_width"] * 2.3
            if wrong_distance < max_space:
                wrong_distance = max_space
        distances.append([wrong_distance])
        if len(distances) < 2:
            continue
        X = np.array(distances)
        kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
        if kmeans.cluster_centers_[0][0] < kmeans.cluster_centers_[1][0]:
            wrong_space_label = 1
        else:
            wrong_space_label = 0
        for i in range(len(kmeans.labels_) - 1):
            if kmeans.labels_[i] == wrong_space_label and distances[i][0] > lines[spaces[size][i]["line"]][
                "max_char_width"] * 0.5:
                position = spaces[size][i]["position"]
                lines[spaces[size][i]["line"]]["wrong_spaces"].append(position)
    list_wrong_line = []
    list_wrong_text_line = []
    for line in lines:
        if len(line["wrong_spaces"]) != 0 and len(line["wrong_spaces"]) < 4:
            list_wrong_line.append(line)
            list_wrong_text_line.append(line['text'])
            text = line['text']
            print(line['text'])
            count += 1

        else:
            line["node"].tag = "txtline"

    predict_true, predict_false = run_sentence_predict.get_sentence_lm_1(estimator, list_wrong_text_line)
    # print("Predict False: ")
    for i in predict_false:
        tree = add_sub_line(tree, list_wrong_line[i]["node"], list_wrong_line[i]["wrong_spaces"])
        # print(list_wrong_line[i]['text'])
    # print("Predict True: ")
    for i in predict_true:
        # print(list_wrong_line[i]['text'])
        list_wrong_line[i]["node"].tag = "txtline"

    etree.strip_tags(tree, ["textline", "textbox"])
    for layout in tree.findall(".//layout"):
        layout.getparent().remove(layout)
    for line in tree.findall(".//txtline"):
        line.tag = "textline"
        bbox = get_bbox(line)
        right = bbox[2]
        fonts = []
        sizes = []
        ncolours = []
        char_null_attrib = []
        for char in line:
            if "bbox" not in char.keys():
                char.set("bbox", str(right) + "," + str(bbox[1]) + "," + str(right) + "," + str(bbox[3]))
                continue
            else:
                char_box = get_bbox(char)
                if right < char_box[2]:
                    right = char_box[2]
            if "font" in char.keys():
                fonts.append(char.attrib["font"])
            if "size" in char.keys():
                sizes.append(char.attrib["size"])
            if "ncolour" in char.keys():
                ncolours.append(char.attrib["ncolour"])
        if len(fonts) == 0:
            font = "null"
        else:
            font = max(set(fonts), key=fonts.count)
        if len(sizes) == 0:
            size = "null"
        else:
            size = max(set(sizes), key=fonts.count)
        if len(ncolours) == 0:
            ncolour = "null"
        else:
            ncolour = max(set(ncolours), key=ncolours.count)
        for char in line:
            if "font" not in char.keys():
                char.set("font", font)
            if "size" not in char.keys():
                char.set("size", size)
            if "ncolour" not in char.keys():
                char.set("ncolour", ncolour)
    for fig in tree.findall(".//figure"):
        flag = 0
        for ob in fig:
            if ob.tag == "textline":
                flag = 1
                break
        if flag:
            for chil in fig:
                fig.getparent().append(chil)
            fig.getparent().remove(fig)
    return tree


def add_sub_line(tree, line, wrong_spaces):
    new_line = {}
    f = 0
    #   xác định các tập chữ trong các line mới
    for i in range(len(wrong_spaces) + 1):
        if i < len(wrong_spaces):
            l = wrong_spaces[i]
        elif i == len(wrong_spaces):
            l = len(line)
        if i not in new_line.keys():
            new_line[i] = []
        for j in range(f, l):
            new_line[i].append(line[j])
        if i >= len(wrong_spaces):
            break
        f = wrong_spaces[i]

    #   noi cac the line moi, sua lai tree
    for ch in new_line.keys():
        temp = etree.Element("txtline")
        top = str(get_bbox(new_line[ch][0])[3])
        left = str(get_bbox(new_line[ch][0])[0])
        count = 0
        for i in range(len(new_line[ch])):
            if new_line[ch][i].text and not new_line[ch][i].text.isspace():
                count = i
        right = str(get_bbox(new_line[ch][count])[2])
        bottom = str(get_bbox(new_line[ch][0])[1])
        temp.set("bbox", left + "," + bottom + "," + right + "," + top)
        for c in new_line[ch]:
            temp.append(c)
        line.append(temp)
    return tree


def get_lines(tree):
    pages = []
    for page in tree.findall(".//page"):
        lines = []
        for line in page.findall(".//textline"):
            bbox = get_bbox(line)
            page = int(line.getparent().attrib["id"])
            sizes = []
            size = 0
            for char in line:
                if "size" in char.keys():
                    sizes.append(float(char.attrib["size"]))
            if sizes:
                size = max(set(sizes), key=sizes.count)
            lines.append({
                "node": line,
                "bbox": bbox,
                "paragraph": 0,
                "size": size,
            })
        pages.append(lines)
    return pages


def merger_block(tree):
    pages = get_lines(tree)
    metrics = get_matrix_distance(pages)
    blocks = HAC_blocks(pages, metrics)
    wrong_margins = get_wrong_margins_in_block(blocks)
    new_blocks = get_new_block(blocks, wrong_margins)
    tree = append_new_block_to_tree(tree, new_blocks)
    assert len(pages) > 0, "pages is empty"
    return tree


def align_distance(box1, box2):
    height1 = box1[3] - box1[1]
    height2 = box2[3] - box2[1]
    is_vert_overlap = (box1[0] < box2[2]) and (box2[0] < box1[2])
    vert_distance = min(abs(box1[3] - box2[1]), abs(box2[3] - box1[1]))
    if not is_vert_overlap:
        return 900.0
    else:
        if max(height1, height2) == 0:
            return 900.0
        return vert_distance / max(height1, height2)


def get_matrix_distance(pages):
    metrics = []
    for page in pages:
        distance_matrix = []
        for i in range(len(page)):
            temp = []
            for j in range(len(page)):
                if j == i:
                    temp.append(0)
                else:
                    if abs(page[i]["size"] - page[j]["size"]) >= 0.2:
                        temp.append(10.0)
                    else:
                        temp.append(align_distance(page[i]["bbox"], page[j]["bbox"]))
            distance_matrix.append(temp)
        metrics.append(distance_matrix)
    return metrics


def HAC_blocks(pages, metrics):
    clusters = []
    for i in range(len(metrics)):
        if len(metrics[i]) < 3:
            clusters.append([])
            continue
        X = np.array(metrics[i])
        nn = 2
        samples = []
        for line in pages[i]:
            samples.append([line["bbox"][0], line["bbox"][3], line["bbox"][1]])
        Y = np.array(samples)
        while nn < 64:
            try:
                con = kneighbors_graph(Y, nn, include_self=False)
                clustering = AgglomerativeClustering(affinity='precomputed',
                                                     connectivity=con, distance_threshold=2.0,
                                                     linkage='single', memory=None, n_clusters=None).fit_predict(X)
                clusters.append(clustering)
            except:
                nn *= 2
            else:
                break
        if nn == 64:
            clustering = AgglomerativeClustering(affinity='precomputed',
                                                 connectivity=None, distance_threshold=2.0,
                                                 linkage='single', memory=None, n_clusters=None).fit_predict(X)
            clusters.append(clustering)
    # lấy ra các block được gom lại sau HAC
    blocks = []
    for i in range(len(clusters)):
        if len(clusters[i]) == 0:
            continue
        temp_blocks = {}
        for j in range(len(clusters[i])):
            if clusters[i][j] not in temp_blocks.keys():
                temp_blocks[clusters[i][j]] = []
            temp_blocks[clusters[i][j]].append(pages[i][j])
        for key in temp_blocks:
            blocks.append(temp_blocks[key])
    # Sắp xếp lại các line trong block
    for block in blocks:
        for i in range(len(block)):
            max_id = i
            for j in range(i + 1, len(block)):
                if block[j]["bbox"][3] > block[max_id]["bbox"][3]:
                    max_id = j
                elif block[j]["bbox"][3] == block[max_id]["bbox"][3] and \
                        block[j]["bbox"][0] < block[i]["bbox"][0]:
                    max_id = j
            if max_id != i:
                block[i], block[max_id] = block[max_id], block[i]
    return blocks


def line_distance(box1, box2):
    height1 = abs(box1[3] - box1[1])
    height2 = abs(box2[3] - box2[1])
    return min(abs(box1[3] - box2[1]), abs(box2[3] - box1[1])) / max(height1, height2)


def get_wrong_margins_in_block(blocks):
    # Lấy ra mảng các khoảng cách trong block
    line_margins = []
    margins = []
    for i in range(len(blocks)):
        for j in range(len(blocks[i]) - 1):
            is_vert_overlap = blocks[i][j]["bbox"][0] < blocks[i][j + 1]["bbox"][2] and blocks[i][j + 1]["bbox"][0] < \
                              blocks[i][j]["bbox"][2]
            is_horz_overlap = not is_vert_overlap and blocks[i][j]["bbox"][1] < blocks[i][j + 1]["bbox"][3] and \
                              blocks[i][j + 1]["bbox"][1] < blocks[i][j]["bbox"][3]
            if is_horz_overlap:
                continue
            d = line_distance(blocks[i][j]["bbox"], blocks[i][j + 1]["bbox"])
            line_margins.append({
                "height": d,
                "block": i,
                "position": j + 1
            })
            margins.append([d])
    margins.append([0.8])
    wrong_margins = {}
    if len(margins) <= 2:
        return wrong_margins
    X = np.array(margins)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    # lay tap cac khoang cach sai
    if kmeans.cluster_centers_[0][0] < kmeans.cluster_centers_[1][0]:
        label = 1
    else:
        label = 0
    for l in range(len(kmeans.labels_) - 1):
        if kmeans.labels_[l] == label:
            block_id = line_margins[l]["block"]
            if block_id not in wrong_margins.keys():
                wrong_margins[block_id] = []
            position = line_margins[l]["position"]
            wrong_margins[block_id].append(position)
    return wrong_margins


# ham lay toa do cho block
def get_bbox_block(lines):
    bbox = get_bbox(lines[0]["node"])
    top = bbox[3]
    left = bbox[0]
    right = bbox[2]
    bottom = bbox[1]
    for i in range(1, len(lines)):
        tb = get_bbox(lines[i]["node"])
        if top < tb[3]:
            top = tb[3]
        if left > tb[0]:
            left = tb[0]
        if right < tb[2]:
            right = tb[2]
        if bottom > tb[1]:
            bottom = tb[1]
    return left, bottom, right, top


def get_bbox_sentence(lines):
    bbox = get_bbox(lines[0])
    top = bbox[3]
    left = bbox[0]
    right = bbox[2]
    bottom = bbox[1]
    for i in range(1, len(lines)):
        tb = get_bbox(lines[i])
        if top < tb[3]:
            top = tb[3]
        if left > tb[0]:
            left = tb[0]
        if right < tb[2]:
            right = tb[2]
        if bottom > tb[1]:
            bottom = tb[1]
    return left, bottom, right, top


def get_new_block(blocks, wrong_margins):
    # lay mang cac block moi
    new_blocks = []
    for i in range(len(blocks)):
        if i in wrong_margins:
            for j in range(len(wrong_margins[i]) + 1):
                lines = []
                if j == 0:
                    lines = blocks[i][:wrong_margins[i][j]]
                elif j == len(wrong_margins[i]):
                    lines = blocks[i][wrong_margins[i][-1]:]
                else:
                    lines = blocks[i][wrong_margins[i][j - 1]:wrong_margins[i][j]]
                new_blocks.append({
                    "bbox": get_bbox_block(lines),
                    "lines": lines,
                    "page": int(lines[0]["node"].getparent().attrib["id"])
                })
        else:
            new_blocks.append({
                "bbox": get_bbox_block(blocks[i]),
                "lines": blocks[i],
                "page": int(blocks[i][0]["node"].getparent().attrib["id"])
            })
    # sap xep lai cac block
    for i in range(len(new_blocks)):
        max_id = i
        for j in range(i + 1, len(new_blocks)):
            if new_blocks[j]["page"] < new_blocks[max_id]["page"]:
                max_id = j
            elif new_blocks[j]["page"] == new_blocks[max_id]["page"]:
                if new_blocks[j]["bbox"][3] > new_blocks[max_id]["bbox"][3]:
                    max_id = j
                elif new_blocks[j]["bbox"][3] == new_blocks[max_id]["bbox"][3] and \
                        new_blocks[j]["bbox"][0] < new_blocks[max_id]["bbox"][0]:
                    max_id = j
        if max_id != i:
            new_blocks[i], new_blocks[max_id] = new_blocks[max_id], new_blocks[i]
    return new_blocks


def append_new_block_to_tree(tree, new_blocks):
    count = 1
    for block in new_blocks:
        page = block["lines"][0]["node"].getparent()
        temp = etree.Element("paragraph")
        for line in block["lines"]:
            temp.append(line["node"])
        bbox = block["bbox"]
        temp.set("bbox", str(bbox[0]) + "," + str(bbox[1]) + "," + str(bbox[2]) + "," + str(bbox[3]))
        temp.set("id", str(count))
        page.append(temp)
        count += 1
    return tree


import os
import subprocess
import sys


def _can_read(fpath: str) -> bool:
    return fpath.lower().endswith("pdf")


def cleansing_sentence(sentences):
    # sentences = re.sub(r'[^a-zA-Z0-9 \- + @ =]', '', sentences.strip())
    sentences = re.sub('=', ' ', sentences.strip())
    return sentences.strip()


def check_special(sentence):
    if (sentence.__contains__('@') and len(sentence.split(' ')) < 3):
        return True
    if (sentence.__contains__('protected') and len(sentence.split(' ')) < 3):
        return True
    return False


def check_next_sentence(sentence_before: str, sentence_after):
    try:
        if (len(sentence_before) < 5):
            return True
        if check_special(sentence_before):
            return True
        if check_special(sentence_after):
            return True
        if (sentence_before[-1] == '.'):
            return True
        if (sentence_after[0].islower()):
            return False
        if ((sentence_after[0] == '-' or sentence_after[0] == '+') and (
                sentence_after[1].isalpha() or sentence_after[2].isalpha())):
            return True

        if (len(sentence_after.split(' ')) < 6):
            if (sentence_after[0].isnumeric()):
                return True
            else:
                return False
        if (sentence_after[0] == '-'):
            return True
        return True
    except:
        return True
        print("FALSEEEEE!", sentence_before, sentence_after)

from deepsegment2 import DeepSegment
segmenter = DeepSegment(checkpoint_path='/Users/trinhgiang/Downloads/deepsegment3/checkpoint',params_path='/Users/trinhgiang/Downloads/deepsegment3/params', utils_path='/Users/trinhgiang/Downloads/deepsegment3/utils')
def check_sentence_boundering(sentence):
    list_sentences = segmenter.segment_long(sentence.replace("-",""))
    if (len(list_sentences) > 1):
        return False
    return True


def check_next_sentence_2(sentence_before: str, sentence_after):
    try:
        if (check_special(sentence_before)):
            return True
        if (check_special(sentence_after)):
            return True
        # if (sentence_before[-1] == '.'):
        #     return True
        # if (sentence_after[0].islower()):
        #     return False
        # if (len(sentence_before.split(' ')) < 4 and len(sentence_after.split(' ')) < 4):
        #     return True
        # if ((sentence_after[0] == '-' or sentence_after[0] == '+') and (
        #         sentence_after[1].isupper() or sentence_after[2].isupper())):
        #     return True
        sentence_check = sentence_before + ' ' + sentence_after
        # if (run_sentence_predict.get_sentence_lm_2(estimator, [sentence_check])):
        #     return False
        if (check_sentence_boundering(sentence_check)):
            return False
        return True
    except:
        return True
        print("FALSEEEEE!", sentence_before, sentence_after)


def _get_files(path):
    if os.path.isfile(path):
        fpaths = [path]
    elif os.path.isdir(path):
        fpaths = [os.path.join(path, f) for f in os.listdir(path)]
    else:
        fpaths = glob.glob(path)
    fpaths = [x for x in fpaths if _can_read(x)]
    if len(fpaths) > 0:
        return sorted(fpaths)
    else:
        raise IOError(f"File or directory not found: {path}")


def detect_sentence(tree):
    for par in tree.findall(".//paragraph"):
        lines = par.findall(".//textline")
        list_text_line = []
        for i in range(len(lines)):
            line = ''
            for char in lines[i]:
                line += char.text
            list_text_line.append(cleansing_sentence(line[:-1]))

        # sentence = ''

        i = 0
        while (i < len(list_text_line)):
            # sentence = list_text_line[i]
            sentence = etree.Element("sentence")
            sentence_lines = []
            sentence.append(lines[i])
            sentence_lines.append(lines[i])
            j = i + 1
            while (j < len(list_text_line) and not (check_next_sentence_2(list_text_line[j - 1], list_text_line[j]))):
                # sentence = sentence + ' ' + list_text_line[j]
                sentence.append(lines[j])
                sentence_lines.append(lines[j])
                j += 1
            # sentences.append(sentence)
            bbox = get_bbox_sentence(sentence_lines)
            sentence.set("bbox", str(bbox[0]) + "," + str(bbox[1]) + "," + str(bbox[2]) + "," + str(bbox[3]))
            par.append(sentence)
            i = j

    return tree


#
def get_text_line(line):
    text_line = ''
    for char in line:
        text_line += char.text
    text_line = cleansing_sentence(text_line[:-1])
    last_char_line = line[-1]
    first_char_line = line[0]
    return text_line, first_char_line, last_char_line


def check_style_character(char_1, char_2):
    if ((char_1.attrib["font"] == char_2.attrib["font"]) and (char_1.attrib["size"] == char_2.attrib["size"]) and (
            char_1.attrib["ncolour"] == char_2.attrib["ncolour"])):
        return True
    else:
        return False


def check_next_sentence_3(line_before, line_after):
    text_line_before, first_char_line_before, last_char_line_before = get_text_line(line_before)
    text_line_after, first_char_line_after, last_char_line_after = get_text_line(line_after)
    if not check_style_character(last_char_line_before, first_char_line_after):
        return True
    else:
        return check_next_sentence_2(text_line_before, text_line_after)


def detect_sentence_2(tree):
    for par in tree.findall(".//paragraph"):
        lines = par.findall(".//textline")

        i = 0
        while (i < len(lines)):
            sentence = etree.Element("sentence")
            sentence_lines = []
            sentence.append(lines[i])
            sentence_lines.append(lines[i])
            j = i + 1
            while (j < len(lines) and not (check_next_sentence_3(lines[j - 1], lines[j]))):
                sentence.append(lines[j])
                sentence_lines.append(lines[j])
                j += 1
            # sentences.append(sentence)
            bbox = get_bbox_sentence(sentence_lines)
            sentence.set("bbox", str(bbox[0]) + "," + str(bbox[1]) + "," + str(bbox[2]) + "," + str(bbox[3]))
            par.append(sentence)
            i = j

    return tree


def analysis(tree):
    tree = split_lines(tree)
    tree = merger_block(tree)
    tree = detect_sentence_2(tree)
    return tree


# DOC_PATH = '/Users/trinhgiang/Downloads/Gold_Label'
# files = _get_files(DOC_PATH)
# file = '/Users/trinhgiang/Downloads/Gold_Label/LOAN-B-TopCV.vn-151219.225339.pdf'
# # Tach line
# xml_content = subprocess.check_output(
#     f"pdf2txt.py -t xml -M 3 -A '{file}' ", shell=True
# )
# soup = BeautifulSoup(xml_content, "lxml")
# all_xml_elements = soup.find_all("pages")
# if len(all_xml_elements) != 1:
#     raise NotImplementedError(
#         f"unsupported format file: {file}"
#     )
# text = all_xml_elements[0]
# tree = etree.fromstring(str(text))
#
# tree = split_lines(tree)
# # print("SPLIT LINE")
# # for line in tree.findall(".//textline"):
# #     t = ""
# #     for text in line:
# #         t += text.text
# #     print(t)
# #     print("---------")
#
# tree = merger_block(tree)
#
# print("MERGE BLOCK")
#
# from deepsegment2 import DeepSegment
# segmenter = DeepSegment(checkpoint_path='/Users/trinhgiang/Downloads/deepsegment3/checkpoint',params_path='/Users/trinhgiang/Downloads/deepsegment3/params', utils_path='/Users/trinhgiang/Downloads/deepsegment3/utils')
#
# for par in tree.findall(".//paragraph"):
#     p = ""
#     for line in par:
#         t=''
#         for c in line:
#             if c.text is None:
#                 t += r"!0"
#             else:
#                 t += c.text
#         p = p + " " + t[:-1].strip()
#         # print(t)
#
#     list_sentences = segmenter.segment_long(p.replace("-",""))
#
#     for sent in list_sentences:
#         print(sent+"\n")
#     print("----------------------------------------------------------------------------------------")

# tree = detect_sentence_2(tree)
# with open("./test.txt", 'a+', encoding="utf-8") as total_file:
#     for par in tree.findall(".//paragraph"):
#         total_file.write('\n')
#         for sentence in par.findall(".//sentence"):
#             lines = sentence.findall(".//textline")
#             sentence = ''
#             for i in range(len(lines)):
#                 line = ''
#                 for char in lines[i]:
#                     line += char.text
#                 sentence = sentence + " " + cleansing_sentence(line[:-1])
#             total_file.write(sentence)
#             total_file.write('\n')

# print(check_sentence_boundering("- Xây dựng, quản lý và phát triển content trên các kênh Facebook, Instagram, Zalo...- Sáng tạo nội dung hình ảnh, thực hiện thiết kế hình ảnh."))
# print(re.sub(r'[^a-zA-Z0-9 \- + @ ]', '', "Ngày sinh"))
