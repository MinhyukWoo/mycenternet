from xml.etree import ElementTree
import numpy as np
import cv2

LABEL_NAME_TO_NUM = {
    "aeroplane": 0,
    "bicycle": 1,
    "bird": 2,
    "boat": 3,
    "bottle": 4,
    "bus": 5,
    "car": 6,
    "cat": 7,
    "chair": 8,
    "cow": 9,
    "diningtable": 10,
    "dog": 11,
    "horse": 12,
    "motorbike": 13,
    "person": 14,
    "pottedplant": 15,
    "sheep": 16,
    "sofa": 17,
    "train": 18,
    "tvmonitor": 19,
}


def center(bounding_box):
    xmin = int(bounding_box.find("xmin").text)
    xmax = int(bounding_box.find("xmax").text)
    ymin = int(bounding_box.find("ymin").text)
    ymax = int(bounding_box.find("ymax").text)
    return ((ymin + ymax) // 2, (xmin + xmax) // 2)


def label_num(name):
    return LABEL_NAME_TO_NUM[name.text]


def get_center_points(file_path):
    annotation = ElementTree.parse(file_path)
    centers = [
        (*center(obj.find("bndbox")), label_num(obj.find("name")))
        for obj in annotation.findall("object")
    ]
    return centers


def get_image_size(file_path):
    annotation = ElementTree.parse(file_path)
    size = annotation.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)
    return (height, width)


def get_raw_label(file_path, image_size=None):
    centers = get_center_points(file_path)
    if image_size is None:
        image_size = get_image_size(file_path)
    else:
        real_image_size = get_image_size(file_path)
        centers = [
            (
                int(y * (image_size[0] / real_image_size[0])),
                int(x * (image_size[1] / real_image_size[1])),
                c,
            )
            for y, x, c in centers
        ]
    out = np.zeros((*image_size, len(LABEL_NAME_TO_NUM)))
    for center in centers:
        out[center[0], center[1], center[2]] = 1
    return out


def get_label(file_path, image_size=None):
    raw_label = get_raw_label(file_path, image_size)
    label = cv2.GaussianBlur(raw_label, (0, 0), 5)
    label = label / label.max()
    return label
