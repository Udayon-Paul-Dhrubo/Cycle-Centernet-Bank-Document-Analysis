import xml.etree.ElementTree as ET

import numpy as np


def xml_to_np(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    if root.tag != "annotation":
        raise Exception("pascal voc xml root element should be annotation, rather than {}".format(root.tag))
    gt_boxes = []
    for elem in root.findall("object/bndbox"):
        startrow = int(elem.find("startrow").text)
        endrow = int(elem.find("endrow").text)
        startcol = int(elem.find("startcol").text)
        endcol = int(elem.find("endcol").text)
        xmin = float(elem.find("xmin").text)
        ymin = float(elem.find("ymin").text)
        xmax = float(elem.find("xmax").text)
        ymax = float(elem.find("ymax").text)

        gt_boxes.append([xmin, ymin, xmax, ymax, startrow, endrow, startcol, endcol])
    np_gt_boxes = np.array(gt_boxes)
    np.set_printoptions(precision=2, suppress=True)
    return np_gt_boxes


def all_xml_to_np(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    if root.tag != "annotation":
        raise Exception("pascal voc xml root element should be annotation, rather than {}".format(root.tag))
    gt_boxes = []
    for elem in root.findall("object/bndbox"):
        startrow = int(elem.find("startrow").text)
        endrow = int(elem.find("endrow").text)
        startcol = int(elem.find("startcol").text)
        endcol = int(elem.find("endcol").text)
        xmin = float(elem.find("xmin").text)
        ymin = float(elem.find("ymin").text)
        xmax = float(elem.find("xmax").text)
        ymax = float(elem.find("ymax").text)

        x1 = float(elem.find("x1").text)
        y1 = float(elem.find("y1").text)
        x2 = float(elem.find("x2").text)
        y2 = float(elem.find("y2").text)
        x3 = float(elem.find("x3").text)
        y3 = float(elem.find("y3").text)
        x4 = float(elem.find("x4").text)
        y4 = float(elem.find("y4").text)

        gt_boxes.append(
            [x1, y1, x2, y2, x3, y3, x4, y4, xmin, ymin, xmax, ymax, startrow, endrow, startcol, endcol]
        )
    np_gt_boxes = np.array(gt_boxes)
    np.set_printoptions(precision=2, suppress=True)
    return np_gt_boxes
