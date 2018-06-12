import glob
import os

import cv2
from lxml import etree

# set your working directory
os.chdir('../../../')
goal_dir = './Others/Annotation/'

c_name = 'Alef'
dirList = glob.glob("Others/monkbrill_171005/" + c_name + "/*.pgm")
for d in dirList:
    image = cv2.imread(d)
    H, W = image.shape[:2]

    # create XML
    root = etree.Element('annotation')

    # another child with text
    folder = etree.Element('folder')
    folder.text = 'characters'
    root.append(folder)

    filename = etree.Element('filename')
    filename.text = d.split('/')[-1]
    root.append(filename)

    path = etree.Element('path')
    path.text = '/' + d
    root.append(path)

    source = etree.Element('source')
    database = etree.SubElement(source, "database")
    database.text = 'Unknown'
    root.append(source)

    size = etree.Element('size')
    width = etree.SubElement(size, 'width')
    width.text = str(W)
    height = etree.SubElement(size, 'height')
    height.text = str(H)
    depth = etree.SubElement(size, 'depth')
    depth.text = '3'
    root.append(size)

    segmented = etree.Element('segmented')
    segmented.text = '0'
    root.append(segmented)

    object = etree.Element('object')
    name = etree.SubElement(object, 'name')
    name.text = c_name
    pose = etree.SubElement(object, 'pose')
    pose.text = 'Unspecified'
    truncated = etree.SubElement(object, 'truncated')
    truncated.text = '0'
    difficult = etree.SubElement(object, 'difficult')
    difficult.text = '0'
    root.append(object)

    bndbox = etree.Element('bndbox')
    xmin = etree.SubElement(bndbox, 'xmin')
    xmin.text = '0'
    ymin = etree.SubElement(bndbox, 'ymin')
    ymin.text = '0'
    xmax = etree.SubElement(bndbox, 'xmax')
    xmax.text = str(W)
    ymax = etree.SubElement(bndbox, 'ymax')
    ymax.text = str(H)
    object.append(bndbox)

    # pretty string
    # print(etree.tostring(root, pretty_print=True, encoding='unicode'))
    with open(goal_dir + d.split('/')[-1].split('.')[0] + '.xml', 'w') as file:
        file.write(etree.tostring(root, pretty_print=True, encoding='unicode'))

    print("success")
