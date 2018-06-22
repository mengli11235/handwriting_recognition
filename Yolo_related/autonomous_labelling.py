import glob
import os
from shutil import copyfile

import cv2
from lxml import etree

# set your working directory
os.chdir('../../../')
goal_dir = './Others/Annotation/'
goal_img_dir = './Others/Characters/'
source_dir = 'Others/monkbrill_171005/'
print('Start process...')

for f in [x[0] for x in os.walk(source_dir)]:
    c_name = f.split('/')[-1]
    print(c_name.replace('-', ''))

    dirList = glob.glob(f + "/*.pgm")
    count = 0
    for d in dirList:
        # copyfile(d, goal_img_dir + c_name.replace('-', '') + str(count) + '.pgm')

        image = cv2.imread(d)
        H, W = image.shape[:2]

        cv2.imwrite(goal_img_dir + c_name.replace('-', '') + str(count) + '.jpg', image)

        # create XML
        root = etree.Element('annotation')

        # another child with text
        folder = etree.Element('folder')
        folder.text = 'Characters'
        root.append(folder)

        filename = etree.Element('filename')
        filename.text = c_name.replace('-', '') + str(count) + '.jpg'
        root.append(filename)

        path = etree.Element('path')
        path.text = '/' + c_name.replace('-', '') + str(count) + '.jpg'
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
        name.text = c_name.replace('-', '')
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
        with open(goal_dir + c_name.replace('-', '') + str(count) + '.xml', 'w') as file:
            file.write(etree.tostring(root, pretty_print=True, encoding='unicode'))

        count += 1
