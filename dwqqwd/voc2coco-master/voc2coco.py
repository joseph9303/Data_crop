# #!/usr/bin/python
#
# # pip install lxml
#
# import sys
# import os
# import json
# # import xml.etree.ElementTree as ET
# #
# #
# # START_BOUNDING_BOX_ID = 1
# # PRE_DEFINE_CATEGORIES = {}
# # If necessary, pre-define category and its id
# PRE_DEFINE_CATEGORIES = {'back_ground': 0,
#         'roundabout': 1,
#         'tennis-court': 2,
#         'swimming-pool': 3,
#         'storage-tank': 4,
#         'soccer-ball-field': 5,
#         'small-vehicle': 6,
#         'ship': 7,
#         'plane': 8,
#         'large-vehicle': 9,
#         'helicopter': 10,
#         'harbor': 11,
#         'ground-track-field': 12,
#         'bridge': 13,
#         'basketball-court': 14,
#         'baseball-diamond': 15
#                          }
#
# #
# def get(root, name):
#     vars = root.findall(name)
#     return vars
#
#
# def get_and_check(root, name, length):
#     vars = root.findall(name)
#     if len(vars) == 0:
#         raise NotImplementedError('Can not find %s in %s.'%(name, root.tag))
#     if length > 0 and len(vars) != length:
#         raise NotImplementedError('The size of %s is supposed to be %d, but is %d.'%(name, length, len(vars)))
#     if length == 1:
#         vars = vars[0]
#     return vars
#
#
# def get_filename_as_int(filename):
#     try:
#         filename = os.path.splitext(filename)[0]
#         return int(filename)
#     except:
#         raise NotImplementedError('Filename %s is supposed to be an integer.'%(filename))
#
#
# def convert(xml_list, xml_dir, json_file):
#     list_fp = open(xml_list, 'r')
#     json_dict = {"images":[], "type": "instances", "annotations": [],
#                  "categories": []}
#     categories = PRE_DEFINE_CATEGORIES
#     bnd_id = START_BOUNDING_BOX_ID
#     for line in list_fp:
#         line = line.strip()
#         print("Processing %s"%(line))
#         xml_f = os.path.join(xml_dir, line)
#         tree = ET.parse(xml_f)
#         root = tree.getroot()
#         path = get(root, '/home/joseph/joseph_data/Dota v1.5/croping/coco/train/xml_h')
#         if len(path) == 1:
#             filename = os.path.basename(path[0].text)
#         elif len(path) == 0:
#             filename = get_and_check(root, 'filename', 1).text
#         else:
#             raise NotImplementedError('%d paths found in %s'%(len(path), line))
#         ## The filename must be a number
#         image_id = get_filename_as_int(filename)
#         size = get_and_check(root, 'size', 1)
#         width = int(get_and_check(size, 'width', 1).text)
#         height = int(get_and_check(size, 'height', 1).text)
#         image = {'file_name': filename, 'height': height, 'width': width,
#                  'id':image_id}
#         json_dict['images'].append(image)
#         ## Cruuently we do not support segmentation
#         #  segmented = get_and_check(root, 'segmented', 1).text
#         #  assert segmented == '0'
#         for obj in get(root, 'object'):
#             category = get_and_check(obj, 'name', 1).text
#             if category not in categories:
#                 new_id = len(categories)
#                 categories[category] = new_id
#             category_id = categories[category]
#             bndbox = get_and_check(obj, 'bndbox', 1)
#             xmin = int(get_and_check(bndbox, 'xmin', 1).text) - 1
#             ymin = int(get_and_check(bndbox, 'ymin', 1).text) - 1
#             xmax = int(get_and_check(bndbox, 'xmax', 1).text)
#             ymax = int(get_and_check(bndbox, 'ymax', 1).text)
#             assert(xmax > xmin)
#             assert(ymax > ymin)
#             o_width = abs(xmax - xmin)
#             o_height = abs(ymax - ymin)
#             ann = {'area': o_width*o_height, 'iscrowd': 0, 'image_id':
#                    image_id, 'bbox':[xmin, ymin, o_width, o_height],
#                    'category_id': category_id, 'id': bnd_id, 'ignore': 0,
#                    'segmentation': []}
#             json_dict['annotations'].append(ann)
#             bnd_id = bnd_id + 1
#
#     for cate, cid in categories.items():
#         cat = {'supercategory': 'none', 'id': cid, 'name': cate}
#         json_dict['categories'].append(cat)
#     json_fp = open(json_file, 'w')
#     json_str = json.dumps(json_dict)
#     json_fp.write(json_str)
#     json_fp.close()
#     list_fp.close()
#
#
# if __name__ == '__main__':
#     if len(sys.argv) <= 1:
#         print('3 auguments are need.')
#         print('Usage: %s XML_LIST.txt XML_DIR OUTPU_JSON.json'%(sys.argv[0]))
#         exit(1)
#
#     convert(, '/home/joseph/joseph_data/Dota v1.5/croping/coco/train/xml_h', '/home/joseph/joseph_data/Dota v1.5/croping/coco/train/root/train.json')
#
# #
# import os
# import xml.etree.ElementTree as ET
# import xmltodict
# import json
# from xml.dom import minidom
# from collections import OrderedDict
# import glob
#
#
# # attrDict = {"images":[{"file_name":[],"height":[], "width":[],"id":[]}], "type":"instances", "annotations":[], "categories":[]}
#
# # xmlfile = "000023.xml"
#
#
# def generateVOC2Json(rootDir):
#     attrDict = dict()
#     # images = dict()
#     # images1 = list()
#     attrDict["categories"] = [{"supercategory": "none", "id": 0, "name": "background"},
#                               {"supercategory": "none", "id": 1, "name": "roundabout"},
#                               {"supercategory": "none", "id": 2, "name": "tennis-court"},
#                               {"supercategory": "none", "id": 3, "name": "swimming-pool"},
#                               {"supercategory": "none", "id": 4, "name": "storage-tank"},
#                               {"supercategory": "none", "id": 5, "name": "soccer-ball-field"},
#                               {"supercategory": "none", "id": 6, "name": "small-vehicle"},
#                               {"supercategory": "none", "id": 7, "name": "ship"},
#                               {"supercategory": "none", "id": 8, "name": "plane"},
#                               {"supercategory": "none", "id": 9, "name": "large-vehicle"},
#                               {"supercategory": "none", "id": 10, "name": "helicopter"},
#                               {"supercategory": "none", "id": 11, "name": "harbor"},
#                               {"supercategory": "none", "id": 12, "name": "ground-track-field"},
#                               {"supercategory": "none", "id": 13, "name": "bridge"},
#                               {"supercategory": "none", "id": 14, "name": "basketball-court"},
#                               {"supercategory": "none", "id": 15, "name": "baseball-diamond"},
#
#                               ]
#     images = list()
#     annotations = list()
#     for root, dirs, files in os.walk(rootDir):
#         image_id = 0
#         for count, xml in enumerate(glob.glob(rootDir + '/*.xml')):
#             # to avoid path error in different development platform
#             xml = xml.replace('\\', '/')
#
#             fileName = xml.split('/')[-1].split('.')[0] + '.xml'
#             xmlFiles = list()
#
#             #fileName = line.strip()
#
#
#             xmlFiles.append(fileName + ".xml")
#
#         for file in xmlFiles:
#             image_id = image_id + 1
#             if file in files:
#
#                 # image_id = image_id + 1
#                 annotation_path = os.path.abspath(os.path.join('/home/joseph/joseph_data/Dota v1.5/croping/coco/train/xml_h'))
#
#                 # tree = ET.parse(annotation_path)#.getroot()
#                 image = dict()
#                 # keyList = list()
#                 doc = xmltodict.parse(open(annotation_path).read())
#                 # print doc['annotation']['filename']
#                 image['file_name'] = str(doc['annotation']['filename'])
#                 # keyList.append("file_name")
#                 image['height'] = int(doc['annotation']['size']['height'])
#                 # keyList.append("height")
#                 image['width'] = int(doc['annotation']['size']['width'])
#                 # keyList.append("width")
#
#                 # image['id'] = str(doc['annotation']['filename']).split('.jpg')[0]
#                 image['id'] = image_id
#                 print
#                 "File Name: {} and image_id {}".format(file, image_id)
#                 images.append(image)
#                 # keyList.append("id")
#                 # for k in keyList:
#                 # 	images1.append(images[k])
#                 # images2 = dict(zip(keyList, images1))
#                 # print images2
#                 # print images
#
#                 # attrDict["images"] = images
#
#                 # print attrDict
#                 # annotation = dict()
#                 id1 = 1
#                 if 'object' in doc['annotation']:
#                     for obj in doc['annotation']['object']:
#                         for value in attrDict["categories"]:
#                             annotation = dict()
#                             # if str(obj['name']) in value["name"]:
#                             if str(obj['name']) == value["name"]:
#                                 # print str(obj['name'])
#                                 # annotation["segmentation"] = []
#                                 annotation["iscrowd"] = 0
#                                 # annotation["image_id"] = str(doc['annotation']['filename']).split('.jpg')[0] #attrDict["images"]["id"]
#                                 annotation["image_id"] = image_id
#                                 x1 = int(obj["bndbox"]["xmin"]) - 1
#                                 y1 = int(obj["bndbox"]["ymin"]) - 1
#                                 x2 = int(obj["bndbox"]["xmax"]) - x1
#                                 y2 = int(obj["bndbox"]["ymax"]) - y1
#                                 annotation["bbox"] = [x1, y1, x2, y2]
#                                 annotation["area"] = float(x2 * y2)
#                                 annotation["category_id"] = value["id"]
#                                 annotation["ignore"] = 0
#                                 annotation["id"] = id1
#                                 annotation["segmentation"] = [
#                                     [x1, y1, x1, (y1 + y2), (x1 + x2), (y1 + y2), (x1 + x2), y1]]
#                                 id1 += 1
#
#                                 annotations.append(annotation)
#
#                 else:
#                     print
#                     "File: {} doesn't have any object".format(file)
#             # image_id = image_id + 1
#
#             else:
#                 print
#                 "File: {} not found".format(file)
#
#     attrDict["images"] = images
#     attrDict["annotations"] = annotations
#     attrDict["type"] = "instances"
#
#     # print attrDict
#     jsonString = json.dumps(attrDict)
#     with open("receipts_valid.json", "w") as f:
#         f.write(jsonString)


# rootDir = "/netscratch/pramanik/OBJECT_DETECTION/detectron/lib/datasets/data/Receipts/Annotations"
# for root, dirs, files in os.walk(rootDir):
# 	for file in files:
# 		if file.endswith(".xml"):
# 			annotation_path = str(os.path.abspath(os.path.join(root,file)))
# 			#print(annotation_path)
# 			generateVOC2Json(annotation_path)
# trainFile = "/home/joseph/joseph_data/Dota v1.5/croping/coco/train/root/root.txt"
# trainXMLFiles = list()
# with open(trainFile, "rb") as f:
#     for line in f:
#         fileName = line.strip()
#         print
#         fileName
#         trainXMLFiles.append(fileName + ".xml")
#
# rootDir = "/home/joseph/joseph_data/Dota v1.5/croping/coco/train/xml_h"
# generateVOC2Json(rootDir)
#
# # -*- coding: utf-8 -*-
# """
# Created on Thu Mar 08 16:01:57 2018
# Convert VOC dataset into COCO dataset
# @author: wkoa
# # """
# import os
# import sys
# import json
# import xml.etree.ElementTree as ET
# import numpy as np
# import cv2
#
#
# def _isArrayLike(obj):
#     return hasattr(obj, '__iter__') and hasattr(obj, '__len__')
#
#
# class voc2coco:
#     def __init__(self, devkit_path=None, year=None):
#         self.classes = ('__background__',  # always index 0
#                         'aeroplane', 'bicycle', 'bird', 'boat',
#                         'bottle', 'bus', 'car', 'cat', 'chair',
#                         'cow', 'diningtable', 'dog', 'horse',
#                         'motorbike', 'person', 'pottedplant',
#                         'sheep', 'sofa', 'train', 'tvmonitor')
#         # self.classes = ('__background__',
#         #                'bottle','box','brush','cabbage','dolphin',
#         #                'eggplant','hedgehog','lion','polarbear',
#         #                'squirrel')
#
#         self.num_classes = len(self.classes)
#         assert 'VOCdevkit' in devkit_path, 'VOCdevkit path does not exist: {}'.format(devkit_path)
#         self.data_path = os.path.join(devkit_path, 'VOC' + year)
#         self.annotaions_path = os.path.join(self.data_path, 'Annotations')
#         self.image_set_path = os.path.join(self.data_path, 'ImageSets')
#         self.year = year
#         self.categories_to_ids_map = self._get_categories_to_ids_map()
#         self.categories_msg = self._categories_msg_generator()
#
#     def _load_annotation(self, ids=[]):
#         """
#         Load annotations by ids
#         :param ids (int array) : get amms for idss
#         :return image_msg
#          return annotation_msg
#         """
#         ids = ids if _isArrayLike(ids) else [ids]
#         image_msg = []
#         annotation_msg = []
#         annotation_id = 1
#         for index in ids:
#             filename = '{:0>6}'.format(index)
#             json_file = os.path.join(self.data_path, 'Segmentation_json', filename + '.json')
#             # Labelme label file .json
#             if os.path.exists(json_file):
#                 img_file = os.path.join(self.data_path, 'JPEGImages', filename + '.jpg')
#                 im = cv2.imread(img_file)
#                 width = im.shape[1]
#                 height = im.shape[0]
#                 seg_data = json.load(open(json_file, 'r'))
#                 assert type(seg_data) == type(dict()), 'annotation file format {} not supported'.format(type(seg_data))
#                 for shape in seg_data['shapes']:
#                     seg_msg = []
#                     for point in shape['points']:
#                         seg_msg += point
#                     one_ann_msg = {"segmentation": [seg_msg],
#                                    "area": self._area_computer(shape['points']),
#                                    "iscrowd": 0,
#                                    "image_id": int(index),
#                                    "bbox": self._points_to_mbr(shape['points']),
#                                    "category_id": self.categories_to_ids_map[shape['label']],
#                                    "id": annotation_id,
#                                    "ignore": 0
#                                    }
#                     annotation_msg.append(one_ann_msg)
#                     annotation_id += 1
#             # LabelImg label file .xml
#             else:
#                 xml_file = os.path.join(self.annotaions_path, filename + '.xml')
#                 tree = ET.parse(xml_file)
#                 size = tree.find('size')
#                 objs = tree.findall('object')
#                 width = size.find('width').text
#                 height = size.find('height').text
#                 for obj in objs:
#                     bndbox = obj.find('bndbox')
#                     [xmin, xmax, ymin, ymax] \
#                         = [int(bndbox.find('xmin').text) - 1, int(bndbox.find('xmax').text),
#                            int(bndbox.find('ymin').text) - 1, int(bndbox.find('ymax').text)]
#                     if xmin < 0:
#                         xmin = 0
#                     if ymin < 0:
#                         ymin = 0
#                     bbox = [xmin, xmax, ymin, ymax]
#                     one_ann_msg = {"segmentation": self._bbox_to_mask(bbox),
#                                    "area": self._bbox_area_computer(bbox),
#                                    "iscrowd": 0,
#                                    "image_id": int(index),
#                                    "bbox": [xmin, ymin, xmax - xmin, ymax - ymin],
#                                    "category_id": self.categories_to_ids_map[obj.find('name').text],
#                                    "id": annotation_id,
#                                    "ignore": 0
#                                    }
#                     annotation_msg.append(one_ann_msg)
#                     annotation_id += 1
#             one_image_msg = {"file_name": filename + ".jpg",
#                              "height": int(height),
#                              "width": int(width),
#                              "id": int(index)
#                              }
#             image_msg.append(one_image_msg)
#         return image_msg, annotation_msg
#
#     def _bbox_to_mask(self, bbox):
#         """"
#         Generate mask by bbox
#         :param bbox e.g. [xmin,xmax,ymin,ymax]
#         :return mask [points]
#         """
#         assert len(bbox) == 4, 'Wrong bndbox!'
#         mask = [bbox[0], bbox[2], bbox[0], bbox[3], bbox[1], bbox[3], bbox[1], bbox[2]]
#         return [mask]
#
#     def _bbox_area_computer(self, bbox):
#         """
#         Area computer
#         """
#         width = bbox[1] - bbox[0]
#         height = bbox[3] - bbox[2]
#         return width * height
#
#     def _save_json_file(self, filename=None, data=None):
#         """
#         Save result in json
#         :param filename (str) : name of json file
#          param data           : coco format data
#         :return
#         """
#         json_path = os.path.join(self.data_path, 'cocoformatJson')
#         assert filename is not None, 'lack filename'
#         if os.path.exists(json_path) == False:
#             os.mkdir(json_path)
#         if not filename.endswith('.json'):
#             filename += '.json'
#         assert type(data) == type(dict()), 'data format {} not supported'.format(type(data))
#         with open(os.path.join(json_path, filename), 'w') as f:
#             f.write(json.dumps(data))
#
#     def _get_categories_to_ids_map(self):
#         """
#         Generate categories to ids map
#         """
#         return dict(zip(self.classes, xrange(self.num_classes)))
#
#     def _get_all_indexs(self):
#         """
#         Get all images and annotations indexs
#         :param
#         :return ids (str array)
#         """
#         ids = []
#         for root, dirs, files in os.walk(self.annotaions_path, topdown=False):
#             for f in files:
#                 if str(f).endswith('.xml'):
#                     id = int(str(f).strip('.xml'))
#                     ids.append(id)
#         assert ids is not None, 'There is none xml file in {}'.format(self.annotaions_path)
#         return ids
#
#     def _get_indexs_by_image_set(self, image_set=None):
#         """
#         Get images and nnotations indexs in image_set
#         """
#         if image_set is None:
#             return self._get_all_indexs()
#         else:
#             image_set_path = os.path.join(self.image_set_path, 'Main', image_set + '.txt')
#             assert os.path.exists(image_set_path), 'Path does not exist: {}'.format(image_set_path)
#             with open(image_set_path) as f:
#                 ids = [x.strip() for x in f.readlines()]
#             return ids
#
#     def _points_to_mbr(self, points):
#         """
#         Transfer points to min bounding rectangle
#         :param: points (a list of lists)
#         :return: [x,y,width,height]
#         """
#         assert _isArrayLike(points), 'Points should be array like!'
#         x = [point[0] for point in points]
#         y = [point[1] for point in points]
#         assert len(x) == len(y), 'Wrong point quantity'
#         xmin, xmax, ymin, ymax = min(x), max(x), min(y), max(y)
#         height = ymax - ymin
#         width = xmax - xmin
#         return [xmin, ymin, width, height]
#
#     def _categories_msg_generator(self):
#         categories_msg = []
#         for category in self.classes:
#             if category == '__background__':
#                 continue
#             one_categories_msg = {"supercategory": "none",
#                                   "id": self.categories_to_ids_map[category],
#                                   "name": category
#                                   }
#             categories_msg.append(one_categories_msg)
#         return categories_msg
#
#     def _area_computer(self, points):
#         """
#         :param: one shape's points (int array array)
#         :return: shape's area
#         """
#         assert _isArrayLike(points), 'Points should be array like!'
#         tmp_contour = []
#         for point in points:
#             tmp_contour.append([point])
#         contour = np.array(tmp_contour, dtype=np.int32)
#         area = cv2.contourArea(contour)
#         return area
#
#     def voc_to_coco_converter(self):
#         """
#         Convert voc dataset to coco dataset
#         """
#         img_sets = ['trainval', 'test']
#
#         for img_set in img_sets:
#             ids = self._get_indexs_by_image_set(img_set)
#             img_msg, ann_msg = self._load_annotation(ids)
#             result_json = {"images": img_msg,
#                            "type": "instances",
#                            "annotations": ann_msg,
#                            "categories": self.categories_msg}
#             self._save_json_file('voc_' + self.year + '_' + img_set, result_json)
#
#
# def demo():
#     converter = voc2coco('D:\\MyCAS\\VOC2007\\VOCdevkit', '2007')
#     converter.voc_to_coco_converter()
#
#
# if __name__ == "__main__":
#     demo()
#
# import xml.etree.ElementTree as ET
# import os
# import json
#
# coco = dict()
# coco['images'] = []
# coco['type'] = 'instances'
# coco['annotations'] = []
# coco['categories'] = []
#
# category_set = dict()
# image_set = set()
#
# category_item_id = -1
# image_id = 20180000000
# annotation_id = 0
#
#
# def addCatItem(name):
#     global category_item_id
#     category_item = dict()
#     category_item['supercategory'] = 'none'
#     category_item_id += 1
#     category_item['id'] = category_item_id
#     category_item['name'] = name
#     coco['categories'].append(category_item)
#     category_set[name] = category_item_id
#     return category_item_id
#
#
# def addImgItem(file_name, size):
#     global image_id
#     if file_name is None:
#         raise Exception('Could not find filename tag in xml file.')
#     if size['width'] is None:
#         raise Exception('Could not find width tag in xml file.')
#     if size['height'] is None:
#         raise Exception('Could not find height tag in xml file.')
#     image_id += 1
#     image_item = dict()
#     image_item['id'] = image_id
#     image_item['file_name'] = file_name
#     image_item['width'] = size['width']
#     image_item['height'] = size['height']
#     coco['images'].append(image_item)
#     image_set.add(file_name)
#     return image_id
#
#
# def addAnnoItem(object_name, image_id, category_id, bbox):
#     global annotation_id
#     annotation_item = dict()
#     annotation_item['segmentation'] = []
#     seg = []
#     # bbox[] is x,y,w,h
#     # left_top
#     seg.append(bbox[0])
#     seg.append(bbox[1])
#     # left_bottom
#     seg.append(bbox[0])
#     seg.append(bbox[1] + bbox[3])
#     # right_bottom
#     seg.append(bbox[0] + bbox[2])
#     seg.append(bbox[1] + bbox[3])
#     # right_top
#     seg.append(bbox[0] + bbox[2])
#     seg.append(bbox[1])
#
#     annotation_item['segmentation'].append(seg)
#
#     annotation_item['area'] = bbox[2] * bbox[3]
#     annotation_item['iscrowd'] = 0
#     annotation_item['ignore'] = 0
#     annotation_item['image_id'] = image_id
#     annotation_item['bbox'] = bbox
#     annotation_item['category_id'] = category_id
#     annotation_id += 1
#     annotation_item['id'] = annotation_id
#     coco['annotations'].append(annotation_item)
#
#
# def parseXmlFiles(xml_path):
#     for f in os.listdir(xml_path):
#         if not f.endswith('.xml'):
#             continue
#
#         bndbox = dict()
#         size = dict()
#         current_image_id = None
#         current_category_id = None
#         file_name = None
#         size['width'] = None
#         size['height'] = None
#         size['depth'] = None
#
#         xml_file = os.path.join(xml_path, f)
#         print(xml_file)
#
#         tree = ET.parse(xml_file)
#         root = tree.getroot()
#         if root.tag != 'annotation':
#             raise Exception('pascal voc xml root element should be annotation, rather than {}'.format(root.tag))
#
#         # elem is <folder>, <filename>, <size>, <object>
#         for elem in root:
#             current_parent = elem.tag
#             current_sub = None
#             object_name = None
#
#             if elem.tag == 'folder':
#                 continue
#
#             if elem.tag == 'filename':
#                 file_name = elem.text
#                 if file_name in category_set:
#                     raise Exception('file_name duplicated')
#
#             # add img item only after parse <size> tag
#             elif current_image_id is None and file_name is not None and size['width'] is not None:
#                 if file_name not in image_set:
#                     current_image_id = addImgItem(file_name, size)
#                     print('add image with {} and {}'.format(file_name, size))
#                 else:
#                     raise Exception('duplicated image: {}'.format(file_name))
#                     # subelem is <width>, <height>, <depth>, <name>, <bndbox>
#             for subelem in elem:
#                 bndbox['xmin'] = None
#                 bndbox['xmax'] = None
#                 bndbox['ymin'] = None
#                 bndbox['ymax'] = None
#
#                 current_sub = subelem.tag
#                 if current_parent == 'object' and subelem.tag == 'name':
#                     object_name = subelem.text
#                     if object_name not in category_set:
#                         current_category_id = addCatItem(object_name)
#                     else:
#                         current_category_id = category_set[object_name]
#
#                 elif current_parent == 'size':
#                     if size[subelem.tag] is not None:
#                         raise Exception('xml structure broken at size tag.')
#                     size[subelem.tag] = int(subelem.text)
#
#                 # option is <xmin>, <ymin>, <xmax>, <ymax>, when subelem is <bndbox>
#                 for option in subelem:
#                     if current_sub == 'bndbox':
#                         if bndbox[option.tag] is not None:
#                             raise Exception('xml structure corrupted at bndbox tag.')
#                         bndbox[option.tag] = float(option.text)
#
#                 # only after parse the <object> tag
#                 if bndbox['xmin'] is not None:
#                     if object_name is None:
#                         raise Exception('xml structure broken at bndbox tag')
#                     if current_image_id is None:
#                         raise Exception('xml structure broken at bndbox tag')
#                     if current_category_id is None:
#                         raise Exception('xml structure broken at bndbox tag')
#                     bbox = []
#                     # x
#                     bbox.append(bndbox['xmin'])
#                     # y
#                     bbox.append(bndbox['ymin'])
#                     # w
#                     bbox.append(bndbox['xmax'] - bndbox['xmin'])
#                     # h
#                     bbox.append(bndbox['ymax'] - bndbox['ymin'])
#                     print('add annotation with {},{},{},{}'.format(object_name, current_image_id, current_category_id,
#                                                                    bbox))
#                     addAnnoItem(object_name, current_image_id, current_category_id, bbox)
#
#
# if __name__ == '__main__':
#     xml_path = '/home/joseph/joseph_data/Dota v1.5/croping/coco/train/xml_h'  # where is the xml
#     json_file = '/home/joseph/joseph_data/Dota v1.5/croping/coco/train/train.json'  # create a empty .json
#     parseXmlFiles(xml_path)  #
#     json.dump(coco, open(json_file, 'w'))


import json
import xml.etree.ElementTree as ET
import os

def load_load_image_labels(LABEL_PATH, class_name=[]):
    # temp=[]
    images=[]
    type="instances"
    annotations=[]
    #assign your categories which contain the classname and calss id
    #the order must be same as the class_nmae
    categories = [

        {
            "id": 1,
            "name": "roundabout",
            "supercategory": "roundabout"
        },
        {
            "id": 2,
            "name": "tennis-court",
            "supercategory": "tennis-court"
        },
        {
            "id": 3,
            "name": "swimming-pool",
            "supercategory": "swimming-pool"
        },
        {
            "id": 4,
            "name": "storage-tank",
            "supercategory": "storage-tank"
        },
        {
            "id": 5,
            "name": "soccer-ball-field",
            "supercategory": "soccer-ball-field"
        },
        {
            "id": 6,
            "name": "small-vehicle",
            "supercategory": "small-vehicle"
        },
        {
            "id": 7,
            "name": "ship",
            "supercategory": "ship"
        },
        {
            "id": 8,
            "name": "plane",
            "supercategory": "plane"
        },
        {
            "id": 9,
            "name": "large-vehicle",
            "supercategory": "large-vehicle"
        },
        {
            "id": 10,
            "name": "helicopter",
            "supercategory": "helicopter"
        },
        {
            "id": 11,
            "name": "harbor",
            "supercategory": "harbor"
        },
        {
            "id": 12,
            "name": "ground-track-field",
            "supercategory": "ground-track-field"
        },
        {
            "id": 13,
            "name": "bridge",
            "supercategory": "bridge"
        },
        {
            "id": 14,
            "name": "basketball-court",
            "supercategory": "basketball-court"
        },
        {
            "id": 15,
            "name": "baseball-diamond",
            "supercategory": "baseball-diamond"
        }
	]
    # load ground-truth from xml annotations
    id_number=0
    for image_id, label_file_name in enumerate(os.listdir(LABEL_PATH)):
        print(str(image_id)+' '+label_file_name)
        label_file=LABEL_PATH + label_file_name
        image_file = label_file_name.split('.')[0] + '.png'
        tree = ET.parse(label_file)
        root = tree.getroot()

        size=root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)

        images.append({
            "file_name": image_file,
			"height": height,
			"width": width,
			"id": image_id
		})# id of the image. referenced in the annotation "image_id"

        for anno_id, obj in enumerate(root.iter('object')):
            name = obj.find('name').text
            bbox=obj.find('bndbox')
            cls_id = class_name.index(name)
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            xlen = abs(xmax-xmin)
            ylen = abs(ymax-ymin)
            annotations.append({
                                "segmentation" : [[xmin, ymin, xmin, ymax, xmax, ymax, xmax, ymin],],
                                "area" : xlen*ylen,
                                "iscrowd": 0,
                                "image_id": image_id,
                                "bbox" : [xmin, ymin, xlen, ylen],
                                "category_id": cls_id,
                                "id": id_number,
                                "ignore":0
                                })
            # print([image_file,image_id, cls_id, xmin, ymin, xlen, ylen])
            id_number += 1

    return {"images":images,"annotations":annotations,"categories":categories}

if __name__=='__main__':
    LABEL_PATH='/home/joseph/joseph_data/dota_coco/val/xml/'
    classes=[

        'roundabout',
        'tennis-court',
        'swimming-pool',
        'storage-tank',
        'soccer-ball-field',
        'small-vehicle',
        'ship',
        'plane',
        'large-vehicle',
        'helicopter',
        'harbor',
        'ground-track-field',
        'bridge',
        'basketball-court',
        'baseball-diamond']

    label_dict = load_load_image_labels(LABEL_PATH, classes)
    jsonfile='/home/joseph/joseph_data/dota_coco/val/dota_val.json'#location where you would like to save the coco format annotations
    with open('/home/joseph/joseph_data/dota_coco/val/dota_val.json','w') as json_file:
        json_file.write(json.dumps(label_dict, ensure_ascii=False))
        json_file.close()