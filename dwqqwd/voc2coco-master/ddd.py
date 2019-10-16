import os
import re
import xml.etree.cElementTree as ET


class DirContentToXML:

    def __init__(self, dir_path, ext):
        # target directory
        self.dir_path = '/home/joseph/joseph_data/Dota v1.5/croping/coco/train/xml_h'
        # use on files with this extension only
        self.ext = ext
        self.file_set = set()
        self.grouped_files = dict()

    def create_set_of_files(self):
        self.file_set = {
            f for f in os.listdir(self.dir_path)
            if f.endswith(self.ext)
        }

    def get_core_name(self, regex, name):
        # This is going to be a dict key for a list of similar files
        return re.match(regex, name).group(0)

    def group_files_by_core_name(self, regex):
        self.create_set_of_files()
        for f in self.file_set:
            core_name = self.get_core_name(regex, f)
            if core_name in self.grouped_files:
                self.grouped_files[core_name].append(f)
                # The idea was that alphabeticaly first name is going to be
                # the most important and should be first in xml list
                self.grouped_files[core_name] = sorted(self.grouped_files[core_name])
            else:
                self.grouped_files.update({core_name: [f]})

    def create_xml_tree(self):
        xml_tree = ET.Element('products')
        for id, files in self.grouped_files.items():
            xml_product = ET.SubElement(xml_tree, 'product')
            ET.SubElement(xml_product, 'index').text = id

            xml_images = ET.SubElement(xml_product, 'files')
            for image_file in files:
                ET.SubElement(xml_images, 'file_name').text = image_file

        return xml_tree

    def save_to_xml(self, output_file_name):
        with open(output_file_name, 'wb') as xml_file:
            ET.ElementTree(self.create_xml_tree()).write(xml_file, encoding='utf-8')
