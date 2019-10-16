import unittest
import os
import xml.etree.cElementTree as ET
from file_list_to_xml import DirContentToXML

class TestDirContentToXML(unittest.TestCase):
    
    def setUp(self):
        self.dir_path = '/home/joseph/joseph_data/Dota v1.5/croping/coco/train/xml_h'
        self.ext = '.xml'
        self.xpi = DirContentToXML(self.dir_path, self.ext)
        self.output_file_name = 'train_xml.xml'

    def test_class_exists(self):
        self.assertTrue(isinstance(self.xpi, DirContentToXML), True)
        
    def test_set_of_files_in_dir_path(self):
        # tworzy zbiór unikatowych nazw plików we wskazanej lokalizacji
        os_set_dir = {f for f in os.listdir(self.dir_path) if f.endswith(self.ext)}
        self.xpi.create_set_of_files()
        self.assertEqual(self.xpi.file_set, os_set_dir)

    def test_get_core_name(self):
        # znajduje główna cześć pliku wg wskazanego wyszukania
        # np. wszystko do pierwszego zabronionego znaku
        # w tym wypadku do jakiejkolwiel małej litery
        regex = '.*[^a-z ^. ]'
        self.assertEqual(self.xpi.get_core_name(regex, '01021-01a.jpg'), '01021-01')

    # def test_group_file_by_main_name(self):
    #     # tworzy słownik wszystkich plików pasujących do
    #     # głównej nazwy pliku wg. get_core_name
    #     test_data = {'01022-01' : ['01022-01.jpg', '01022-01b.jpg']}
    #     self.xpi.create_set_of_files()
    #     regex = '.*[^a-z ^. ]'
    #     self.xpi.group_files_by_core_name(regex)
    #     self.assertTrue(self.xpi.grouped_files != {})
    #     self.assertListEqual(self.xpi.grouped_files['01022-01'], test_data['01022-01'], msg=self.xpi.grouped_files)

    # def test_create_xml_tree(self):
    #     # testuje strukturę xml
    #     self.xpi.create_set_of_files()
    #     regex = '.*[^a-z ^. ]'
    #     self.xpi.group_files_by_core_name(regex)
    #     # sprawdza nazwę root'a
    #     xml_tree = self.xpi.create_xml_tree()
    #     self.assertEqual(xml_tree.tag, 'products')
    #     # sprawdza nazwę tagów dla pierwszego pliku dla którego tworzona jest struktura xml
    #     self.assertEqual(xml_tree[0].tag, 'product')
    #     self.assertEqual(xml_tree[0][0].tag, 'index')
    #     self.assertEqual(xml_tree[0][1].tag, 'files')
    #     self.assertEqual(xml_tree[0][1][0].tag, 'file_name')
    #     self.assertEqual(xml_tree[0][1][1].tag, 'file_name')


    def test_save_to_xml_file(self):
        # testuje strukturę xml
        self.xpi.create_set_of_files()
        # regex dla szukanej nazwy głównej
        regex = '.*[^a-z ^. ]'
        self.xpi.group_files_by_core_name(regex)
        self.xpi.save_to_xml(self.output_file_name)
        self.assertTrue(
            os.path.exists(self.output_file_name), 
            msg='{} plik został utworzony'.format(self.output_file_name)
            )

    def test_check_if_all_data_is_in_xml(self):
        self.xpi.create_set_of_files()
        # regex dla szukanej nazwy głównej
        regex = '.*[^a-z ^. ]'
        self.xpi.group_files_by_core_name(regex)

        xml_file = ET.parse(os.path.join(os.getcwd(), self.output_file_name))
        root = xml_file.getroot()
        for product in root.findall('product'):
            index = product.find('index').text
            self.assertIn(index, self.xpi.grouped_files)

            files = product.find('files').findall('file_name')
            self.assertListEqual([f.text for f in files], self.xpi.grouped_files[index])



def main():
    unittest.main()

if __name__ == '__main__':
    main()