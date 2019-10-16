# FileListToXML
Creates XML file from files in target directory.
Small class for creating a simple xml file, which contains a list of files in target directory.
This class was designed to be used on a directory with files of similar names. exp: 01021-01.jpg, 01021-01a.jpg, 01021-01b.jpg

# Xml structure:

```xml
<products>
  <product>
    <index> (core part of similar files names; 01021-01)</index>
    <files>
      <file_name>(one nod for every file containing core in name)</file_name>
      ...
    </files>
  </produkt>
  ...
</products>
