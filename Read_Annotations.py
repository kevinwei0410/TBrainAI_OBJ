import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import random

img_path = "./Train_Images"
anno_path = "./Train_Annotations"
split_rate = 0.8


classes = [] 
sets = ['train','validation']

def gen_classes(image_id):
    print('%s/Annotations/%s.xml'%(anno_path,image_id))
    in_file = open('%s/%s.xml'%(anno_path,image_id))
    tree = ET.parse(in_file)
    root = tree.getroot()    
    for obj in root.iter('object'):
        cls_name = obj.find('name').text
        if cls_name in classes:
            pass
        else:
            classes.append(cls_name)
    return classes

def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 
    y = (box[2] + box[3])/2.0 
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(image_set,image_id):
    in_file = open('%s/%s.xml'%(anno_path,image_id))
    out_file = open('%s/%s'%(anno_path,image_set)+'_labels/%s.txt'%image_id, 'w')
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        cls = cls.lower()
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

def generate_image_set():
    name_list = os.listdir(img_path)
    random.shuffle(name_list)
    
    train = name_list[:int(len(name_list) * split_rate)]
    validation = name_list[-int(len(name_list) * (1.0 - split_rate)):]
    
    for image_set in sets:
        f = open("./{}.txt".format(image_set), 'w+')
        for name in locals()[image_set]:
            #f.write(name.split('.')[0]+"\n")
            f.write("./Train_Images/"+name+"\n")
        f.close()
    
    
generate_image_set()

for image_set in sets:
    if not os.path.exists('%s/%s'%(anno_path,image_set)+'_labels/'):
        os.makedirs('%s/%s'%(anno_path,image_set)+'_labels/')
    image_ids = open('./{}.txt'.format(image_set)).read().strip().split()
    for image_id in image_ids:
        gen_classes(image_id)
        convert_annotation(image_set,image_id)
    classes_file = open('%s/%s'%(anno_path,image_set)+'_labels/classes.txt','w')
    classes_file.write("\n".join([a for a in classes]))
    classes_file.close()