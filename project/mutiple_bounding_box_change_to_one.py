import os
import xml.etree.ElementTree as ET
from PIL import Image
import argparse

def split_annotations_and_images(root_path, output_folder):
    image_folder = os.path.join(root_path, "JPEGImages")
    ann_folder = os.path.join(root_path, "Annotations")
    train_sample_list = os.path.join(root_path, "train.txt")
    with open(train_sample_list, 'r') as f:
        image_ids = f.read().strip().split()
    
    new_image_folder = os.path.join(output_folder, "JPEGImages")
    new_ann_folder = os.path.join(output_folder, "Annotations")
    new_image_ids = []

    os.makedirs(new_image_folder, exist_ok=True)
    os.makedirs(new_ann_folder, exist_ok=True)
    
    for img_id in image_ids:
        ann_path = os.path.join(ann_folder, f"{img_id}.xml")
        img_path = os.path.join(image_folder, f"{img_id}.jpg")
        
        img = Image.open(img_path).convert("RGB")
        tree = ET.parse(ann_path)
        root = tree.getroot()
        
        for i, obj in enumerate(root.findall("object")):

            label = obj.find("name").text
            if label in ["mining waste", "industry waste"]:
                continue

            new_img_id = f"{img_id}_{i}"
            new_image_ids.append(new_img_id)
            
            new_img_path = os.path.join(new_image_folder, f"{new_img_id}.jpg")
            img.save(new_img_path)

            new_ann_path = os.path.join(new_ann_folder, f"{new_img_id}.xml")
            new_root = ET.Element("annotation")
        
            for elem in root:
                if elem.tag != "object":
                    new_root.append(elem)
            
            new_object = ET.SubElement(new_root, "object")
            for subelem in obj:
                new_object.append(subelem)

            tree = ET.ElementTree(new_root)
            tree.write(new_ann_path)

    new_train_txt = os.path.join(out_path, "train.txt")
    with open(new_train_txt, 'w') as f:
        for img_id in new_image_ids:
            f.write(f"{img_id}\n")


# root_path = 'E:\COMP\COMP9444\project\dumpsite_data\VOC2012\\train'
# out_path = "E:\COMP\COMP9444\project\\new_dunmpsite_data"

def main():
    parser = argparse.ArgumentParser(description="Split multiple bounding boxes to multiple targets whose only have one bounding box and label")
    parser.add_argument("root_path", type=str, help="Path to the root directory of the VOC dataset")
    parser.add_argument("output_folder", type=str, help="Path to the output directory")
    args = parser.parse_args()
    split_annotations_and_images(args.root_path, args.output_folder)

if __name__ == "__main__":
    main()
