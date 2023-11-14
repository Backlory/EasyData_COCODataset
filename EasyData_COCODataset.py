'''
Author: backlory's desktop dbdx_liyaning@126.com
Date: 2023-11-14 13:58:36
LastEditors: backlory's desktop dbdx_liyaning@126.com
LastEditTime: 2023-11-14 16:03:44
Description: 

用于在图像分割任务中解析“EasyData平台”导出的COCO格式数据集。

Copyright (c) 2023 by Backlory, (email: dbdx_liyaning@126.com), All Rights Reserved.
'''
import os
import cv2
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import json
import pycocotools.mask as maskUtils

class EasyData_COCODataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        # 按名称排序
        self.images_dir = os.path.join(root_dir, 'images')
        self.image_files = os.listdir(self.images_dir)
        self.image_files = sorted(self.image_files, key=lambda x: int(os.path.splitext(x)[0]))
        #
        self.annotations_dir = os.path.join(root_dir, 'annotations')
        self.annotations_file = os.path.join(self.annotations_dir, 'coco_info.json')
        with open(self.annotations_file, 'r', encoding='utf8') as f:
            self.annotations = json.load(f)
        
        # 标签类别
        self.categories = self.annotations['categories']
        
        # 记录文件名到id的映射
        self.image_info = self.annotations['images']
        self.img2id = {}
        for img_info in self.image_info:
            self.img2id[img_info['file_name']] = img_info['id']

        # 获取每张图片对应的标注信息
        self.annotations_info = self.annotations['annotations']
        self.image_id_to_annotation_ids = {}
        for annotation_info in self.annotations_info:
            image_id = annotation_info['image_id']
            if image_id not in self.image_id_to_annotation_ids:
                self.image_id_to_annotation_ids[image_id] = []
            self.image_id_to_annotation_ids[image_id].append(annotation_info['id'])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 拿到图像
        image_file = self.image_files[idx]
        image_path = os.path.join(self.images_dir, image_file)
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        height, width = image.shape[:2]
        
        # 拿到mask，但不一定有
        if image_file not in self.img2id:
            return image, None, None
        else:
            image_id = self.img2id[image_file]
            annotation_ids = self.image_id_to_annotation_ids[image_id]
            masks = []
            categories = []
            for annotation_id in annotation_ids:
                annotation_info = self.annotations_info[annotation_id - 1]
                # 类别
                category_id = annotation_info['category_id']
                category = self.categories[category_id - 1]['name']
                categories.append(category)
                # mask
                rle_obj = {"counts": annotation_info['mask'],
                        "size": [height, width]}
                mask = maskUtils.decode(rle_obj)
                masks.append(mask)
            masks = np.stack(masks, axis=0)
            categories = np.array(categories)
            return image, masks, categories


# 测试
if __name__ == "__main__":
    dataset = EasyData_COCODataset('1970649_1699932428')
    print(len(dataset))
    #data = dataset[26]
    for idx1, data in enumerate(dataset):
        img, masks, categories = data
        print(img.shape)
        if masks is not None:
            print("="*50)
            img_show = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            for idx, mask in enumerate(masks):
                mask = mask * 255
                mask = mask.astype(np.uint8)
                mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                img_show = np.concatenate([img_show, mask], axis=1)
            cv2.imshow('img_'+str(idx1), img_show)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
