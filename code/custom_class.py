import os
import cv2
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
from PIL.Image import Image as PilImage
import glob
import json
import torch
import shutil
import skimage.feature as feature
from sklearn.model_selection import train_test_split
from utils2 import *


class custom_class:
    def __init__(self, data_folder="./data/Original/"):
        """
        Initiate with folders containing the original data lists
        """
        self.original_folder = data_folder + "Train/"
        self.original_files = self.get_files(self.original_folder)
        self.dotted_folder = data_folder + "TrainDotted/"
        self.dotted_files = self.get_files(self.dotted_folder)

    def get_files(self, data_folder):
        """
        Get training file names, with and without dots
        """
        folder = data_folder
        files = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if (
                os.path.isfile(os.path.join(folder, f))
                and "csv" not in f
                and "json" not in f
            )
        ]
        files = sorted(files)
        return files

    def downsample_images(self, source_folder, destination_folder, dotted=False):
        """
        Downsample images to make image training more manageable
        """
        files = glob.glob(source_folder + "*.jpg")
        i = 0
        for file in files:
            if i % 100 == 0:
                print(i / len(files))
            if "csv" not in file and "json" not in file:
                im = Image.open(file)
                # image size
                size = (int(im.size[0] / 4), int(im.size[1] / 4))
                # resize image
                out = im.resize(size, resample=2)
                # save resized image
                out.save(destination_folder + file.split("/")[-1])
                i = i + 1
        if dotted == False:
            self.ds_original_files = self.get_files(destination_folder)
            self.ds_original_folder = destination_folder
        else:
            self.ds_dotted_files = self.get_files(destination_folder)
            self.ds__dotted_folder = destination_folder

    def image_differences(self, destination_folder):
        """
        Create a full-resolution intermediary image containing the downsampled_differences
        between images with and without dots. Doing the same from downsampled images
        leads to inaccurate differences resulting from pixel interpolation.
        """
        folder = self.original_folder
        files = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if (
                os.path.isfile(os.path.join(folder, f))
                and "csv" not in f
                and "json" not in f
            )
        ]
        files = sorted(files)
        folderdotted = self.dotted_folder
        filesdotted = [
            os.path.join(folderdotted, f)
            for f in os.listdir(folderdotted)
            if (
                os.path.isfile(os.path.join(folderdotted, f))
                and "csv" not in f
                and "json" not in f
            )
        ]
        filesdotted = sorted(filesdotted)
        classes = ["category_1", "category_2", "category_3", "category_4", "category_5"]
        # dataframe to store results in
        count_df = pd.DataFrame(index=files, columns=classes).fillna(0)
        blob_list = []
        for i in tqdm(range(len(files))):
            # read the Train and Train Dotted images
            image_1 = cv2.imread(files[i])
            image_2 = cv2.imread(filesdotted[i])
            if image_1.shape != image_2.shape:
                image_2 = cv2.resize(image_2, (image_2.shape[1], image_2.shape[0]))
            # absolute difference between Train and Train Dotted
            image_3 = cv2.absdiff(image_1, image_2)

            # mask out blackened regions from Train Dotted
            mask_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
            mask_1[mask_1 < 20] = 0
            mask_1[mask_1 > 0] = 255

            mask_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)
            mask_2[mask_2 < 20] = 0
            mask_2[mask_2 > 0] = 255

            image_4 = cv2.bitwise_or(image_3, image_3, mask=mask_1)
            image_5 = cv2.bitwise_or(image_4, image_4, mask=mask_2)

            # convert to grayscale to be accepted by skimage.feature.blob_log
            image_6 = cv2.cvtColor(image_5, cv2.COLOR_BGR2GRAY)
            image_7 = cv2.resize(
                image_6, (int(image_6.shape[0] / 4), int(image_6.shape[1]))
            )
            cv2.imwrite(destination_folder + files[i].split("/")[-1], image_7)

    def blob_detection(self, blob_folder):
        """
        Create a list containing the location and files of blobs (the differences)
        and write them to a tensorflow-format, itself converted to a coco format
        compatible with Detectron2 use. Downsampled images are not recommended as
        down sampled images change the color of the dots used for seal detection.
        """
        folder_blobs = blob_folder
        files_blobs = [
            os.path.join(folder_blobs, f)
            for f in os.listdir(folder_blobs)
            if (
                os.path.isfile(os.path.join(folder_blobs, f))
                and "csv" not in f
                and "json" not in f
            )
        ]
        files_blobs = sorted(files_blobs)
        classes = ["category_1", "category_2", "category_3", "category_4", "category_5"]
        # dataframe to store results in
        count_df = pd.DataFrame(index=files_blobs, columns=classes).fillna(0)
        blob_list = []
        for i in tqdm(range(len(files_blobs))):

            # read the Train and Train Dotted images
            image_1 = cv2.imread(files_blobs[i])

            # detect blobs
            blobs = feature.blob_log(
                image_1, min_sigma=3, max_sigma=4, num_sigma=1, threshold=0.02
            )

            # prepare the image to plot the results on
            # image_7 = cv2.cvtColor(image_6, cv2.COLOR_GRAY2BGR)
            blob_type = []
            for blob in blobs:
                # get the coordinates for each blob
                y, x, s, d = blob
                # get the color of the pixel from Train Dotted in the center of the blob
                b, g, r = image_1[int(y)][int(x)][:]
                # decision tree to pick the class of the blob by looking at the color in Train Dotted
                if r > 190 and b < 40 and g < 55:
                    count_df["category_1"][files_blobs[i]] += 1
                    blob_type.append("category_1")
                elif r > 180 and b > 180 and g < 40:
                    count_df["category_2"][files_blobs[i]] += 1
                    blob_type.append("category_2")
                elif r < 70 and b < 70 and 160 < g < 180:
                    count_df["category_3"][files_blobs[i]] += 1
                    blob_type.append("category_3")
                elif r < 80 and 80 < b and g < 80:
                    count_df["category_4"][files_blobs[i]] += 1
                    blob_type.append("category_4")
                else:
                    count_df["category_5"][files_blobs[i]] += 1
                    blob_type.append("category_5")
            blob_list.append([blobs, blob_type, files_blobs[i]])

        img_bb_list = []
        for blobs in blob_list:
            bb_list = []
            i = 0
            for blob in blobs[0]:
                y, x, s, d = blob
                if blobs[1][i] == "category_1":
                    size = 15
                elif blobs[1][i] == "category_2":
                    size = 13
                elif blobs[1][i] == "category_3":
                    size = 5
                elif blobs[1][i] == "category_4":
                    size = 10
                elif blobs[1][i] == "category_5":
                    size = 10
                y_min = np.int(np.floor(y - (size) * s))
                y_max = np.int(np.floor(y + (size) * s))
                x_min = np.int(np.floor(x - (size) * s))
                x_max = np.int(np.floor(x + (size) * s))
                bb_list.append([y_min, y_max, x_min, x_max])
                i = i + 1
            img_bb_list.append(bb_list)
        self.blobs = img_bb_list
        self.c2tensorflow(blob_list, img_bb_list)

    def c2tensorflow(self, blob_list, img_bb_list, destination_folder=""):
        """
        Convert blob list with annotations to a tensorflow format
        """
        print("Convert to TS format")
        temp = []
        for i in range(len(blob_list)):
            img_shape = plt.imread(self.ds_original_files[i]).shape
            for j in range(len(img_bb_list[i])):
                temp.append(
                    np.array(
                        [
                            self.ds_original_files[i],
                            blob_list[i][1][j],
                            img_shape[1],
                            img_shape[0],
                            img_bb_list[i][j][2],
                            img_bb_list[i][j][0],
                            img_bb_list[i][j][3],
                            img_bb_list[i][j][1],
                        ]
                    )
                )

        file_data = pd.DataFrame(
            temp,
            columns=[
                "filename",
                "class",
                "width",
                "height",
                "xmin",
                "ymin",
                "xmax",
                "ymax",
            ],
        )
        file_data.to_csv(destination_folder + "tensorflow.csv")
        self.c2coco(destination_folder + "tensorflow.csv")

    def c2coco(self, file_path):
        """
        Convert tensorflow annotation format to coco format
        """
        print("Convert to coco format")
        save_json_path = "annotations.json"
        data = pd.read_csv("tensorflow.csv")

        images = []
        categories = []
        annotations = []

        category = {}
        category["supercategory"] = "None"
        category["id"] = 0
        category["name"] = "None"
        categories.append(category)

        data["fileid"] = data["filename"].astype("category").cat.codes
        data["categoryid"] = pd.Categorical(data["class"], ordered=True).codes
        data["categoryid"] = data["categoryid"] + 1
        data["annid"] = data.index

        def image(row):
            image = {}
            image["height"] = row.height
            image["width"] = row.width
            image["id"] = row.fileid
            image["file_name"] = row.filename
            return image

        def category(row):
            category = {}
            category["supercategory"] = "Seal"
            category["id"] = row.categoryid
            category["name"] = row[3]
            return category

        def annotation(row):
            annotation = {}
            area = (row.xmax - row.xmin) * (row.ymax - row.ymin)
            annotation["segmentation"] = []
            annotation["iscrowd"] = 0
            annotation["area"] = area
            annotation["image_id"] = row.fileid
            annotation["bbox"] = [
                row.xmin,
                row.ymin,
                row.xmax - row.xmin,
                row.ymax - row.ymin,
            ]
            annotation["category_id"] = row.categoryid
            annotation["id"] = row.annid
            return annotation

        for row in data.itertuples():
            annotations.append(annotation(row))

        imagedf = data.drop_duplicates(subset=["fileid"]).sort_values(by="fileid")
        for row in imagedf.itertuples():
            images.append(image(row))

        catdf = data.drop_duplicates(subset=["categoryid"]).sort_values(by="categoryid")
        for row in catdf.itertuples():
            row
            categories.append(category(row))

        data_coco = {}
        data_coco["images"] = images
        data_coco["categories"] = categories
        data_coco["annotations"] = annotations
        data_coco["info"] = {}
        data_coco["licenses"] = {}
        json.dump(data_coco, open(save_json_path, "w"), indent=4)
        self.coco2yolo(
            output_path="./data/Downsampled/Train/", file_path="annotations.json"
        )

    def coco2yolo(self, output_path, file_path):
        """
        Convert the yolo annotations to coco
        """
        with open(file_path) as f:
            json_data = json.load(f)

        # write _darknet.labels, which holds names of all classes (one class per line)
        label_file = os.path.join(output_path, "_darknet.labels")
        with open(label_file, "w") as f:
            for category in tqdm(json_data["categories"], desc="Categories"):
                category_name = category["name"]
                f.write(f"{category_name}\n")

        for image in tqdm(json_data["images"], desc="Annotation txt for each image"):
            img_id = image["id"]
            img_name = image["file_name"]
            img_width = image["width"]
            img_height = image["height"]

            anno_in_image = [
                anno for anno in json_data["annotations"] if anno["image_id"] == img_id
            ]
            anno_txt = os.path.join(
                output_path, img_name.split("/")[-1].split(".")[0] + ".txt"
            )
            with open(anno_txt, "w") as f:
                for anno in anno_in_image:
                    category = anno["category_id"]
                    bbox_COCO = anno["bbox"]
                    x, y, w, h = convert_bbox_coco2yolo(
                        img_width, img_height, bbox_COCO
                    )
                    f.write(f"{category} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

    def predict(self, image_path="static/img/test_image.jpg"):
        """
        Initiate a Yolov5 model, returns instances detected with the boudning boxes
        and the category associated.
        """
        # Model
        model = torch.hub.load(
            "yolov5",
            "custom",
            path="best.pt",
            force_reload=True,
            source="local",
        )

        # Inference
        results = model(image_path)
        # Results
        image = results.render()
        Image.fromarray(image[0]).save("static/img/result_image.jpg")
        return results

    def move_training_images(self):
        """
        Move images into folder tree expected by yolov5
        """
        images = [
            os.path.join("./data/Downsampled/Train/", x)
            for x in os.listdir("./data/Downsampled/Train")
            if x[-3:] == "jpg"
        ]
        annotations = [
            os.path.join("./data/Downsampled/Train/", x)
            for x in os.listdir("./data/Downsampled/Train/")
            if x[-3:] == "txt"
        ]

        images.sort()
        annotations.sort()

        # Split the dataset into train-valid-test splits
        train_images, val_images, train_annotations, val_annotations = train_test_split(
            images, annotations, test_size=0.2, random_state=1
        )
        val_images, test_images, val_annotations, test_annotations = train_test_split(
            val_images, val_annotations, test_size=0.5, random_state=1
        )

        path = "./datasets/images/train"
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
        path = "./datasets/images/val"
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
        path = "./datasets/images/test"
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
        path = "./datasets/labels/train"
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
        path = "./datasets/labels/test"
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
        path = "./datasets/labels/val"
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)

        # Move the splits into their folders
        copy_files_to_folder(train_images, "./datasets/images/train")
        copy_files_to_folder(val_images, "./datasets/images/val/")
        copy_files_to_folder(test_images, "./datasets/images/test/")
        copy_files_to_folder(train_annotations, "./datasets/labels/train/")
        copy_files_to_folder(val_annotations, "./datasets/labels/val/")
        copy_files_to_folder(test_annotations, "./datasets/labels/test/")
