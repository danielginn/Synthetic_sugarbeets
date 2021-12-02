import os
import my_common_modules as my_modules
import Occlusion_functions as my_functions
import numpy as np
import matplotlib.pyplot as plt
import glob
import random
import json
from math import sqrt,pi,cos,sin
from PIL import Image
import cv2
import itertools

#Plant list
plant_names = ["Sugarbeat","Capsella","Galium"]

# Load in backgrounds
SAVE_IMAGES = True
OCCLUSION_PERCENT = 0.50
if OCCLUSION_PERCENT < 0.01:
    NEW_LOCATION_CHANCE = 1.0
else:
    NEW_LOCATION_CHANCE = 0.2
data_dir = '.\\data\\backgrounds_cleaned\\'
bkgr_image_filepaths, bkgr_label_filepaths, bkgr_color_filepaths, bkgr_meta_filepaths = my_modules.import_filepaths(
    "png", "png", "json", data_dir)
bkgr_image_filepaths = bkgr_image_filepaths + bkgr_image_filepaths
bkgr_label_filepaths = bkgr_label_filepaths + bkgr_label_filepaths
bkgr_color_filepaths = bkgr_color_filepaths + bkgr_color_filepaths
bkgr_meta_filepaths = bkgr_meta_filepaths + bkgr_meta_filepaths

bkgr_count = len(bkgr_image_filepaths)
print(bkgr_count)
seed = 123
random.seed(seed)

TOTAL_IMAGES = 1890

bkgr_image_samples = random.sample(bkgr_image_filepaths, TOTAL_IMAGES)
random.seed(seed)
bkgr_label_samples = random.sample(bkgr_label_filepaths, TOTAL_IMAGES)
random.seed(seed)
bkgr_color_samples = random.sample(bkgr_color_filepaths, TOTAL_IMAGES)
random.seed(seed)
bkgr_meta_samples = random.sample(bkgr_meta_filepaths, TOTAL_IMAGES)
metadata = []
rows = 360
cols = 480
cx = cols / 2
cy = rows / 2
total_pix = rows * cols
background_count = 0
new_plant_data_bundles = []
for (lbl_path, col_lbl_path, img_path, meta_path) in zip(bkgr_label_samples, bkgr_color_samples, bkgr_image_samples,
                                                         bkgr_meta_samples):
    if len(new_plant_data_bundles) < 50:
        print("reloading new plant data due to len being " + str(len(new_plant_data_bundles)))
        new_plant_data_bundles = my_functions.getShuffledPlants()

    print(lbl_path)
    lbl = cv2.imread(lbl_path, cv2.IMREAD_GRAYSCALE)
    col_lbl = cv2.imread(col_lbl_path)
    img = cv2.imread(img_path)
    meta = {"Background": lbl_path[-8:-4]}
    # should background be flipped?
    toFlip = random.choice([0, 1])
    meta["Background_flipped"] = toFlip
    meta_json = open(meta_path)
    stem_meta = json.load(meta_json)
    meta_json.close()
    stem = stem_meta["stem"]
    if toFlip:
        lbl = cv2.flip(lbl, 1)  # Horizontal flip left to right
        col_lbl = cv2.flip(col_lbl, 1)
        img = cv2.flip(img, 1)
        if stem is not None:
            stem_xs = stem["x"]
            for i in range(len(stem_xs)):
                stem_xs[i] = int(2 * cx - stem_xs[i])
            stem["x"] = stem_xs

    # Bounding boxes for background
    bboxes = []
    white_mask = (lbl > 0)
    contours, hierarchy = cv2.findContours(white_mask.astype('uint8'), 0, 2)
    bboxid = 1
    col_output = col_lbl.copy()
    plant_id_mask = np.zeros([rows, cols], dtype='int')
    radius_mask = np.zeros([rows, cols], dtype='int')
    stem_mask = np.zeros([rows, cols, 3], dtype='int')
    stem_mask[:,:,0] = np.ones([rows, cols], dtype='int')
    plant_details = []

    # Need to associate unconnected contours
    compactContours = my_functions.associateUnconnectedContours(contours, stem)
    circles = my_functions.findingEnclosingCircles(lbl, stem)

    for stemCnts, circle in zip(compactContours,circles):
        center = circle[0]
        radius = circle[1]
        plant_mask = np.zeros([rows, cols], dtype='uint8')
        area = 0
        for cnt in stemCnts:
            cv2.drawContours(plant_mask, [cnt], 0, 1, -1)
            cv2.drawContours(plant_id_mask, [cnt], 0, bboxid, -1)
            cv2.drawContours(radius_mask, [cnt], 0, radius, -1)
            area += cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(plant_mask)
        #cv2.rectangle(col_output, (x, y), (x + w - 1, y + h - 1), (255, 255, 255), 2)

        plant_id = np.max(lbl[y:(y + h - 1), x:(x + w - 1)])
        plant_details.append([area, 0])
        stem_x, stem_y = my_functions.findStemForBBox(stem["x"], stem["y"], x, y, w, h)

        if plant_id > 1.5:
            cv2.circle(stem_mask, (stem_x, stem_y), 11, [0.0,0.0,1.0], thickness=-1)
        else:
            cv2.circle(stem_mask, (stem_x, stem_y), 11, [0.0,1.0,0.0], thickness=-1)

        bbox = {
            "plant_id": bboxid,
            "inserted": False,
            "plant_name": plant_names[plant_id - 1],
            "species_id": int(plant_id),
            "original_pixels": area,
            "occluded_pixels": 0,
            "stem": {
                "x": int(stem_x),
                "y": int(stem_y)
            },
            "bndbox": {
                "xmin": x,
                "ymin": y,
                "xmax": x + w - 1,
                "ymax": y + h - 1},
            "bndcircle": {
                "center": center,
                "radius": radius
            }
        }
        bboxes.append(bbox)
        bboxid += 1

    # Bundle Background image files together
    plant_instances = (plant_id_mask, radius_mask, plant_details)
    img_data_bundle = (img, lbl, col_lbl, plant_instances, bboxes, stem_mask)

    # Add synthetic images
    new_plants_added = random.randint(5, 12)  # How many new plants to add
    new_plant_data_bundle_list = []
    area_array = np.zeros(new_plants_added)
    for i in range(new_plants_added):
        new_plant_filepaths = new_plant_data_bundles.pop()
        new_img = cv2.imread(new_plant_filepaths[0])
        new_lbl = cv2.imread(new_plant_filepaths[1], cv2.IMREAD_GRAYSCALE)
        new_col_lbl = cv2.imread(new_plant_filepaths[2])
        meta_json = open(new_plant_filepaths[3])
        stem_meta = json.load(meta_json)
        meta_json.close()
        new_stem = stem_meta["stem"]
        area = np.sum((new_lbl > 0).astype('int'))
        area_array[i] = area
        new_plant_data_bundle_list.append((new_img, new_lbl, new_col_lbl, new_stem))

    sorted_images = np.flip(np.argsort(area_array))
    sorted_plant_data_bundle_list = []
    for i in range(new_plants_added):
        sorted_plant_data_bundle_list.append(new_plant_data_bundle_list[sorted_images[i]])

    img_data_bundle = my_functions.addPlant(img_data_bundle, sorted_plant_data_bundle_list, occlusion_percent=OCCLUSION_PERCENT,
                               new_location_chance=NEW_LOCATION_CHANCE)
    img = img_data_bundle[0]
    lbl = img_data_bundle[1]
    col_lbl = img_data_bundle[2]
    plant_instances = img_data_bundle[3]
    bboxes = img_data_bundle[4]
    stem_mask = img_data_bundle[5]
    plant_id_mask = plant_instances[0]

    background_count += 1
    dst_folder = ".\\data\\occlusion_50\\"
    if SAVE_IMAGES == True:
        cv2.imwrite(dst_folder + "rgb\\" + str(background_count).zfill(4) + ".png", img)
        cv2.imwrite(dst_folder + "label\\" + str(background_count).zfill(4) + ".png", lbl)
        cv2.imwrite(dst_folder + "color_label\\" + str(background_count).zfill(4) + ".png", col_lbl)
        cv2.imwrite(dst_folder + "instance_mask\\" + str(background_count).zfill(4) + ".png", plant_id_mask)
        cv2.imwrite(dst_folder + "stem_mask\\" + str(background_count).zfill(4) + ".png", stem_mask)
        out_file = open(dst_folder + "meta\\" + str(background_count).zfill(4) + ".json", "w")
        json.dump(bboxes, out_file, indent=4)
        out_file.close()
