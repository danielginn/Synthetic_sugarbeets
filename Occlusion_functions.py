import my_common_modules as my_modules
import random
import numpy as np
from math import sqrt,pi,cos,sin
import cv2
import itertools

plant_names = ["Sugarbeat", "Capsella", "Galium"]  # Plant list

def getShuffledPlants(smallest_set=359):
    # Load in backgrounds
    image_filepaths = []
    label_filepaths = []
    color_filepaths = []
    meta_filepaths = []

    data_dirs = ['.\\data\\sugarbeat_cleaned\\' ,'.\\data\\capsella_cleaned\\' ,'.\\data\\galium_cleaned\\']

    for data_dir in data_dirs:
        add_image_filepaths ,add_label_filepaths ,add_color_filepaths ,add_meta_filepaths = my_modules.import_filepaths("png" ,"png" ,"json", data_dir)
        add_count = len(add_image_filepaths)
        if add_count > smallest_set:
            samples = list(zip(add_image_filepaths ,add_label_filepaths ,add_color_filepaths ,add_meta_filepaths))
            samples = random.sample(samples ,smallest_set)
            add_image_filepaths ,add_label_filepaths ,add_color_filepaths ,add_meta_filepaths = zip(*samples)
        image_filepaths += add_image_filepaths
        label_filepaths += add_label_filepaths
        color_filepaths += add_color_filepaths
        meta_filepaths += add_meta_filepaths
    sample_filepaths = list(zip(image_filepaths ,label_filepaths ,color_filepaths ,meta_filepaths))
    random.shuffle(sample_filepaths)
    plant_data_bundles = []
    for plant_data_bundle in sample_filepaths:
        plant_data_bundles.append(plant_data_bundle)

    return plant_data_bundles

def checkNewPlantOcclusion(instance_mask, radius_mask, new_radius_mask, new_plant_id):
    # 1. Calculate pixels per plant instance within crop
    # 1a. For existing plants
    num_plants = new_plant_id - 1
    plant_px = np.zeros(new_plant_id)
    for plant_id in range(1, num_plants + 1):
        plant_px[plant_id - 1] = np.sum(instance_mask == plant_id)
    # 1b. For new plant
    plant_px[-1] = round(np.sum(new_radius_mask) / np.max(new_radius_mask))

    # 2. Compare radii per pixel
    new_plant_pixel_added = (radius_mask < new_radius_mask).astype('int')
    existing_plant_pixel_kept = (radius_mask >= new_radius_mask)

    # 3. Set instance values occluded by new plant to zero
    remaining_original_px = np.multiply(instance_mask, existing_plant_pixel_kept)

    # 4. Calculate newly occluded pixels
    lost_px = np.zeros(num_plants + 1, dtype='int')
    for plant_id in range(1, num_plants + 1):
        lost_px[plant_id - 1] = plant_px[plant_id - 1] - np.sum(remaining_original_px == plant_id)
    lost_px[-1] = plant_px[-1] - np.sum(new_plant_pixel_added)

    return lost_px, new_plant_pixel_added, existing_plant_pixel_kept

def findStemForBBox(stem_xs, stem_ys, x, y, w, h):
    stem_x = 0
    stem_y = 0
    for i in range(len(stem_xs)):
        stem_x = int(stem_xs[i])
        stem_y = int(stem_ys[i])
        if (stem_x > x) and (stem_x < (x + w)) and (stem_y > y) and (stem_y < (y + h)):
            return stem_x, stem_y
    if ((stem_x == 0) and (stem_y == 0)):
        print("WARNING: Stem not in bounding box!!!")
    return stem_x, stem_y

def findingEnclosingCircles(lbl, stem, kernal_size=5):
    allPlantPartsConnected = False
    dilationAttemptsMax = 2
    dilationAttempts = 0
    while allPlantPartsConnected == False:
        allPlantPartsConnected = True
        dilationAttempts += 1
        diluted_white_mask = (lbl > 0)
        # Taking a matrix of size 5 as the kernel
        kernel = np.ones((kernal_size, kernal_size), np.uint8)
        diluted_white_mask = cv2.dilate(diluted_white_mask.astype('uint8'), kernel, iterations=1)
        diluted_white_mask = cv2.erode(diluted_white_mask, kernel, iterations=1)

        diluted_contours, hierarchy = cv2.findContours(diluted_white_mask.astype('uint8'), 0, 2)
        circles_unordered = []
        for cnt in diluted_contours:
            (cir_x, cir_y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(cir_x), int(cir_y))
            radius = int(radius)
            circles_unordered.append([center, radius])

        # Error checks
        if (stem is not None) and (len(stem["x"]) != len(circles_unordered)):
            allPlantPartsConnected = False
            kernal_size = 7
            if dilationAttempts == dilationAttemptsMax:
                raise NameError("More circles found than stems")
    if (stem is None) and (len(circles_unordered) > 0):
        raise NameError("Circles found when there are no stems")

    # Sort circles
    circles = []
    if stem is not None:
        for i in range(len(stem["x"])):
            min_d = 9999
            min_j = -1
            for j in range(len(circles_unordered)):
                diff_x = stem["x"][i] - circles_unordered[j][0][0]
                diff_y = stem["y"][i] - circles_unordered[j][0][1]
                d = sqrt(diff_x * diff_x + diff_y * diff_y)
                if d < min_d:
                    min_d = d
                    min_j = j
            circles.append(circles_unordered[min_j])

    return circles

"""
Finds index of stem closest to each contour
"""
def associateUnconnectedContours(contours, stem):
    compactContours = []
    if stem is not None:
        for i in range(len(stem["x"])):
            compactContours.append([])
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        bbox_cx = x + w / 2
        bbox_cy = y + h / 2
        stem_xs = stem["x"]
        stem_ys = stem["y"]
        min_d = 9999
        min_i = -1
        for i in range(len(stem_xs)):
            x_diff = bbox_cx - stem_xs[i]
            y_diff = bbox_cy - stem_ys[i]
            d = sqrt(x_diff * x_diff + y_diff * y_diff)
            if d < min_d:
                min_d = d
                min_i = i
        compactContours[min_i].append(cnt)
    return compactContours

def addPlant(img_data_bundle, new_plant_data_bundle_list, occlusion_percent=0, new_location_chance=0.33, debug_level=0):
    if debug_level > 0:
        print("Debug level > 1")
        MAX_TRIES = 20
        POINTS_ON_LINE = 5
    else:
        MAX_TRIES = 20
        POINTS_ON_LINE = 20

    img, lbl, col_lbl, plant_instances, bboxes, stem_mask = img_data_bundle
    plant_id_mask, plant_size_mask, plant_details = plant_instances

    rows = np.shape(img)[0]
    cols = np.shape(img)[1]

    for (new_plant_img, new_plant_lbl, new_plant_col_lbl, new_stem) in new_plant_data_bundle_list:
        new_plant_species = np.max(new_plant_lbl)
        new_plant_lbl_mask = (new_plant_lbl >= 1).astype('int')
        new_plant_pixels = np.sum(new_plant_lbl_mask)
        new_plant_details = [new_plant_pixels, 0]
        plant_details.append(new_plant_details)
        new_plant_id = len(plant_details)
        radius_new = round(np.shape(new_plant_img)[0] / 2)
        new_radius_mask = new_plant_lbl_mask * radius_new
        foundGoodSpot = False
        # If plant to be added for occlusion
        if (random.random() >= new_location_chance) and (len(bboxes) > 0):
            for try_num in range(MAX_TRIES):
                if debug_level > 0:
                    theta = pi / 2
                    bbox = bboxes[0]
                else:
                    theta = random.uniform(-pi, pi)
                    bbox = random.choice(bboxes)

                plant_id = int(bbox['plant_id'])
                radius = int(bbox['bndcircle']['radius'])
                center = bbox['bndcircle']['center']
                dmax = radius + radius_new
                count = 0
                for try_d in np.arange(dmax / POINTS_ON_LINE, dmax, dmax / POINTS_ON_LINE):
                    center_new = (int((try_d + radius_new) * cos(theta) + center[0]),
                                  int((try_d + radius_new) * sin(theta) + center[1]))
                    xmin = round(center_new[0] - radius_new)
                    ymin = round(center_new[1] - radius_new)
                    xmax = round(center_new[0] + radius_new)
                    ymax = round(center_new[1] + radius_new)
                    # check if out of bounds
                    if ((xmin < 0) or (ymin < 0) or (xmax >= cols) or (ymax >= rows)):
                        continue
                    foundGoodSpot = True
                    # add new plant to background crop
                    losses, new_plant_pix_vis, existing_plant_pixel_kept = checkNewPlantOcclusion(
                        plant_id_mask[ymin:ymax, xmin:xmax], plant_size_mask[ymin:ymax, xmin:xmax], new_radius_mask,
                        new_plant_id)
                    plant1_occlusion = (plant_details[plant_id - 1][1] + losses[plant_id - 1]) / \
                                       plant_details[plant_id - 1][0]
                    plant2_occlusion = losses[-1] / plant_details[-1][0]
                    count += 1
                    for i in range(np.shape(losses)[0]):
                        original_px = plant_details[i][0]
                        occluded_px = plant_details[i][1] + losses[i]
                        new_occlusion_pcnt = occluded_px / original_px
                        if ((new_occlusion_pcnt > (occlusion_percent + 0.05)) or (
                                (plant1_occlusion < (occlusion_percent - 0.05)) and (
                                plant2_occlusion < (occlusion_percent - 0.05)))):
                            foundGoodSpot = False

                    if foundGoodSpot:
                        # Update pixels
                        # 1. Remove occluded pixels from new plant using new_plant_pix_vis
                        new_plant_lbl = np.multiply(new_plant_pix_vis, new_plant_lbl)
                        for c in range(3):
                            new_plant_img[:, :, c] = np.multiply(new_plant_pix_vis, new_plant_img[:, :, c])
                            new_plant_col_lbl[:, :, c] = np.multiply(new_plant_pix_vis, new_plant_col_lbl[:, :, c])

                        # 2. Set pixels in original image to be replaced to zero
                        # img,lbl,col_lbl,plant_instances,bboxes
                        lbl[ymin:ymax, xmin:xmax] = np.multiply(lbl[ymin:ymax, xmin:xmax],
                                                                existing_plant_pixel_kept.astype('int'))
                        plant_id_mask[ymin:ymax, xmin:xmax] = np.multiply(plant_id_mask[ymin:ymax, xmin:xmax],
                                                                          existing_plant_pixel_kept.astype('int'))
                        plant_size_mask[ymin:ymax, xmin:xmax] = np.multiply(plant_size_mask[ymin:ymax, xmin:xmax],
                                                                            existing_plant_pixel_kept.astype('int'))
                        for c in range(3):
                            img[ymin:ymax, xmin:xmax, c] = np.multiply(img[ymin:ymax, xmin:xmax, c],
                                                                       existing_plant_pixel_kept.astype('int'))
                            col_lbl[ymin:ymax, xmin:xmax, c] = np.multiply(col_lbl[ymin:ymax, xmin:xmax, c],
                                                                           existing_plant_pixel_kept.astype('int'))

                        # 3. Add new plant pixels to blackened out original image
                        lbl[ymin:ymax, xmin:xmax] = lbl[ymin:ymax, xmin:xmax] + new_plant_lbl
                        img[ymin:ymax, xmin:xmax, :] = img[ymin:ymax, xmin:xmax, :] + new_plant_img
                        col_lbl[ymin:ymax, xmin:xmax, :] = col_lbl[ymin:ymax, xmin:xmax, :] + new_plant_col_lbl
                        plant_id_mask[ymin:ymax, xmin:xmax] = plant_id_mask[ymin:ymax,
                                                              xmin:xmax] + new_plant_id * new_plant_pix_vis.astype(
                            'int')
                        plant_size_mask[ymin:ymax, xmin:xmax] = plant_size_mask[ymin:ymax,
                                                                xmin:xmax] + radius_new * new_plant_pix_vis.astype(
                            'int')

                        if new_plant_species > 1.5:
                            cv2.circle(stem_mask, (xmin + int(new_stem["x"]), ymin + int(new_stem["y"])), 11, [0.0,0.0,1.0], thickness=-1)
                        else:
                            cv2.circle(stem_mask, (xmin + int(new_stem["x"]), ymin + int(new_stem["y"])), 11, [0.0,1.0,0.0], thickness=-1)

                        # 4. Update plant details
                        for i in range(new_plant_id):
                            plant_details[-i][1] += losses[i]

                        # 5. Update bboxes
                        xmin_bbox = xmin
                        ymin_bbox = ymin
                        xmax_bbox = xmin + 2 * radius_new - 1
                        ymax_bbox = ymin + 2 * radius_new - 1
                        for x in range(0, radius_new):
                            if np.sum(new_plant_lbl[:, x]) > 0:
                                xmin_bbox = xmin + x
                                break
                        for x in range(2 * radius_new - 1, radius_new, -1):
                            if np.sum(new_plant_lbl[:, x]) > 0:
                                xmax_bbox = xmin + x
                                break
                        for y in range(0, radius_new):
                            if np.sum(new_plant_lbl[y, :]) > 0:
                                ymin_bbox = ymin + y
                                break
                        for y in range(2 * radius_new - 1, radius_new, -1):
                            if np.sum(new_plant_lbl[y, :]) > 0:
                                ymax_bbox = ymin + y
                                break

                        bbox = {
                            "plant_id": int(new_plant_id),
                            "inserted": True,
                            "plant_name": plant_names[new_plant_species - 1],
                            "species_id": int(new_plant_species),
                            "original_pixels": int(new_plant_pixels),
                            "occluded_pixels": int(0),
                            "stem": {"x": xmin + int(new_stem["x"]),
                                     "y": ymin + int(new_stem["y"])
                                     },
                            "bndbox": {
                                "xmin": int(xmin_bbox),
                                "ymin": int(ymin_bbox),
                                "xmax": int(xmax_bbox),
                                "ymax": int(ymax_bbox)
                            },
                            "bndcircle": {
                                "center": center_new,
                                "radius": int(radius_new)
                            }
                        }
                        bboxes.append(bbox)
                        bbox_count = 0
                        for bbox in bboxes:
                            bbox['occluded_pixels'] = int(bbox['occluded_pixels'] + losses[bbox_count])
                            bbox_count += 1
                        break

                if foundGoodSpot:
                    break

        if not (foundGoodSpot):  # place randomly
            # First block out current bounding boxes
            original_occupation_mask = np.zeros([rows, cols])
            for bbox in bboxes:
                bndbox = bbox['bndbox']
                xmin = int(bndbox['xmin'])
                ymin = int(bndbox['ymin'])
                xmax = int(bndbox['xmax'])
                ymax = int(bndbox['ymax'])
                cv2.rectangle(original_occupation_mask, (xmin, ymin), (xmax, ymax), 1, -1)

            w = np.shape(new_plant_img)[1]
            h = np.shape(new_plant_img)[0]
            if debug_level > 0:
                print("[h,w]=[" + str(h) + "," + str(w) + "]")
                print("Occupation Mask:")
                print(original_occupation_mask)
            # start randomly trying
            for try_num in range(3 * MAX_TRIES):
                rand_xmin = random.randint(0, cols - w)
                rand_ymin = random.randint(0, rows - h)
                trial_occupation_mask = original_occupation_mask[rand_ymin:rand_ymin + h,
                                        rand_xmin:rand_xmin + w] + np.ones([h, w])
                if debug_level > 0:
                    print("tring position: [" + str(rand_ymin) + "," + str(rand_xmin) + "]")
                    print(trial_occupation_mask)
                if np.sum((trial_occupation_mask == 2).astype('int')) == 0:  # if there is no overlap
                    foundGoodSpot = True
                    break

            if foundGoodSpot:
                # 1. blacken out original pixels
                original_pixels_obscured = np.invert(new_plant_lbl_mask.astype(np.bool)).astype('int')
                lbl[rand_ymin:rand_ymin + h, rand_xmin:rand_xmin + w] = np.multiply(original_pixels_obscured,
                                                                                    lbl[rand_ymin:rand_ymin + h,
                                                                                    rand_xmin:rand_xmin + w])
                plant_id_mask[rand_ymin:rand_ymin + h, rand_xmin:rand_xmin + w] = np.multiply(
                    original_pixels_obscured.astype('int'),
                    plant_id_mask[rand_ymin:rand_ymin + h, rand_xmin:rand_xmin + w])
                plant_size_mask[rand_ymin:rand_ymin + h, rand_xmin:rand_xmin + w] = np.multiply(
                    original_pixels_obscured.astype('int'),
                    plant_size_mask[rand_ymin:rand_ymin + h, rand_xmin:rand_xmin + w])
                for c in range(3):
                    img[rand_ymin:rand_ymin + h, rand_xmin:rand_xmin + w, c] = np.multiply(original_pixels_obscured,
                                                                                           img[rand_ymin:rand_ymin + h,
                                                                                           rand_xmin:rand_xmin + w, c])
                    col_lbl[rand_ymin:rand_ymin + h, rand_xmin:rand_xmin + w, c] = np.multiply(original_pixels_obscured,
                                                                                               col_lbl[
                                                                                               rand_ymin:rand_ymin + h,
                                                                                               rand_xmin:rand_xmin + w,
                                                                                               c])

                # 2. add new plant to original image
                img[rand_ymin:rand_ymin + h, rand_xmin:rand_xmin + w, :] += new_plant_img
                lbl[rand_ymin:rand_ymin + h, rand_xmin:rand_xmin + w, ] += new_plant_lbl
                col_lbl[rand_ymin:rand_ymin + h, rand_xmin:rand_xmin + w, :] += new_plant_col_lbl
                plant_id_mask[rand_ymin:rand_ymin + h, rand_xmin:rand_xmin + w] += (
                            new_plant_lbl_mask.astype('int') * new_plant_id)
                plant_size_mask[rand_ymin:rand_ymin + h, rand_xmin:rand_xmin + w] += (
                            new_plant_lbl_mask.astype('int') * radius_new)

                if new_plant_species > 1.5:
                    cv2.circle(stem_mask, (rand_xmin + int(new_stem["x"]), rand_ymin + int(new_stem["y"])), 11, [0.0,0.0,1.0], thickness=-1)
                else:
                    cv2.circle(stem_mask, (rand_xmin + int(new_stem["x"]), rand_ymin + int(new_stem["y"])), 11, [0.0,1.0,0.0], thickness=-1)

                # 3. Update bboxes
                plant_outline = (plant_id_mask == new_plant_id).astype('uint8')
                # plt.figure(figsize=(12,12))
                # plt.imshow((plant_outline*250).astype('uint8'))
                # plt.show()
                ret, thresh = cv2.threshold(plant_outline, 0, 255, 0)
                contours, hierarchy = cv2.findContours(thresh, 1, 2)
                biggestarea = 0
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area > biggestarea:
                        biggestcnt = cnt
                x_bbox, y_bbox, w_bbox, h_bbox = cv2.boundingRect(biggestcnt)

                center_new = (round(rand_xmin + w / 2), round(rand_ymin + h / 2))
                bbox = {
                    "plant_id": int(new_plant_id),
                    "inserted": True,
                    "plant_name": plant_names[new_plant_species - 1],
                    "species_id": int(new_plant_species),
                    "original_pixels": int(new_plant_pixels),
                    "occluded_pixels": int(0),
                    "stem": {"x": rand_xmin + int(new_stem["x"]),
                             "y": rand_ymin + int(new_stem["y"])
                             },
                    "bndbox": {
                        "xmin": int(x_bbox),
                        "ymin": int(y_bbox),
                        "xmax": int(x_bbox + w_bbox - 1),
                        "ymax": int(y_bbox + h_bbox - 1)
                    },
                    "bndcircle": {
                        "center": center_new,
                        "radius": int(radius_new)
                    }
                }
                bboxes.append(bbox)
        if not (foundGoodSpot):
            plant_details.pop()

    img_data_bundle = (img, lbl, col_lbl, plant_instances, bboxes, stem_mask)
    return img_data_bundle