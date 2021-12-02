def import_dataset_filepaths(folder, trainfile=None, testfile=None):
    import os

    # trainfilepath provided
    trainfilepath = os.path.join(folder, trainfile)
    testfilepath = os.path.join(folder, testfile)
    my_file = open(trainfilepath, "r")
    contents = my_file.read().splitlines()
    my_file.close()
    my_file = open(testfilepath, "r")
    contents += my_file.read().splitlines()
    my_file.close()

    image_filepaths = []
    label_filepaths = []
    instance_filepaths = []
    color_filepaths = []
    meta_filepaths = []
    stem_filepaths = []
    for tail in contents:
        image_filepaths.append(os.path.join(os.path.join(folder, 'rgb'), tail))
        label_filepaths.append(os.path.join(os.path.join(folder, 'label'), tail))
        instance_filepaths.append(os.path.join(os.path.join(folder, 'instance_mask'), tail))
        color_filepaths.append(os.path.join(os.path.join(folder, 'color_label'), tail))
        stem_filepaths.append(os.path.join(os.path.join(folder, 'stem_mask'), tail))
        meta_filepaths.append(os.path.join(os.path.join(folder, 'meta'), tail[:-3]+"json"))

    return {'image': image_filepaths, 'label': label_filepaths, 'instance': instance_filepaths,
            'color': color_filepaths, 'meta': meta_filepaths, 'stem': stem_filepaths}, len(image_filepaths)

def import_filepaths(image_ext,label_ext,meta_ext, folder='.\\data\\'):
    import glob
    image_filepaths = glob.glob(folder+'rgb\\*.'+image_ext) # Obtain filepaths of images
    label_filepaths = glob.glob(folder+'gt\\*.'+label_ext)
    color_filepaths = glob.glob(folder+'gt_color\\*.' + label_ext)
    meta_filepaths = glob.glob(folder+'meta\\*.' + meta_ext)
    if image_filepaths == []:
        raise NameError("NO IMAGES FOUND")
    elif label_filepaths == []:
        raise NameError("NO LABELS FOUND")
    elif color_filepaths == []:
        raise NameError("NO COLOR LABELS FOUND")
    elif meta_filepaths == []:
        raise NameError("NO META DATA FOUND")
    return image_filepaths,label_filepaths, color_filepaths, meta_filepaths