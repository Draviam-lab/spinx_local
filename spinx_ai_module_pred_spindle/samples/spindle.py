"""
Mask R-CNN
Train on the toy Custom dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------
python3 /home/axyom/Desktop/projects/spinx/Mask_RCNN/samples/cell/spindle.py train --dataset=/home/axyom/Desktop/projects/spinx/Mask_RCNN/datasets/spindle --weights=coco

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 Custom.py train --dataset=/Users/daviddang/MASK_R-CNN_orig/Mask_RCNN/datasets/spindle --weights=coco
    python3 Custom.py train --dataset=/Users/daviddang/MASK_R-CNN_orig/Mask_RCNN/datasets/Custom --weights=coco

    python3 Custom.py train --dataset=/Users/daviddang/MASK_R-CNN_orig/Mask_RCNN/datasets/spindle --weights=coco
    # Resume training a model that you had trained earlier
    python3 Custom.py train --dataset=/path/to/Custom/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 Custom.py train --dataset=/path/to/Custom/dataset --weights=imagenet

    # Apply color splash to an image
    python3 Custom.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 Custom.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import imgaug
import imgaug as ia
import imgaug.augmenters as iaa

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
#COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
COCO_WEIGHTS_PATH = "/home/axyom/Desktop/projects/spinx/Mask_RCNN/mask_rcnn_coco.h5"

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "/home/axyom/Desktop/projects/spinx/Mask_RCNN/logs/")


############################################################
#  Configurations
############################################################


class CustomConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "spindle"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + Custom

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100
    VALIDATION_STEPS = 50
    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

class CustomDataset(utils.Dataset):

    def load_Custom(self, dataset_dir, subset):
        """Load a subset of the Custom dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("spindle", 1, "spindle")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # VGG Image Annotator (up to version 1.6) saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        # Note: In VIA 2.0, regions was changed from a dict to a list.
        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']]

                # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "spindle",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a Custom dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "spindle":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]

        path, filename = os.path.split(info["path"])
        # Construct mask path
        mask_folder = os.path.join(path, 'mask')
        mask_path = os.path.join(mask_folder, filename)
        mask_orig = skimage.color.rgb2gray(skimage.io.imread(mask_path))

        import cv2
        contours, hierarchy = cv2.findContours(mask_orig.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        mask = np.zeros([info["height"], info["width"], len(contours)],
                        dtype=np.uint8)
        for i, contour in enumerate(contours):
            # Blank canvas
            out = np.zeros_like(mask_orig)
            # in white, and set the thickness to be 3 pixels
            single_mask = cv2.drawContours(out, [contour], -1, 255, cv2.FILLED)
            single_mask_update = cv2.bitwise_and(mask_orig, mask_orig, mask=single_mask)

            # coord = np.transpose(np.nonzero(single_mask_update))
            mask[:, :, i] = single_mask_update
            # Define class id (1 because we have only 1 class)
            mask[0, 0, i] = 0

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)
        # return info

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "spindle":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def data_augmentation(input_image, masks,
                      h_flip=True,
                      v_flip=True,
                      rotation=360,
                      zoom=1.5,
                      brightness=0.5,
                      crop=False):
    # first is input all other are output
    # Data augmentation
    output_image = input_image.copy()
    output_masks = masks.copy()
    # random crop
    # if crop and random.randint(0, 1):
    # h, w, c = output_images[0].shape
    # upper_h, new_h, upper_w, new_w = locs_for_random_crop(h, w)
    # output_images = [input_image[upper_h:upper_h + new_h, upper_w:upper_w + new_w, :] for input_image in output_images]

    # random flip
    if h_flip and random.randint(0, 1):
        output_image = np.fliplr(output_image)
        output_masks = np.fliplr(output_masks)

    if v_flip and random.randint(0, 1):
        output_image = np.flipud(output_image)
        output_masks = np.flipud(output_masks)

    factor = 1.0 + abs(random.gauss(mu=0.0, sigma=brightness))
    if random.randint(0, 1):
        factor = 1.0 / factor
    table = np.array([((i / 255.0) ** factor) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
    output_image = cv2.LUT(output_image, table)
    if rotation:
        rotate_times = random.randint(0, rotation / 90)
    else:
        rotate_times = 0.0
    for r in range(0, rotate_times):
        output_image = np.rot90(output_image)
        output_masks = np.rot90(output_masks)

    #     if zoom:
    #         scale = random.randint(50, zoom * 100) / 100
    #     else:
    #         scale = 1.0
    #     # print(angle, scale)
    #     if rotation or zoom:
    #         for i, input_image in enumerate(output_images):
    #             M = cv2.getRotationMatrix2D((input_image.shape[1] // 2, input_image.shape[0] // 2), angle, scale)
    #             # M = cv2.getRotationMatrix2D((input_image.shape[1] // 2, input_image.shape[0] // 2), 45, 1)
    #             output_images[i] = cv2.warpAffine(input_image, M, (input_image.shape[1], input_image.shape[0]))
    #     # print('len of output %s' % len(output_images))
    return output_image, output_masks


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = CustomDataset()
    dataset_train.load_Custom(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = CustomDataset()
    dataset_val.load_Custom(args.dataset, "val")
    dataset_val.prepare()

    # Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
    # e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    aug_v1 = iaa.Sometimes(0.9, [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.Multiply((0.8, 1.2)),
        iaa.GaussianBlur(sigma=(0.0, 5.0))
    ])
    # Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
    # e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
    st = lambda aug: iaa.Sometimes(0.3, aug)
    aug_v2 = iaa.Sequential(
        [
            # apply the following augmenters to most images
            iaa.Fliplr(0.5),  # horizontally flip 50% of all images
            iaa.Flipud(0.2),  # vertically flip 20% of all images
            iaa.SomeOf((0, 5),
                       [
                        iaa.GaussianBlur(sigma=(0.0, 5.0)),
                        iaa.GammaContrast((0.5, 1.5)),
                        iaa.Add((-10, 10), per_channel=0.5),
                        iaa.Multiply((0.8, 1.2), per_channel=0.2),
                        iaa.ContrastNormalization((0.75, 1.5))
                       ]),
            # crop images by -5% to 10% of their height/width
            st(iaa.Crop(percent=(0, 0.1))), # crop images by 0-10% of their height/width
            st(iaa.Dropout((0.0, 0.1), per_channel=0.5)), # randomly remove up to 10% of the pixels
            st(iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)})), # scale images to 80-120% of their size, individually per axis
            sometimes(iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                # scale images to 80-120% of their size, individually per axis
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # translate by -20 to +20 percent (per axis)
                rotate=(-10, 10),  # rotate by -45 to +45 degrees
                shear=(-8, 8),  # shear by -8 to +8 degrees
            )),
        ],
        random_order=True
    )

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    # LR (default): 0.001

    n_epoch = 300
    b = 0.2*n_epoch
    lr = 0.00101+0.001*np.cos(b+np.cos(b+np.cos(b+np.cos(b+np.cos(b+np.cos(b+np.cos(b)))))))
    # https: // github.com / matterport / Mask_RCNN / issues / 289

    model_inference = modellib.MaskRCNN(mode="inference", config=CustomConfig(), model_dir=args.logs)
    mean_average_precision_callback = modellib.MeanAveragePrecisionCallback(model, model_inference, dataset_val, calculate_map_at_every_X_epoch=5, verbose=1)


    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=lr,
                epochs=n_epoch,
                layers='all',
                augmentation=aug_v2,
                custom_callbacks=[mean_average_precision_callback]
                )


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect spindle.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/dataset/",
                        help='Directory of the dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video, \
            "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = CustomConfig()
    else:
        class InferenceConfig(CustomConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1


        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
