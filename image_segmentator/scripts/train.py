import matplotlib

matplotlib.use("Agg")  # noqa
import os
import json
import numpy as np
import skimage.draw
from mrcnn import model as modellib, utils

from config import BaseConfig

ROOT_DIR = "./"
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
############################################################
#  Configurations
############################################################


class SegmentationConfig(BaseConfig):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """

    # Adjust down if you use a bigger/smaller GPU.
    IMAGES_PER_GPU = 4
    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100
    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.7


############################################################
#  Dataset
############################################################
class SegmentationDataset(utils.Dataset):
    def load_segmentation(self, dataset_dir, subset):
        """Load a subset of the Segmentation dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        self.add_class("segmentation", 1, "leg")
        self.add_class("segmentation", 2, "body")
        self.add_class("segmentation", 3, "hook")
        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)
        # TODO:The format has changed in version 2 of the VIA Tool, document.
        # Load annotations
        # VGG Image Annotator saves each image in the form:
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
        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(
            annotations["_via_img_metadata"].values()
        )  # don't need the dict keys
        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a["regions"]]
        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. There are stores in the
            # shape_attributes (see json format above)
            # TODO: use own class info data structure
            classes = ["back", "leg", "body", "hook"]
            polygons = [r["shape_attributes"] for r in a["regions"]]
            class_ids = np.array(
                [classes.index(r["region_attributes"]["label"]) for r in a["regions"]],
                dtype=np.int32,
            )
            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a["filename"])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]
            self.add_image(
                "segmentation",
                image_id=a["filename"],  # use file name as a unique image id
                path=image_path,
                width=width,
                height=height,
                polygons=polygons,
                class_ids=class_ids,
            )

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a segmentation dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "segmentation":
            return super().load_mask(image_id)
        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros(
            [info["height"], info["width"], len(info["polygons"])], dtype=np.uint8
        )
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p["all_points_y"], p["all_points_x"])
            mask[rr, cc, i] = 1
        # Return mask, and array of class IDs of each instance.
        return mask.astype(np.bool), info["class_ids"]

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "segmentation":
            return info["path"]
        else:
            super().image_reference(image_id)


def train(model, dataset):
    """Train the model."""
    # Training dataset.
    dataset_train = SegmentationDataset()
    dataset_train.load_segmentation(dataset, "train")
    dataset_train.prepare()
    # Validation dataset
    dataset_val = SegmentationDataset()
    dataset_val.load_segmentation(dataset, "val")
    dataset_val.prepare()
    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(
        dataset_train,
        dataset_val,
        learning_rate=config.LEARNING_RATE,
        epochs=30,
        layers="heads",
    )


############################################################
#  Training
############################################################
if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Train Mask R-CNN to detect segmentations."
    )
    parser.add_argument(
        "--dataset",
        required=False,
        metavar="/path/to/segmentation/dataset/",
        help="Directory of the Segmentation dataset",
    )
    parser.add_argument(
        "--weights",
        required=True,
        metavar="/path/to/weights.h5",
        help="Path to weights .h5 file or 'coco'",
    )
    parser.add_argument(
        "--logs",
        required=False,
        default=DEFAULT_LOGS_DIR,
        metavar="/path/to/logs/",
        help="Logs and checkpoints directory (default=logs/)",
    )
    args = parser.parse_args()
    # Validate arguments
    assert args.weights, "Argument --weights is required for training"
    assert args.dataset, "Argument --dataset is required for training"
    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)
    # Configurations
    config = SegmentationConfig()
    config.display()
    # Create model
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=args.logs)
    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()[1]
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
        model.load_weights(
            weights_path,
            by_name=True,
            exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"],
        )
    else:
        model.load_weights(weights_path, by_name=True)
    train(model, args.dataset)
