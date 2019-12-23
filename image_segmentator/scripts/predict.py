import matplotlib

matplotlib.use("Agg")  # noqa
import matplotlib.pyplot as plt
from PIL import Image
import io
import os
import pickle
import numpy as np
from mrcnn import visualize
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


class InferenceConfig(BaseConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Skip detections with less than this confidence
    DETECTION_MIN_CONFIDENCE = 0.7


def detect_and_color_splash(model, video_path):
    import cv2

    # Video capture
    vcapture = cv2.VideoCapture(video_path)
    width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = vcapture.get(cv2.CAP_PROP_FPS)
    # Define codec and create video writer
    file_name = video_path + "_processed.avi"
    vwriter = cv2.VideoWriter(
        file_name, cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height)
    )
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
            visualize.display_instances(
                image,
                r["rois"],
                r["masks"],
                r["class_ids"],
                ["background", "leg", "body", "hook"],
                r["scores"],
                show_bbox=False,
                show_mask=True,
                title="Predictions",
            )
            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            im = Image.open(buf)
            # RGB -> BGR to save image to video
            data = np.asarray(im)
            im = Image.fromarray(data)
            im = im.resize((width, height))
            im = np.array(im)
            im = im[:, :, :3]
            im = im[..., ::-1]
            # Add image to video writer
            vwriter.write(im)
            count += 1
            plt.close()
    vwriter.release()
    print("Saved to ", file_name)


def dump_metadata(model, video_path):
    import cv2

    # Video capture
    vcapture = cv2.VideoCapture(video_path)
    width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = vcapture.get(cv2.CAP_PROP_FPS)
    # Define codec and create video writer
    file_name = video_path + "_metadata.pkl"
    count = 0
    success = True
    metadata = {}
    metadata["filename"] = video_path
    metadata["width"] = width
    metadata["height"] = height
    metadata["fps"] = fps
    metadata["frame_metadata"] = {}
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
            metadata["frame_metadata"][count] = r
            count += 1
    with open(file_name, "wb") as dumpfile:
        pickle.dump(metadata, dumpfile)
    print("Saved to ", file_name)


############################################################
#  Prediction using model
############################################################
if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Predict segments based on Mask R-CNN model"
    )
    parser.add_argument("command", metavar="<command>", help="'dump' or 'splash'")
    parser.add_argument(
        "--weights",
        required=True,
        metavar="/path/to/weights.h5",
        help="Path to weights .h5 file or 'coco'",
    )
    parser.add_argument(
        "--video",
        required=True,
        metavar="path or URL to video",
        help="Video to apply the color splash effect on",
    )
    args = parser.parse_args()
    print("Weights: ", args.weights)
    config = InferenceConfig()
    config.display()
    # Create model
    model = modellib.MaskRCNN(
        mode="inference", config=config, model_dir=DEFAULT_LOGS_DIR
    )
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
    # Evaluate
    if args.command == "splash":
        detect_and_color_splash(model, video_path=args.video)
    elif args.command == "dump":
        dump_metadata(model, video_path=args.video)
    else:
        print("'{}' is not recognized. Use 'splash' or 'dump'".format(args.command))
