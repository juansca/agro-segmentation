from mrcnn.config import Config


class BaseConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """

    # Give the configuration a recognizable name
    NAME = "segmentation"
    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + segmentation (just one class)
