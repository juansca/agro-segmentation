"""Create Datasets from video.

Usage:
  create_datasets.py -i <file> [-o <path>] [-s <int>] [-v <float>]
  create_datasets.py -h | --help
Options:
  -i <file>     Video file to generate the dataset.
  -o <path>     Path where the dataset will be saved. [default: .]
  -s <float>    Step length, in seconds, to capture the frames on video.
                [default: 4]
  -v <float>    Percentage from the dataset to validation. [default: 0.2]
  -h --help     Show this screen.

Example of Use:
    python create_datasets.py -i video.mp4 -s 4 -v 0.2
"""

from docopt import docopt
import cv2
from sklearn.model_selection import train_test_split
from pathlib import Path

from log_helpers import set_logger

log = set_logger(__name__)
PATH_SAVE_IMAGE = "data/"


def extract_images(path_in: Path, step_frame: float = 4) -> list:
    """
    Extract image frames from given video.

    Args:
        path_in: string path video
        step_frame: Step length, in seconds, to capture the frames on video
    Return:
        List of tuples with index of the frame in the video and the
        corresponding frame
    """
    vidcap = cv2.VideoCapture(path_in.as_posix())
    success = True
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT) - 1)
    log.info("The total frames for this video is {}".format(total_frames))

    framerate = int(vidcap.get(cv2.CAP_PROP_FPS))
    count, id_img = 0, 0

    frames_ret = []
    while success:

        success, image = vidcap.read()
        count += 1
        id_img += 1
        if count >= framerate * step_frame:
            count -= framerate * step_frame
            frames_ret.append((id_img, image))

    return frames_ret


def make_video_datasets(
    frames: list, train_path: Path, val_path: Path, p_val: float
) -> None:
    """
    Make the video datasets, saving train data and validation data on given directories.

    Args:
        frames: List of tuples with index of the frame in the video and the
                corresponding frame
        train_path: Path to save training data
        val_path: Path to save validation data
        p_val: Percentage from the dataset to validation
    """
    train_frames, val_frames = train_test_split(frames, test_size=p_val)
    log.info("Saving train frames...")
    for i, f in train_frames:
        path_to_save = Path(train_path, "frame_{}.jpg".format(i))
        cv2.imwrite(path_to_save.as_posix(), f)
    log.info("{} frames to train were saved".format(len(train_frames)))

    log.info("Saving val frames...")
    for i, f in val_frames:
        path_to_save = Path(val_path, "frame_{}.jpg".format(i))
        cv2.imwrite(path_to_save.as_posix(), f)
    log.info("{} frames to validation were saved".format(len(val_frames)))


def main(
    video_filename: Path, out_path: Path, frame_step: float, p_val: float
) -> None:
    """Extract and save images from video on video_filename."""
    v_name = video_filename.parts[-1].split(".")[0]
    train_path = Path(out_path, v_name, "train/")
    val_path = Path(out_path, v_name, "val/")
    train_path.mkdir(parents=True, exist_ok=True)
    val_path.mkdir(parents=True, exist_ok=True)

    list_frames = extract_images(video_filename, frame_step)
    make_video_datasets(list_frames, train_path, val_path, p_val)


if __name__ == "__main__":
    opts = docopt(__doc__)
    video_file = Path(opts["-i"])
    out_path = Path(opts["-o"])
    frame_step = float(opts["-s"])
    p_val = float(opts["-v"])

    main(video_file, out_path, frame_step, p_val)
