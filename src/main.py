from pathlib import Path
from typing import Union, Any
import cv2
import numpy as np
import numpy.typing as npt
from definitions import DATA_DIR
from PIL import Image
import matplotlib.pyplot as plt


def load_image(path_to_image: Union[str, Path]) -> npt.NDArray[Any]:
    loaded_image = np.array(Image.open(path_to_image))
    return loaded_image


# TODO: need to use this function eventually
def pad_image(image: npt.NDArray[Any]) -> npt.NDArray[Any]:
    return np.pad(image, 1, mode="constant", constant_values=(0))


def impute_image(image: np.typing.NDArray[np.uint16], calibration_image: np.typing.NDArray[np.float64]):
    # divide and conquer strategy - check pixels between 1:end-1
    for (i, j), calibration_value in np.ndenumerate(calibration_image):

        # skip boundary in this loop

        if i == 0 or j == 0 or i > calibration_image.shape[0] - 1 or j > calibration_image.shape[1] - 1:
            pass
        else:

            # get neighbouring indices
            neighbouring_indices = [
                (i + 1, j),
                (i + 1, j + 1),
                (i + 1, j - 1),
                (i - 1, j),
                (i - 1, j + 1),
                (i - 1, j - 1),
                (i, j + 1),
                (i, j - 1),
            ]
            if calibration_value == -1:
                threshold = np.sum([calibration_image[indices] for indices in neighbouring_indices])
                if threshold == -8:
                    break  # move on to next cell
                else:
                    total = 0
                    total_index = 0
                    mean = 0
                    for indices in neighbouring_indices:
                        neighbouring_calib_value = calibration_image[indices]
                        # check which pixels are 0 around the query pixel in the calibration image
                        if neighbouring_calib_value == 0:
                            total += image[indices]
                            total_index += 1
                            # and use those to compute the average in the actual image
                    mean = total / total_index
                    # impute the pixel in the actual image
                    image[i, j] = mean

    return image


def main(path_to_image: Union[str, Path], path_to_calibration_image: Union[str, Path]) -> npt.NDArray[
    np.uint16]:
    image = load_image(path_to_image)
    calibration_image = load_image(path_to_calibration_image)

    calibration_image = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, -1, -1, 0, 0, 0, 0, 0],
        [0, 0, -1, -1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])

    image = np.array([[1., 1., 1., 1., 1., 1., 1., 1., 1.],
                      [1., 1., 1., 1., 1., 1., 1., 1., 1.],
                      [1., 1., 3., 2., 1., 1., 1., 1., 1.],
                      [1., 1., 6., 15., 1., 1., 1., 1., 1.],
                      [1., 1., 1., 1., 1., 1., 1., 1., 1.],
                      [1., 1., 1., 1., 1., 1., 1., 1., 1.]])

    image = impute_image(image, calibration_image)

    return image


if __name__ == '__main__':
    path_to_image = DATA_DIR / "bead_image.tif"
    path_to_calibration_image = DATA_DIR / "hotpixel_calibration.tif"

    corrected_image = main(path_to_image, path_to_calibration_image)
    print(corrected_image)
