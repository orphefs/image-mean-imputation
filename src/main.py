from pathlib import Path
from typing import Union, Any, Tuple
from scipy import stats
import cv2
import pandas as pd
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


def impute_image(image: np.typing.NDArray[np.uint16], calibration_image: np.typing.NDArray[np.float32]):
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
                # update the calibration image from -1 to 0 for that index
                calibration_image[i, j] = 0

    return image


def filter_outliers(df: pd.DataFrame) -> pd.DataFrame:
    # Compute IQR
    filtered = df.copy()
    Q1 = filtered["values"].quantile(0.25)
    Q3 = filtered["values"].quantile(0.75)
    IQR = Q3 - Q1
    # Filtering Values between Q1-1.5IQR and Q3+1.5IQR
    filtered = filtered.query('(@Q1 - 1.5 * @IQR) <= {} <= (@Q3 + 1.5 * @IQR)'.format("values"))
    return filtered


def equalize_histogram(image: npt.NDArray[Any]) -> np.ndarray:
    equ = cv2.equalizeHist(image)
    res = np.hstack((image, equ))  # stacking images side-by-side

    return res


def normalize_image(image: npt.NDArray[Any], max_val: int) -> npt.NDArray[Any]:
    res = image.copy()
    cv2.normalize(image, res, 0, max_val, cv2.NORM_MINMAX)
    return res


def plot_image_statistics(image: np.typing.NDArray[np.uint16],
                          calibration_image: np.typing.NDArray[np.float32]) -> Tuple[plt.figure, plt.axes]:
    fig, ax = plt.subplots(nrows=3, ncols=2)

    # Image plots
    # image
    ax[0, 0].imshow(np.hstack([image, normalize_image(image, 65535)]))
    ax[0, 0].set_title("Image and normalized version")

    # calibration image
    ax[0, 1].imshow(calibration_image)
    ax[0, 1].set_title("Calibration image")

    # Scatter and histogram plots

    # image
    prob = stats.probplot(image.flatten(), dist=stats.norm, plot=ax[1, 0])
    ax[1, 0].set_title('Probability plot against normal distribution')
    pd.DataFrame({"values": normalize_image(image, 65535).flatten()}).plot.hist(bins=100, ax=ax[2, 0])
    ax[2, 0].set_title("Histogram after image normalization")

    # calibration image
    ax[1, 1].scatter(x=np.arange(len(calibration_image.flatten())), y=calibration_image.flatten())
    ax[1, 1].set_title("Scatter plot")

    filter_outliers(pd.DataFrame({"values": calibration_image.flatten()})).plot.hist(bins=100, ax=ax[2, 1])

    return fig, ax


def draw_values(ax: plt.Axes, image: npt.NDArray[Any], ):
    for (j, i), label in np.ndenumerate(image):
        ax.text(i, j, label, ha='center', va='center')


def plot_processing_results(image: np.typing.NDArray[np.uint16],
                            calibration_image: np.typing.NDArray[np.float32],
                            imputed_image: np.typing.NDArray[np.uint16]) -> Tuple[plt.figure, plt.axes]:
    fig, ax = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True)
    # TODO: plot values of pixels on image
    ax[0].imshow(normalize_image(image, 65535))
    ax[0].set_title("Original image")
    calibration_image[np.where(calibration_image < 0.0)] = 0.0
    calibration_image[np.where(calibration_image > 0.0)] = 1.0
    # draw_values(ax[0], normalize_image(image, 65535))

    ax[1].imshow(calibration_image)
    ax[1].set_title("Calibration image")
    # draw_values(ax[1], calibration_image)

    ax[2].imshow(normalize_image(imputed_image, 65535))
    ax[2].set_title("Imputed image")
    # draw_values(ax[2], normalize_image(imputed_image, 65535))
    #
    # cv2.imshow('image', imputed_image)
    # cv2.waitKey(0)

    return fig, ax


def main(path_to_image: Union[str, Path], path_to_calibration_image: Union[str, Path]) -> npt.NDArray[
    np.uint16]:
    imputed_image = impute_image(image=load_image(path_to_image),
        calibration_image=load_image(path_to_calibration_image))

    fig, ax = plot_image_statistics(image=load_image(path_to_image),
        calibration_image=load_image(path_to_calibration_image), )

    fig, ax = plot_processing_results(image=load_image(path_to_image),
        calibration_image=load_image(path_to_calibration_image),
        imputed_image=imputed_image)

    plt.show()

    return imputed_image


if __name__ == '__main__':
    path_to_image = DATA_DIR / "bead_image.tif"
    path_to_calibration_image = DATA_DIR / "hotpixel_calibration.tif"

    corrected_image = main(path_to_image, path_to_calibration_image)
    print(corrected_image)
