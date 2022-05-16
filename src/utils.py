from pathlib import Path
from typing import Union, Any, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
from numpy import typing as npt


def load_image(path_to_image: Union[str, Path]) -> npt.NDArray[Any]:
    loaded_image = np.array(Image.open(path_to_image))
    return loaded_image


def filter_outliers(df: pd.DataFrame) -> pd.DataFrame:
    # Compute IQR
    filtered = df.copy()
    Q1 = filtered["values"].quantile(0.25)
    Q3 = filtered["values"].quantile(0.75)
    IQR = Q3 - Q1
    # Filtering Values between Q1-1.5IQR and Q3+1.5IQR
    filtered = filtered.query(
        '(@Q1 - 1.5 * @IQR) <= {} <= (@Q3 + 1.5 * @IQR)'.format("values"))
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

    # prob = stats.probplot(image.flatten(), dist=stats.norm, plot=ax[1, 0])
    # ax[1, 0].set_title('Probability plot against normal distribution')
    # pd.DataFrame({"values": normalize_image(image, 65535).flatten()}).plot.hist(bins=100, ax=ax[2, 0])
    ax[1, 0].set_title("Histogram before image normalization")
    pd.DataFrame({"values": image.flatten()}).plot.hist(bins=100, ax=ax[1, 0])

    ax[2, 0].set_title("Histogram after image normalization")
    pd.DataFrame({"values": normalize_image(image, 65535).flatten()}
                 ).plot.hist(bins=100, ax=ax[2, 0])

    # calibration image
    ax[1, 1].scatter(x=np.arange(len(calibration_image.flatten())),
                     y=calibration_image.flatten())
    ax[1, 1].set_title("Scatter plot")

    filter_outliers(pd.DataFrame({"values": calibration_image.flatten()})).plot.hist(
        bins=100, ax=ax[2, 1])

    return fig, ax


def draw_values(ax: plt.Axes, image: npt.NDArray[Any], ):
    for (j, i), label in np.ndenumerate(image):
        ax.text(i, j, label, ha='center', va='center')


def plot_processing_results(image: npt.NDArray[np.uint16],
                            calibration_image: npt.NDArray[np.float32],
                            imputed_image: npt.NDArray[np.uint16],
                            imputed_image_opencv: npt.NDArray[np.uint16]) -> Tuple[plt.figure, plt.axes]:
    fig, ax = plt.subplots(nrows=1, ncols=4, sharex=True, sharey=True)
    # TODO: plot values of pixels on image
    # normalize_image(image, 65535)
    ax[0].imshow(normalize_image(image, 65535))
    ax[0].set_title("Original image")
    calibration_image[np.where(calibration_image < 0.0)] = 0.0
    calibration_image[np.where(calibration_image > 0.0)] = 1.0
    # draw_values(ax[0], normalize_image(image, 65535))

    ax[1].imshow(calibration_image)
    ax[1].set_title("Calibration image")
    # draw_values(ax[1], calibration_image)

    ax[2].imshow(normalize_image(imputed_image, 65535))
    ax[2].set_title("Imputed image (pyoniip)")

    ax[3].imshow(normalize_image(imputed_image_opencv, 65535))
    ax[3].set_title("Imputed image (cv2.inpaint)")
    # draw_values(ax[2], normalize_image(imputed_image, 65535))
    #
    # cv2.imshow('image', imputed_image)
    # cv2.waitKey(0)

    return fig, ax


def write_image(image: npt.NDArray, path_to_imputed_image: Union[Path, str]):
    im1 = Image.fromarray(image)
    im1.save(path_to_imputed_image)
