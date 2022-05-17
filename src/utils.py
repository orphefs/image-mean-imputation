import copy
from pathlib import Path
from typing import Union, Any, Tuple, Dict

import cv2
import matplotlib
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt, colors
from matplotlib.ticker import FormatStrFormatter, LogFormatterSciNotation
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


def draw_values(ax: plt.Axes, image: npt.NDArray[Any], indices: npt.NDArray[Any]):
    for (i, j) in list(zip(*indices)):
        ax.text(i, j, image[i, j], ha='center', va='center')


def plot_diagnostics(image: npt.NDArray[np.uint16],
                     calibration_image: npt.NDArray[np.float32],
                     imputed_image_python: npt.NDArray[np.uint16],
                     imputed_image_cpp: npt.NDArray[np.uint16],
                     imputed_image_opencv: npt.NDArray[np.uint16]) -> Tuple[plt.figure, plt.axes]:
    def get_colormap_kwargs(image: npt.NDArray, colormap: matplotlib.colors.Colormap) -> Dict:

        return {"norm": colors.LogNorm(vmin=np.max([image.min(), 1]), vmax=image.max()), "cmap": colormap}

    def get_hist_kwargs(image: npt.NDArray, hist_bins: int = 100) -> Dict:

        d = {"bins": np.logspace(start=np.log10(
            np.max([image.min(), 1])), stop=np.log10(image.max()), num=hist_bins)}
        return d

    # init configuration
    calibration_image[np.where(calibration_image >= 0.0)] = 255
    calibration_image[np.where(calibration_image < 0.0)] = 0
    difference = np.abs(image - imputed_image_cpp)

    # init axes
    fig, ax = plt.subplots(nrows=2, ncols=6)
    ax[0, 0].get_shared_x_axes().join(*ax[0, :])
    ax[0, 0].get_shared_y_axes().join(*ax[0, :])
    ax[1, 0].get_shared_x_axes().join(*ax[1, :])
    ax[1, 0].get_shared_y_axes().join(*ax[1, :])

    data = [image, calibration_image, imputed_image_python, imputed_image_cpp, imputed_image_opencv,
            difference]
    titles = ["Original image", "Calibration image", "Imputed image (python)", "Imputed image (pyoniip)",
              "Imputed image (cv2.inpaint)", "abs[original - pyoniip]"]
    colormaps = ["hsv", "hsv", "hsv", "hsv", "hsv", "hsv"]
    # images
    for index, (current_image, title, colormap) in enumerate(zip(data, titles, colormaps)):
        ax[0, index].imshow(
            current_image, **get_colormap_kwargs(current_image, colormap))
        ax[0, index].set_title(title)
        ax[0, index].axis('off')

    # histograms
    for index, current_image in enumerate(
            [image, calibration_image, imputed_image_python, imputed_image_cpp, imputed_image_opencv,
             difference]):
        if index in [1, 5]:
            ax[1, index].set_visible(False)
        else:
            ax[1, index].hist(current_image.flatten(), **
                              get_hist_kwargs(current_image))
            ax[1, index].set_title("Histogram")
            ax[1, index].set_xscale("log")
            ax[1, index].yaxis.set_major_formatter(LogFormatterSciNotation())

    return fig, ax


def write_image(image: npt.NDArray, path_to_imputed_image: Union[Path, str]):
    im1 = Image.fromarray(image)
    im1.save(path_to_imputed_image)
