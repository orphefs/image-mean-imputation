from src.utils import load_image, plot_diagnostics, write_image
from src.algorithm import impute_image_opencv
from pyoniip import impute_image as impute_image_pyoniip
from src import impute_image
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Union
import numpy as np
import numpy.typing as npt

from definitions import DATA_DIR
import matplotlib
matplotlib.use('TkAgg')


def main(path_to_image: Union[str, Path], path_to_calibration_image: Union[str, Path],
         path_to_imputed_image: Union[str, Path], is_plot: bool):
    """
    Imputes image using nearest-neighbour averaging and serializes the result to disk.
    :param path_to_image: Union[str, Path]
    :param path_to_calibration_image: Union[str, Path]
    :param path_to_imputed_image: Union[str, Path]
    :param is_plot: bool
    """
    imputed_image_pyoniip = impute_image_pyoniip(load_image(path_to_image).astype(np.uint16),
                                                 load_image(path_to_calibration_image).astype(np.float32))
    imputed_image_python = impute_image(load_image(path_to_image).astype(np.uint16),
                                        load_image(path_to_calibration_image).astype(np.float32))
    write_image(imputed_image_pyoniip, path_to_imputed_image)

    if is_plot:

        fig, ax = plot_diagnostics(
            image=load_image(path_to_image).astype(np.uint16),
            calibration_image=load_image(
                path_to_calibration_image).astype(np.float32),
            imputed_image_python=imputed_image_python.astype(np.uint16),
            imputed_image_cpp=imputed_image_pyoniip.astype(np.uint16),
            imputed_image_opencv=impute_image_opencv(image=load_image(path_to_image).astype(np.uint16),
                                                     calibration_image=load_image(path_to_calibration_image).astype(np.float32))
        )

        plt.show()


if __name__ == '__main__':
    import argparse

    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('--image', help="Path to the uint16 image", action='store', type=str,
                           required=False, default=DATA_DIR / "raw" / "bead_image.tif")
    my_parser.add_argument('--calibration_image', help="Path to the float calibration image", action='store',
                           type=str, required=False, default=DATA_DIR / "raw" / "hotpixel_calibration.tif")
    my_parser.add_argument('--output_image', help="Path to the imputed image", action='store',
                           type=str, required=False, default=DATA_DIR / "processed" / "imputed_image.tif")
    my_parser.add_argument('--plot', help="Display diagnostic plots", action='store',
                           type=str, required=False, default=False)
    args = my_parser.parse_args()

    main(args.image, args.calibration_image, args.output_image, args.plot)
