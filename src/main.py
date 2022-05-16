from pathlib import Path
from typing import Union
import numpy as np
import numpy.typing as npt

from definitions import DATA_DIR

import matplotlib.pyplot as plt

from src import impute_image
from src.algorithm import impute_image_opencv
from src.utils import load_image, plot_image_statistics, plot_processing_results, write_image


def main(path_to_image: Union[str, Path], path_to_calibration_image: Union[str, Path],
         path_to_imputed_image: Union[str, Path], is_plot: bool) -> npt.NDArray[
        np.uint16]:
    imputed_image = impute_image(image=load_image(path_to_image),
                                 calibration_image=load_image(path_to_calibration_image))
    write_image(imputed_image, path_to_imputed_image)
    if is_plot:
        fig, ax = plot_image_statistics(image=load_image(path_to_image),
                                        calibration_image=load_image(path_to_calibration_image), )

        fig, ax = plot_processing_results(
            image=load_image(path_to_image),
            calibration_image=load_image(path_to_calibration_image),
            imputed_image=imputed_image,
            imputed_image_opencv=impute_image_opencv(image=load_image(path_to_image),
                                                     calibration_image=load_image(path_to_calibration_image))
        )

        plt.show()

    return imputed_image


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

    # print(args.input)
    corrected_image = main(
        args.image, args.calibration_image, args.output_image, args.plot)
