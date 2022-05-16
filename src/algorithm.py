import numpy as np
import numpy.typing as npt
import cv2


def impute_image(image: npt.NDArray[np.uint16], calibration_image: npt.NDArray[np.float32]):
    # divide and conquer strategy - check pixels between 1:end-1
    for (i, j), calibration_value in np.ndenumerate(calibration_image):

        # skip boundary in this loop

        if i == 0 or j == 0 or i == calibration_image.shape[0] - 1 or j == calibration_image.shape[1] - 1:
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
            if calibration_value < 0:
                total = 0
                total_index = 0
                mean = 0
                for indices in neighbouring_indices:
                    neighbouring_calib_value = calibration_image[indices]
                    # check which pixels are 0 around the query pixel in the calibration image
                    if neighbouring_calib_value >= 0:
                        total += image[indices]
                        total_index += 1
                        # and use those to compute the average in the actual image
                mean = total // total_index
                # impute the pixel in the actual image
                image[i, j] = np.uint16(mean)
                # update the calibration image from negative to 0 for that index
                calibration_image[i, j] = 0

    return image


def impute_image_opencv(image: npt.NDArray[np.uint16], calibration_image: npt.NDArray[np.float32]):
    mask = calibration_image.copy()
    mask[np.where(mask > 0)] = 0.0
    mask[np.where(mask < 0)] = 1.0
    mask = mask.astype(np.uint8)
    result = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
    return result
