import pytest
import numpy as np
from pytest_lazyfixture import lazy_fixture
from pyoniip import impute_image as impute_image_cpp

from src import impute_image
from definitions import calibration_image_dtype, image_dtype


@pytest.fixture
def no_boundary_2x2_blob():
    calibration_image = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ], dtype=calibration_image_dtype)

    image = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1],
                      [1, 1, 1, 1, 1, 1, 1, 1, 1],
                      [1, 1, 3, 2, 1, 1, 1, 1, 1],
                      [1, 1, 6, 15, 1, 1, 1, 1, 1],
                      [1, 1, 1, 1, 1, 1, 1, 1, 1],
                      [1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=image_dtype)

    expected_corrected_image = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1],
                                         [1, 1, 1, 1, 1, 1, 1, 1, 1],
                                         [1, 1, 1, 1, 1, 1, 1, 1, 1],
                                         [1, 1, 1, 1, 1, 1, 1, 1, 1],
                                         [1, 1, 1, 1, 1, 1, 1, 1, 1],
                                         [1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=image_dtype)
    return {"image": image,
            "calibration_image": calibration_image,
            "expected_corrected_image": expected_corrected_image}


@pytest.fixture
def no_boundary_3x3_blob():
    calibration_image = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, -1, -1, -1, 0, 0, 0, 0],
        [0, 0, -1, -1, -1, 0, 0, 0, 0],
        [0, 0, -1, -1, -1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=calibration_image_dtype)

    image = np.array([[1., 1., 1., 1., 1., 1., 1., 1., 1.],
                      [1., 1., 2., 13., 4., 1., 1., 1., 1.],
                      [1., 1., 3., 2., 5., 1., 1., 1., 1.],
                      [1., 1., 6., 15., 9., 1., 1., 1., 1.],
                      [1., 1., 1., 1., 1., 1., 1., 1., 1.],
                      [1., 1., 1., 1., 1., 1., 1., 1., 1.]], dtype=image_dtype)

    expected_corrected_image = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1],
                                         [1, 1, 1, 1, 1, 1, 1, 1, 1],
                                         [1, 1, 1, 1, 1, 1, 1, 1, 1],
                                         [1, 1, 1, 1, 1, 1, 1, 1, 1],
                                         [1, 1, 1, 1, 1, 1, 1, 1, 1],
                                         [1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=image_dtype)
    return {"image": image,
            "calibration_image": calibration_image,
            "expected_corrected_image": expected_corrected_image}


@pytest.fixture
def no_boundary_4x4_blob():
    calibration_image = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                  [0., 0., -1., -1., -1., -1., 0., 0., 0.],
                                  [0., 0., -1., -1., -1., -1., 0., 0., 0.],
                                  [0., 0., -1., -1., -1., -1., 0., 0., 0.],
                                  [0., 0., -1., -1., -1., -1., 0., 0., 0.],
                                  [0., 0., 0., 0., 0., 0., 0., 0., 0.]], dtype=calibration_image_dtype)

    image = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1],
                      [1, 1, 2, 13, 4, 100, 1, 1, 1],
                      [1, 1, 3, 2, 5, 2, 1, 1, 1],
                      [1, 1, 6, 15, 9, 9, 1, 1, 1],
                      [1, 1, 5, 6, 7, 9, 1, 1, 1],
                      [1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=image_dtype)

    expected_corrected_image = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1],
                                         [1, 1, 1, 1, 1, 1, 1, 1, 1],
                                         [1, 1, 1, 1, 1, 1, 1, 1, 1],
                                         [1, 1, 1, 1, 1, 1, 1, 1, 1],
                                         [1, 1, 1, 1, 1, 1, 1, 1, 1],
                                         [1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=image_dtype)
    return {"image": image,
            "calibration_image": calibration_image,
            "expected_corrected_image": expected_corrected_image}


@pytest.fixture
def no_boundary_2_erroneous_pixels():
    calibration_image = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                  [0., 0., -1., 0., 0., 0., 0., 0., 0.],
                                  [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                  [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                  [0., 0., 0., 0., 0., -1., 0., 0., 0.],
                                  [0., 0., 0., 0., 0., 0., 0., 0., 0.]], dtype=calibration_image_dtype)

    image = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1],
                      [1, 1, 2, 13, 4, 100, 1, 1, 1],
                      [1, 1, 3, 2, 5, 2, 1, 1, 1],
                      [1, 1, 6, 15, 9, 9, 1, 1, 1],
                      [1, 1, 5, 6, 7, 9, 1, 1, 1],
                      [1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=image_dtype)

    expected_corrected_image = np.array([[1., 1., 1., 1., 1., 1., 1., 1., 1.],
                                         [1., 1., 2.875, 13., 4.,
                                             100., 1., 1., 1.],
                                         [1., 1., 3., 2., 5., 2., 1., 1., 1.],
                                         [1., 1., 6., 15., 9., 9., 1., 1., 1.],
                                         [1., 1., 5., 6., 7., 3.75, 1., 1., 1.],
                                         [1., 1., 1., 1., 1., 1., 1., 1., 1.]], dtype=image_dtype)
    return {"image": image,
            "calibration_image": calibration_image,
            "expected_corrected_image": expected_corrected_image}


@pytest.fixture
def no_boundary_2_2x2_blobs():
    calibration_image = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                  [0., 0., -1., -1., 0., 0., 0., 0., 0.],
                                  [0., 0., -1., -1., 0., 0., 0., 0., 0.],
                                  [0., 0., 0., 0., 0., 0., -1., -1., 0.],
                                  [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                  [0., 0., 0., 0., 0., 0., 0., 0., 0.]], dtype=calibration_image_dtype)

    image = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1],
                      [1, 1, 2, 13, 1, 100, 1, 1, 1],
                      [1, 1, 3, 2, 1, 2, 1, 1, 1],
                      [1, 1, 1, 1, 1, 1, 1, 1, 1],
                      [1, 1, 1, 1, 1, 1, 1, 1, 1],
                      [1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=image_dtype)

    expected_corrected_image = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1],
                                         [1, 1, 1, 1, 1, 100, 1, 1, 1],
                                         [1, 1, 1, 1, 1, 2, 1, 1, 1],
                                         [1, 1, 1, 1, 1, 1, 1, 1, 1],
                                         [1, 1, 1, 1, 1, 1, 1, 1, 1],
                                         [1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=image_dtype)
    return {"image": image,
            "calibration_image": calibration_image,
            "expected_corrected_image": expected_corrected_image}


tests = [
    lazy_fixture("no_boundary_2x2_blob"),
    lazy_fixture("no_boundary_3x3_blob"),
    lazy_fixture("no_boundary_4x4_blob"),
    lazy_fixture("no_boundary_2_erroneous_pixels"),
    lazy_fixture("no_boundary_2_2x2_blobs"),

]


@pytest.mark.parametrize("data", tests)
def test_impute_image_python(data):
    result = impute_image(
        image=data["image"], calibration_image=data["calibration_image"])
    print(result)
    assert np.all(np.abs(result - data["expected_corrected_image"]) < 1e-8)


@pytest.mark.parametrize("data", tests)
def test_impute_image_cpp(data):
    result = impute_image_cpp(data["image"], data["calibration_image"])
    assert np.all(np.abs(result - data["expected_corrected_image"]) < 1e-8)
