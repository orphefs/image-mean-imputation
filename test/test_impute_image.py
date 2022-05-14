import pytest
import numpy as np
from pytest_lazyfixture import lazy_fixture

from src.main import impute_image


# Arrange
@pytest.fixture
def no_boundary_2x2_blob():
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

    expected_corrected_image = np.array([[1., 1., 1., 1., 1., 1., 1., 1., 1.],
                                         [1., 1., 1., 1., 1., 1., 1., 1., 1.],
                                         [1., 1., 1., 1., 1., 1., 1., 1., 1.],
                                         [1., 1., 1., 1., 1., 1., 1., 1., 1.],
                                         [1., 1., 1., 1., 1., 1., 1., 1., 1.],
                                         [1., 1., 1., 1., 1., 1., 1., 1., 1.]])
    return image, calibration_image, expected_corrected_image


@pytest.fixture
def no_boundary_3x3_blob():
    calibration_image = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, -1, -1, -1, 0, 0, 0, 0],
        [0, 0, -1, -1, -1, 0, 0, 0, 0],
        [0, 0, -1, -1, -1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])

    image = np.array([[1., 1., 1., 1., 1., 1., 1., 1., 1.],
                      [1., 1., 2., 13., 4., 1., 1., 1., 1.],
                      [1., 1., 3., 2., 5., 1., 1., 1., 1.],
                      [1., 1., 6., 15., 9., 1., 1., 1., 1.],
                      [1., 1., 1., 1., 1., 1., 1., 1., 1.],
                      [1., 1., 1., 1., 1., 1., 1., 1., 1.]])

    expected_corrected_image = np.array([[1., 1., 1., 1., 1., 1., 1., 1., 1.],
                                         [1., 1., 1., 1., 1., 1., 1., 1., 1.],
                                         [1., 1., 1., 1., 1., 1., 1., 1., 1.],
                                         [1., 1., 1., 1., 1., 1., 1., 1., 1.],
                                         [1., 1., 1., 1., 1., 1., 1., 1., 1.],
                                         [1., 1., 1., 1., 1., 1., 1., 1., 1.]])
    return image, calibration_image, expected_corrected_image


@pytest.fixture
def no_boundary_4x4_blob():
    calibration_image = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, -1, -1, -1, -1, 0, 0, 0],
        [0, 0, -1, -1, -1, -1, 0, 0, 0],
        [0, 0, -1, -1, -1, -1, 0, 0, 0],
        [0, 0, -1, -1, -1, -1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])

    image = np.array([[1., 1., 1., 1., 1., 1., 1., 1., 1.],
                      [1., 1., 2., 13., 4., 100., 1., 1., 1.],
                      [1., 1., 3., 2., 5., 2., 1., 1., 1.],
                      [1., 1., 6., 15., 9., 9., 1., 1., 1.],
                      [1., 1., 5., 6., 7., 9., 1., 1., 1.],
                      [1., 1., 1., 1., 1., 1., 1., 1., 1.]])

    expected_corrected_image = np.array([[1., 1., 1., 1., 1., 1., 1., 1., 1.],
                                         [1., 1., 1., 1., 1., 1., 1., 1., 1.],
                                         [1., 1., 1., 1., 1., 1., 1., 1., 1.],
                                         [1., 1., 1., 1., 1., 1., 1., 1., 1.],
                                         [1., 1., 1., 1., 1., 1., 1., 1., 1.],
                                         [1., 1., 1., 1., 1., 1., 1., 1., 1.]])
    return image, calibration_image, expected_corrected_image


@pytest.mark.parametrize("data", [
    lazy_fixture("no_boundary_2x2_blob"),
    lazy_fixture("no_boundary_3x3_blob"),
    lazy_fixture("no_boundary_4x4_blob"),

])
def test_impute_image(data):
    result = impute_image(image=data[0], calibration_image=data[1])
    print(result)
    assert np.all(result == data[2])
