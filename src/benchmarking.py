# importing the required modules
import timeit
import numpy as np
import matplotlib.pyplot as plt


def generate_data(size):
    percent = 0.2
    height = size
    width = size
    image = np.random.randint(0, high=65535, size=width * height, dtype=np.uint16).reshape(
        [height, width])
    calibration_pixels_i = np.random.choice(
        np.arange(0, size), size=int(percent * height), replace=False)
    calibration_pixels_j = np.random.choice(
        np.arange(0, size), size=int(percent * width), replace=False)
    calibration_image = np.zeros([height, width], dtype=np.float32)
    calibration_image[(calibration_pixels_i, calibration_pixels_j)] = -1.0
    return image, calibration_image


def execution_time(size, module):
    SETUP_CODE = '''
from __main__ import generate_data
import numpy as np
from {} import impute_image
'''.format(module)

    TEST_CODE = '''
image, calibration_image = generate_data({})
impute_image(image, calibration_image)'''.format(size)

    # timeit.repeat statement
    times = timeit.repeat(setup=SETUP_CODE,
                          stmt=TEST_CODE,
                          repeat=3,
                          number=5,
                          )

    # printing minimum exec. time
    print(
        "Image size: {size}x{size} | {module} time: {time}".format(size=size, time=min(times), module=module))
    return min(times)


def benchmark():
    pyoniip_execution_time = []
    python_execution_time = []

    sizes = np.logspace(1.5, 3, 15)
    for size in sizes:
        pyoniip_execution_time.append(
            execution_time(size=int(size), module="pyoniip"))
        python_execution_time.append(
            execution_time(size=int(size), module="src"))

    plt.plot(sizes, python_execution_time)
    plt.plot(sizes, pyoniip_execution_time)
    plt.title("Python vs. C++ (20% erroneous pixels)")
    plt.yscale('log')
    plt.xlabel("Image side (pixels)")
    plt.ylabel("Time (s)")
    plt.legend(["Python", "pyoniip"])
    plt.show()


if __name__ == "__main__":
    benchmark()
