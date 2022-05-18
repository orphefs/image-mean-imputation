Image pixel-wise mean imputation
================================



![tests workflow](https://github.com/orphefs/oni-image-processing-runner/actions/workflows/tests.yml/badge.svg)

This project is about correcting microscope images by using a calibration mask. The main runner (image
loading, plotting) and tests are written in Python, while the core algorithm is implemented in both Python and
pybind11 in C++17.

## Quick summary

To clone the repository, run the algorithm, tests, and produce plots, run the following in a bash shell:

```bash
git clone https://github.com/orphefs/oni-image-processing-runner.git
cd oni-image-processing-runner
make virtualenv
source venv/bin/activate
python -m pytest
python -m src.main --plot True
```

## Repository structure

```bash
.
├── data
│   ├── processed
│   └── raw
├── docs
├── src
└── test
```

## Installation

### Clone

```bash
git clone git@github.com:orphefs/oni-image-processing-runner.git
```

### Python environment

The recommended Python version for this project is Python 3.8.10. To create a `venv` environment and install the
dependencies, run

```bash
make virtualenv
```

This will create a `.venv` directory in the project root. You can then source that environment via

```bash
source $ROOT_DIR/venv/bin/activate
```

where variable `ROOT_DIR` contains the path to the project root folder.

:warning: Do not directly modify `requirements.txt`. Please modify `unpinned_requirements.txt` instead and
run `make update-requirements-txt`, followed by `make virtualenv` to update your local environment.

### Coding style

The repository conforms to PEP8 guidelines. A [pre-commit](.pre-commit-config.yaml) specification with the
default `autopep8` configuration is used.

### Methodology

The main algorithm is written in C++ and binded to Python via `pybind11`. The module repo
is [here](https://github.com/orphefs/pyoniip) and can be manually installed using the following command:

```bash
pip install git+https://github.com/orphefs/pyoniip
```

The analysis and development was conducted on a machine running Ubuntu 20.04 with a 5.13.0-40-generic kernel.

#### Stack

- vscode (C++)
- pycharm (Python)
- bash
- git
- QGIS Desktop (image analysis)

#### Steps

- Image mimetypes were verified using `file` and TIF info obtained via `tiffinfo`. This step was conducted to
  verify image headers and bit depth.
- Developed code for [diagnostic plotting](src/utils.py) (images and histograms).
- Pseudocode and [test cases](test/test_impute_image.py) were formulated, and solutions calculated manually in
  order to **verify correctness**.
- Python code was developed to solve all test cases.
- C++ code was developed to optimize the main [algorithm](src/algorithm.py), and tested using the available
  tests. Code was [packaged](https://github.com/orphefs/pyoniip) using `pybind11` and compiled using CMake and
  setuptools.

> NOTE: The algorithm skips the boundaries and only corrects for non-edge pixels.

#### Results and evaluation

##### Comparison with SOTA technique

OpenCV's [cv.INPAINT_TELEA inpainting algorithm](https://docs.opencv.org/4.x/df/d3d/tutorial_py_inpainting.html)
was used for comparison purposes. Even through this algorithm is more complicated as it uses
the [Fast Marching method](http://www.olivier-augereau.com/docs/2004JGraphToolsTelea.pdf), which is a
weighted, boundary-first method, it is a good reference basis for comparison. The `pyoniip` algorithm performs
well, successfully imputing all erroneous pixels. A better implementation would be a median filter as it would 
perform better in the presence of noisy pixels.

![alt text](pyoniip_results.gif "Results on sample image and comparison with SOTA")

For completeness, the `pyoniip` algorithm was compared with its Python
counterpart, [benchmarked](src/benchmarking.py) using a sequence of randomly generated images of dimensions
NxN. The C++ implementation clearly outperforms the Python implementation, as expected. The algorithm performs
in O[N] time.

![alt text](comparison.png "Comparison between Python and C++ implementation")

## Usage

To run the computation pipeline, run the script `src/main.py` while in the virtual environment. Plotting is
off by default. To run with plotting on:

```bash
python -m src.main --plot True
```

Output of `python -m src.main --help`:

```bash
usage: main.py [-h] [--image IMAGE] [--calibration_image CALIBRATION_IMAGE] [--output_image OUTPUT_IMAGE] [--plot PLOT]

optional arguments:
  -h, --help            show this help message and exit
  --image IMAGE         Path to the uint16 image
  --calibration_image CALIBRATION_IMAGE
                        Path to the float calibration image
  --output_image OUTPUT_IMAGE
                        Path to the imputed image
  --plot PLOT           Display diagnostic plots

```

For benchmarking, run:

```
python -m src.benchmarking
```

## Tests

To run tests, [source the pip environment](#installation) and run `pytest` on the root directory. The command
will collect all tests in the `test` directory.

This repository also implements a test automation workflow on Github Actions.
