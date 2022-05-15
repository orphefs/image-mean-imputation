# Ride demand forecasting

This project is about forecasting ride demand prediction on a dataset provided by Beat. This [report](docs/report.md) details data preparation and modeling methodology, as well as experimental design.

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

### Python environment

The recommended Python version for this project is Python 3.8. To create a `venv` environment and install the dependencies, run

```bash
make virtualenv
```

This will create a `.venv` directory in the project root. You can then source that environment via

```bash
source $ROOT_DIR/.venv/bin/activate
```

where variable `ROOT_DIR` contains the path to the project root folder.

### Data

To run the ETL pipeline, run

```bash
make data
```

in the command line. Make sure you are running this within the virtual environment.

This command will run the following steps in sequence:

1. Download the source data from Google Drive into `data/raw/routes.csv`.
2. The notebook [eda.ipynb](src/notebooks/eda.ipynb) (in headless mode), which takes as input `data/raw/routes.csv` and produces `data/processed/routes.csv`, which is a cleaned version of the input data.
3. The notebook [temporal_analysis.ipynb](src/notebooks/temporal_analysis.ipynb) (in headless mode), which produces a number of `.csv` files in the `data/processed/` directory which are used for training and testing the model.

:warning: `make data` is unoptimized, so it might take ~10-15 mins, depending on the machine specs.

### Docker installation

To run inference, you will need to install the [Docker engine](https://docs.docker.com/engine/install/) on Linux or [Docker Desktop](https://docs.docker.com/desktop/) on Windows or Mac, and you should make sure that the docker daemon is up and runnning prior to building or running the image.

## Usage

To run inference, a Dockerfile is provided, containing a Flask backend with an HTMl frontend.

![alt text](app_workflow.gif "Running inference using Docker")

### Building and running the Dockerfile

To build the Docker image, navigate to `ROOT_DIR` and run

```bash
sudo docker build --no-cache -t volume-prediction .
```

This will build the image locally, which you can then run using

```bash
sudo docker run -d -p 5000:5000 volume-prediction
```

The Flask app is then accessible on <http://127.0.0.1:5000/> .

### Running inference and viewing results

While on the app,

1. Select the `.csv` file you would like to obtain predictions for. Sample `hourly_volume_test_*.csv` are provided in the `data/processed/` folder. Each file has the following format:

    The contents of the latter `.csv` files is as follows:

    ```bash
    2015-12-05 18:00:00,494
    2015-12-05 19:00:00,550
    2015-12-05 20:00:00,768
    2015-12-05 21:00:00,734
    2015-12-05 22:00:00,577
    2015-12-05 23:00:00,497
    ```

    where the first column is a `pd.DateTime` index indicating a specific hour, and the second column is an `int` quantifying request volume for that hour.

2. Enter the SARIMA (Seasonal AutoRegressive Integrated Moving Average) model parameters. There are four parameters, `p` (trend autoregression order), `d` (Trend difference order), `q` (Trend moving average order), `S` (seasonality, or the number of time steps for a single seasonal period). The best parameters for this dataset have been identified as (2, 1, 1, 24), for reasons explained further in the documentation.

3. Enter the forecast horizon in hours.

4. Click on `Run Inference`.

5. Plot should be updated with the new results, indicating the RMSE (root mean square error) between actual values and predictions, and AIC (Akaike Information Criterion) of the model fit to the data.

### Train

The training process in the case of SARIMA model takes the form of hyperparameter tuning. in the parameters `p`, `d`, `q`, and `S`. The script [train.py](src/train.py).

The script uses a rolling prediction algorithm on the training dataset, using a grid search to find the best parameter set. Using this [training dataset](data/processed/hourly_volume_train_val.csv), the best parameter yielded was

```bash
INFO:root:Best ARIMA(1, 1, 2, 24) MSE=236.821
```