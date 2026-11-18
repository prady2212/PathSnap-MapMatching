# PathSnap-MapMatching

PathSnap prepares road-network/GPS trip data, trains candidate-road scoring models,
and reconstructs routes with an HMM/Viterbi decoder. The Streamlit UI can run with
illustrative mock data or connect to the FastAPI service.

## Setup

Use Python 3.10 or later.

From the project directory, create and activate a virtual environment, then install
the project dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Prepare the data and model

The checked-in raw trip files are inputs; processed features and trained model
artifacts are generated locally. From the project directory, start Jupyter:

```powershell
jupyter lab
```

Run all cells in these notebooks in order:

1. `data_processing/01_Data_Preprocessing_and_Cleaning.ipynb`
2. `feature_engineering/02_Feature_Engineering.ipynb`
3. `model_training/03_Model_Training.ipynb`
4. `hyperparameter_optimisation/optimize.ipynb`
5. `hmm_viterbi/hmm_route_reconstruction.ipynb`

The last notebook exports the model bundle and manifest required by the API. The
API will report a missing-input error until the notebooks have completed and
created `data/processed/features`, `data/processed/cleaned`, and
`artifacts/map_matching_models`.

## Run the application

Start the API from the project directory:

```powershell
uvicorn api.main:app --reload
```

In a second terminal, start the UI:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run ui\app.py
```

The UI opens in mock mode by default. Turn off **Use Mock Data (Testing Mode)** to
call the API at `http://localhost:8000`. The API docs are available at
`http://localhost:8000/docs`; `GET /health` checks readiness and `POST /match`
scores a trip and returns its route geometry.

To run only the UI without preparing data or starting the API, use mock mode.

## Tests

From the project directory, run:

```powershell
python -m unittest discover -s tests -v
```
