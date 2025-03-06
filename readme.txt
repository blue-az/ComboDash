# ComboDash Streamlit Dashboard

This repository provides a modular, deployable Streamlit dashboard based on ComboDash, integrating data from two SQLite databases (`BabPopExt.db` and `ztennis.db`). It leverages Python for data wrangling and Plotly for interactive visualizations.

## Project Structure

```
ComboDash/
├── BabPopExt.db
├── ztennis.db
├── BabWrangle.py
├── UZeppWrangle.py
├── ComboDash.py
├── streamlit_app.py
└── requirements.txt
```

## Setup Instructions

### 1. Clone this repository

```bash
git clone <repository_url>
cd ComboDash
```

### 2. Install dependencies

Use a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Run the Streamlit app locally

```bash
streamlit run streamlit_app.py
```

## Requirements

Make sure these packages are installed:

```bash
streamlit
pandas
plotly
numpy
pytz
```

## Database Structure
- `BabPopExt.db` contains tennis swing data recorded by Babolat.
- `ztennis.db` contains tennis swing data recorded by Zepp sensor.

Wrangling scripts (`BabWrangle.py`, `UZeppWrangle.py`) preprocess and normalize these datasets for combined visualization and analysis.

## Usage

Launch the app locally with Streamlit:

```bash
streamlit run streamlit_app.py
```

The dashboard provides interactive filters and visualizations to explore and analyze tennis swing metrics.

## Deployment

Deploy easily via Streamlit Cloud or a similar hosting provider. Make sure paths to databases and dependencies are correctly set.
