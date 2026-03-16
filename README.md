# Wildfire Prediction App

This Streamlit app combines two wildfire prediction approaches for locations in Québec:

- **Computer vision model (VGG16):** downloads 5 satellite tiles around a selected point and predicts wildfire likelihood from imagery.
- **Meteorological model (LSTM):** snaps the selected point to an H3 cell, fetches ERA5-Land climate data, and predicts wildfire likelihood from environmental conditions.

## Features

- Interactive map-based location selection
- Satellite-image prediction using Mapbox imagery
- ERA5-Land + LSTM prediction for the selected H3 cell
- Side-by-side results in one app
- CSV downloads for prediction outputs

## Link to streamlit.io:
https://firepredictionapp.streamlit.app/
