# Wildfire Prediction App

This Streamlit app combines two wildfire prediction approaches for locations in Québec:

- **Computer vision model (VGG16):** downloads 5 satellite tiles around a selected point and predicts infrastructural wildfire likelihood from imagery.
- **Meteorological model (LSTM):** snaps the selected point to an H3 cell, fetches ERA5-Land climate data, and predicts wildfire likelihood from environmental conditions.


## Screenshot of the interface
<img width="1051" height="1029" alt="image" src="https://github.com/nicotauchmann/fire_prediction/blob/916b541d1a7edfb94376022d2b5117830455c163/Screenshot.jpg" />

## Features

- Interactive map-based location selection
- Satellite-image prediction using Mapbox imagery
- ERA5-Land + LSTM prediction for the selected H3 cell
- Side-by-side results in one app
- CSV downloads for prediction outputs

## Link to streamlit.io:
https://firepredictionapp.streamlit.app/
