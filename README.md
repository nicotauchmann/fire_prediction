# Wildfire Prediction App

This Streamlit app combines two wildfire prediction approaches for locations in Québec:

- **Computer vision model (VGG16):** downloads satellite image tiles around a selected point and predicts infrastructural wildfire likelihood from imagery. This produces an estimate of how likely a fire is to occur at a given location based on its physical and infrastructural characteristics.
- **Meteorological model (LSTM):** snaps the selected point to an H3 cell, fetches ERA5-Land climate data (7 input variables, e.g. current temperature and wind), and predicts wildfire likelihood from environmental conditions. This produces an estimate of how likely a fire is to occur right now based on environmental characteristics.

The model was trained on satellite pictures and historical weather data of locations in the Quebec province of Canada. Due to limitations regarding data availability, the meteorological data and satellite images might not be current.
You can find the relevant information on forest fires here: https://www.donneesquebec.ca/recherche/dataset/feux-de-foret/resource/1edec7b1-c593-45f2-9cb1-20752633b1a0


## Link to streamlit.io:
https://firepredictionapp.streamlit.app/


## Screenshot of the interface
<img width="1051" height="1029" alt="image" src="https://github.com/nicotauchmann/fire_prediction/blob/916b541d1a7edfb94376022d2b5117830455c163/Screenshot.jpg" />


## Features

- Interactive map-based location selection
- Satellite-image prediction using Mapbox imagery
- ERA5-Land + LSTM prediction for the selected H3 cell
- Side-by-side results in one app
- CSV downloads for prediction outputs

