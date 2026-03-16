import io
import math
import os
import tempfile
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import cdsapi
import folium
import h3
import numpy as np
import pandas as pd
import requests
import streamlit as st
import xarray as xr
from PIL import Image
from branca.element import Element, MacroElement, Template
from folium import CircleMarker, Marker
from streamlit_folium import st_folium


# ============================================================
# CONFIG
# ============================================================
APP_TITLE = "Wildfire Prediction — Computer Vision + Meteorology"

# --- Computer vision / satellite ---
STYLE_USER = "mapbox"
STYLE_ID = "satellite-v9"
ZOOM = 15
BEARING = 0
TILE_SIZE = 350
RESCALE = 1.0 / 255.0
SPACING_KM = 3.0
CV_WILDFIRE_INDEX = 1
LIKELY_THRESHOLD = 0.8
CV_MODEL_PATH = Path("saved_model") / "vgg16_model.keras"

# --- Meteorological / ERA5 + LSTM ---
H3_RES = 5
SEQ_LEN = 12
N_FEATURES = 7
LSTM_WILDFIRE_INDEX = 1
LSTM_THRESHOLD = 0.5
LSTM_MODEL_PATH = Path("saved_model") / "lstm_model.keras"
SCALER_PATH = Path("saved_model") / "scaler.pkl"  # optional

FEATURE_NAMES = [
    "2m_temperature",
    "volumetric_soil_water_layer_1",
    "surface_solar_radiation_downwards",
    "total_evaporation",
    "wind_total",
    "total_precipitation",
    "leaf_area_index_high_vegetation",
]

ERA5_VARIABLES = [
    "2m_temperature",
    "volumetric_soil_water_layer_1",
    "surface_solar_radiation_downwards",
    "total_evaporation",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "total_precipitation",
    "leaf_area_index_high_vegetation",
]

# --- UI defaults ---
DEFAULT_CENTER = (52.0, -71.0)  # Québec
DEFAULT_ZOOM_PICK = 5
ESRI_TILE_URL = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/"
    "MapServer/tile/{z}/{y}/{x}"
)
ESRI_ATTR = "Esri World Imagery"

SESSION = requests.Session()


# ============================================================
# GENERIC HELPERS
# ============================================================
def rerun_app():
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


def secret_or_env(name: str, default: str = "") -> str:
    value = st.secrets.get(name, os.getenv(name, default))
    return (value or "").strip()


def latest_era5_reference_date() -> datetime:
    return datetime.utcnow().replace(day=1) - timedelta(days=90)


def add_map_legend(map_obj):
    template = """
    {% macro html(this, kwargs) %}
    <div style="
        position: absolute;
        bottom: 20px;
        right: 20px;
        z-index: 9999;
        background: rgba(255,255,255,0.92);
        border: 1px solid #bdbdbd;
        border-radius: 10px;
        padding: 10px 12px;
        font-size: 12px;
        color: #333333;
        box-shadow: 0 1px 6px rgba(0,0,0,0.25);
        line-height: 1.35;
    ">
      <div style="font-weight:700; margin-bottom:6px;">Legend</div>
      <div><span style="color:#cc2222;">■</span> LSTM high risk</div>
      <div><span style="color:#dd8800;">■</span> LSTM moderate risk</div>
      <div><span style="color:#eecc00;">■</span> LSTM low risk</div>
      <div><span style="color:#33aa33;">■</span> LSTM minimal risk</div>
      <div style="margin:6px 0; border-top:1px solid #d0d0d0;"></div>
      <div><span style="color:red;">●</span> CV tile ≥ 0.80</div>
      <div><span style="color:blue;">●</span> CV tile &lt; 0.80</div>
      <div><span style="color:gray;">●</span> CV tile error</div>
    </div>
    {% endmacro %}
    """
    macro = MacroElement()
    macro._template = Template(template)
    map_obj.get_root().add_child(macro)


def add_map_resize_fix(map_obj):
    map_name = map_obj.get_name()
    resize_js = f"""
    <script>
    setTimeout(function() {{
        try {{
            {map_name}.invalidateSize(true);
        }} catch (e) {{}}
    }}, 300);
    setTimeout(function() {{
        try {{
            {map_name}.invalidateSize(true);
        }} catch (e) {{}}
    }}, 900);
    </script>
    """
    map_obj.get_root().html.add_child(Element(resize_js))


# ============================================================
# COMPUTER VISION HELPERS
# ============================================================
def get_mapbox_token() -> str:
    token = secret_or_env("MAPBOX_ACCESS_TOKEN")
    if not token:
        raise RuntimeError(
            "MAPBOX_ACCESS_TOKEN is missing. Add it to Streamlit secrets or your environment."
        )
    return token


def build_mapbox_url(lon: float, lat: float, token: str) -> str:
    lon = round(float(lon), 6)
    lat = round(float(lat), 6)
    base = f"https://api.mapbox.com/styles/v1/{STYLE_USER}/{STYLE_ID}/static/"
    coords = f"{lon},{lat}"
    rest = (
        f",{ZOOM},{BEARING}/{TILE_SIZE}x{TILE_SIZE}"
        f"?access_token={token}&logo=false&attribution=false"
    )
    return base + coords + rest


def cross5_from_center(lat: float, lon: float):
    dlat = SPACING_KM / 110.574
    dlon = SPACING_KM / (111.320 * math.cos(math.radians(lat)))
    pts = [
        ("center", lat, lon),
        ("north", lat + dlat, lon),
        ("south", lat - dlat, lon),
        ("east", lat, lon + dlon),
        ("west", lat, lon - dlon),
    ]
    return [(name, round(la, 6), round(lo, 6)) for name, la, lo in pts]


def preprocess_pil(img: Image.Image) -> np.ndarray:
    x = np.asarray(img, dtype=np.float32) * RESCALE
    return np.expand_dims(x, axis=0)


def predict_wildfire_prob_cv(model, img: Image.Image) -> float:
    x = preprocess_pil(img)
    y = np.array(model.predict(x, verbose=0))
    if y.ndim == 2 and y.shape[1] == 2:
        return float(y[0, CV_WILDFIRE_INDEX])
    if y.ndim == 2 and y.shape[1] == 1:
        return float(y[0, 0])
    raise ValueError(f"Unexpected CV model output shape: {y.shape}")


def compute_fire_rating(df: pd.DataFrame, threshold: float = LIKELY_THRESHOLD):
    probs = pd.to_numeric(df.get("p_wildfire"), errors="coerce")
    likely_count = int((probs >= threshold).sum())

    if likely_count == 0:
        stars = 0
        msg = "A fire is unlikely in this environment."
    elif likely_count == 1:
        stars = 1
        msg = "The fire potential of this environment is low."
    elif likely_count in (2, 3):
        stars = 2
        msg = "The fire potential of this environment is moderate. Check local safety precautions."
    else:
        stars = 3
        msg = "The fire potential of this environment is high. Check local safety precautions."

    emoji = "🔥" * stars if stars > 0 else "—"
    return likely_count, stars, emoji, msg


@st.cache_resource(show_spinner=False)
def load_cv_model_cached():
    import tensorflow as tf

    if not CV_MODEL_PATH.exists():
        raise FileNotFoundError(f"CV model file not found: {CV_MODEL_PATH.resolve()}")
    return tf.keras.models.load_model(CV_MODEL_PATH)


@st.cache_data(ttl=3600, show_spinner=False)
def download_bytes(url: str) -> bytes:
    r = SESSION.get(url, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    return r.content


def fetch_tile(lon: float, lat: float, token: str) -> Image.Image:
    url = build_mapbox_url(lon, lat, token)
    content = download_bytes(url)
    img = Image.open(io.BytesIO(content)).convert("RGB")
    if img.size != (TILE_SIZE, TILE_SIZE):
        img = img.resize((TILE_SIZE, TILE_SIZE))
    return img


# ============================================================
# LSTM / ERA5 HELPERS
# ============================================================
def get_cds_client():
    url = secret_or_env("CDS_URL")
    key = secret_or_env("CDS_KEY")
    if not url or not key:
        raise RuntimeError(
            "CDS credentials are missing. Add CDS_URL and CDS_KEY to Streamlit secrets."
        )
    return cdsapi.Client(url=url, key=key, quiet=True)


@st.cache_resource(show_spinner=False)
def load_lstm_model_cached():
    import tensorflow as tf

    if not LSTM_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"LSTM model file not found: {LSTM_MODEL_PATH.resolve()}"
        )
    return tf.keras.models.load_model(LSTM_MODEL_PATH)


@st.cache_resource(show_spinner=False)
def load_scaler_cached():
    if not SCALER_PATH.exists():
        return None
    import pickle

    with open(SCALER_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_era5_sequence(lat: float, lon: float, end_date_str: str) -> pd.DataFrame:
    end = datetime.strptime(end_date_str, "%Y-%m-%d")
    latest = latest_era5_reference_date()
    if end > latest:
        end = latest

    months_list = []
    d = end.replace(day=1)
    for _ in range(SEQ_LEN):
        months_list.append(d)
        if d.month == 1:
            d = d.replace(year=d.year - 1, month=12)
        else:
            d = d.replace(month=d.month - 1)
    months_list = sorted(months_list)

    years = sorted({str(d.year) for d in months_list})
    months = sorted({str(d.month).zfill(2) for d in months_list})

    area = [
        round(lat + 0.5, 2),
        round(lon - 0.5, 2),
        round(lat - 0.5, 2),
        round(lon + 0.5, 2),
    ]

    client = get_cds_client()
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        tmp_path = tmp.name

    client.retrieve(
        "reanalysis-era5-land-monthly-means",
        {
            "product_type": "monthly_averaged_reanalysis",
            "variable": ERA5_VARIABLES,
            "year": years,
            "month": months,
            "time": "00:00",
            "area": area,
            "format": "netcdf",
        },
        tmp_path,
    )

    if zipfile.is_zipfile(tmp_path):
        extract_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(tmp_path, "r") as zf:
            zf.extractall(extract_dir)
        candidates = list(Path(extract_dir).glob("*.nc")) + list(
            Path(extract_dir).glob("*.netcdf")
        )
        if not candidates:
            raise RuntimeError("ZIP from CDS contained no NetCDF file.")
        tmp_path = str(candidates[0])

    ds = xr.open_dataset(tmp_path, engine="netcdf4")
    ds_pt = ds.sel(latitude=lat, longitude=lon, method="nearest")
    time_coord = "valid_time" if "valid_time" in ds_pt.coords else "time"

    records = []
    for d in months_list:
        month_str = d.strftime("%Y-%m")

        def _val(var_name: str):
            try:
                times = ds_pt[time_coord].values
                mask = [(str(t)[:7] == month_str) for t in times]
                idx = next(i for i, m in enumerate(mask) if m)
                return float(ds_pt[var_name].isel({time_coord: idx}).values)
            except Exception:
                return np.nan

        u = _val("u10")
        v = _val("v10")
        records.append(
            {
                "date": month_str,
                "2m_temperature": _val("t2m"),
                "volumetric_soil_water_layer_1": _val("swvl1"),
                "surface_solar_radiation_downwards": _val("ssrd"),
                "total_evaporation": _val("e"),
                "wind_total": math.sqrt(u**2 + v**2)
                if not (np.isnan(u) or np.isnan(v))
                else np.nan,
                "total_precipitation": _val("tp"),
                "leaf_area_index_high_vegetation": _val("lai_hv"),
            }
        )

    ds.close()
    try:
        os.unlink(tmp_path)
    except OSError:
        pass

    return pd.DataFrame(records)


def run_lstm(model, scaler, df: pd.DataFrame) -> float:
    x = df[FEATURE_NAMES].values.astype(np.float32)

    if scaler is not None:
        flat = x.reshape(-1, N_FEATURES)
        flat = scaler.transform(flat)
        x = flat.reshape(SEQ_LEN, N_FEATURES)

    x = np.expand_dims(x, axis=0)
    y = np.array(model.predict(x, verbose=0))

    if y.ndim == 2 and y.shape[1] == 2:
        return float(y[0, LSTM_WILDFIRE_INDEX])
    if y.ndim == 2 and y.shape[1] == 1:
        return float(y[0, 0])
    if y.ndim == 1:
        return float(y[0])
    raise ValueError(f"Unexpected LSTM model output shape: {y.shape}")


def risk_info(p: float):
    if p >= 0.75:
        return "High", "🔥🔥🔥", "#cc2222"
    if p >= 0.5:
        return "Moderate", "🔥🔥", "#dd8800"
    if p >= 0.25:
        return "Low", "🔥", "#eecc00"
    return "Minimal", "—", "#33aa33"


def h3_polygon_coords(cell: str):
    boundary = h3.cell_to_boundary(cell)
    return [(lat, lon) for lat, lon in boundary]


# ============================================================
# PAGE / SESSION STATE
# ============================================================
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption(
    "Click a location once, then run the computer-vision model, the meteorological model, or both."
)

st.markdown(
    """
<style>
[data-testid="stAppViewContainer"] {
    background: #2f2f2f;
    color: #f2f2f2;
}
[data-testid="stHeader"] {
    background: #2f2f2f;
}
[data-testid="stSidebar"] {
    background: #3a3a3a;
    border-right: 1px solid #555555;
}
[data-testid="stSidebar"] * {
    color: #f2f2f2 !important;
}
h1, h2, h3, h4, h5, h6, p, div, span, label {
    color: #f2f2f2;
}
.result-card {
    background: #444444;
    border: 1px solid #666666;
    border-radius: 12px;
    padding: 16px 18px;
    margin-bottom: 12px;
}
.stButton > button {
    background: #555555;
    color: #f2f2f2;
    border: 1px solid #777777;
    border-radius: 8px;
}
.stButton > button:hover {
    background: #666666;
    border-color: #888888;
}
[data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
    color: #f2f2f2 !important;
}
code {
    color: #ffffff !important;
}
</style>
""",
    unsafe_allow_html=True,
)

state_defaults = {
    "selected_center": DEFAULT_CENTER,
    "h3_cell": None,
    "cell_lat": None,
    "cell_lon": None,
    "cv_df": None,
    "cv_imgs": [],
    "lstm_prob": None,
    "era5_df": None,
}
for key, value in state_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

sel_lat, sel_lon = st.session_state["selected_center"]
latest_end = latest_era5_reference_date().date()


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("Selected location")
    st.write(f"Exact click: `{sel_lat:.6f}, {sel_lon:.6f}`")

    if st.session_state["h3_cell"]:
        st.write(f"H3 r{H3_RES} cell: `{st.session_state['h3_cell']}`")
        st.write(
            f"H3 centre: `{st.session_state['cell_lat']:.6f}, {st.session_state['cell_lon']:.6f}`"
        )
    else:
        st.info("Click the map to select a location.")

    st.divider()
    st.header("Meteorological settings")
    end_date = st.date_input(
        "Sequence end date",
        value=latest_end,
        max_value=latest_end,
        help="ERA5-Land monthly means are delayed, so the newest selectable date is clamped.",
    )
    st.caption(f"This fetches the last {SEQ_LEN} monthly time steps.")

    st.divider()
    run_both = st.button("Run both models", type="primary", use_container_width=True)
    run_cv = st.button("Run computer vision only", use_container_width=True)
    run_lstm_btn = st.button("Run meteorological only", use_container_width=True)
    clear = st.button("Clear results", use_container_width=True)

if clear:
    for k in ["cv_df", "cv_imgs", "lstm_prob", "era5_df"]:
        st.session_state[k] = state_defaults[k]
    rerun_app()


# ============================================================
# SINGLE MAP
# ============================================================
st.subheader("1) Pick a location")

main_map = folium.Map(
    location=[sel_lat, sel_lon],
    zoom_start=DEFAULT_ZOOM_PICK,
    tiles=ESRI_TILE_URL,
    attr=ESRI_ATTR,
    control_scale=True,
)

Marker(location=[sel_lat, sel_lon], popup="Selected point").add_to(main_map)

if st.session_state["h3_cell"]:
    poly_color = "#4a8a4a"
    if st.session_state["lstm_prob"] is not None:
        _, _, poly_color = risk_info(float(st.session_state["lstm_prob"]))

    folium.Polygon(
        locations=h3_polygon_coords(st.session_state["h3_cell"]),
        color=poly_color,
        fill=True,
        fill_color=poly_color,
        fill_opacity=0.28,
        weight=2,
        popup=(
            f"H3: {st.session_state['h3_cell']}"
            + (
                f"<br>p_lstm={float(st.session_state['lstm_prob']):.3f}"
                if st.session_state["lstm_prob"] is not None
                else ""
            )
        ),
    ).add_to(main_map)

if st.session_state["cv_df"] is not None:
    for _, row in st.session_state["cv_df"].iterrows():
        la = float(row["lat"])
        lo = float(row["lon"])
        p = row.get("p_wildfire", None)

        if p is None or (isinstance(p, float) and np.isnan(p)):
            color = "gray"
            popup = f"{row['point']}: error"
        else:
            p = float(p)
            color = "red" if p >= LIKELY_THRESHOLD else "blue"
            popup = f"{row['point']}: p={p:.3f}"

        CircleMarker(
            location=(la, lo),
            radius=8 if row["point"] != "center" else 10,
            color=color,
            fill=True,
            fill_opacity=0.8,
            popup=popup,
        ).add_to(main_map)

add_map_legend(main_map)
add_map_resize_fix(main_map)

picked = st_folium(main_map, width=950, height=540, key="main_map")

if picked and picked.get("last_clicked"):
    new_lat = round(float(picked["last_clicked"]["lat"]), 6)
    new_lon = round(float(picked["last_clicked"]["lng"]), 6)
    new_center = (new_lat, new_lon)

    new_cell = h3.latlng_to_cell(new_lat, new_lon, H3_RES)
    cell_center = h3.cell_to_latlng(new_cell)

    current_sig = (
        st.session_state["selected_center"],
        st.session_state["h3_cell"],
    )
    new_sig = (new_center, new_cell)

    if new_sig != current_sig:
        st.session_state["selected_center"] = new_center
        st.session_state["h3_cell"] = new_cell
        st.session_state["cell_lat"] = round(cell_center[0], 6)
        st.session_state["cell_lon"] = round(cell_center[1], 6)
        st.session_state["cv_df"] = None
        st.session_state["cv_imgs"] = []
        st.session_state["lstm_prob"] = None
        st.session_state["era5_df"] = None
        rerun_app()

if st.session_state["h3_cell"] is None:
    st.info("Click on the map to choose a location before running a prediction.")


# ============================================================
# RUN MODELS
# ============================================================
want_cv = run_cv or run_both
want_lstm = run_lstm_btn or run_both

if want_cv:
    try:
        token = get_mapbox_token()
        with st.spinner("Loading computer-vision model…"):
            cv_model = load_cv_model_cached()

        pts = cross5_from_center(*st.session_state["selected_center"])
        rows = []
        imgs = []
        with st.spinner("Downloading Mapbox tiles and running CV predictions…"):
            for name, la, lo in pts:
                try:
                    img = fetch_tile(lo, la, token)
                    p = predict_wildfire_prob_cv(cv_model, img)
                    rows.append({"point": name, "lat": la, "lon": lo, "p_wildfire": p})
                    imgs.append((name, la, lo, p, img))
                except Exception as e:
                    rows.append(
                        {
                            "point": name,
                            "lat": la,
                            "lon": lo,
                            "p_wildfire": None,
                            "error": str(e),
                        }
                    )
        st.session_state["cv_df"] = pd.DataFrame(rows)
        st.session_state["cv_imgs"] = imgs
        rerun_app()
    except Exception as e:
        st.error(f"Computer-vision pipeline failed: {e}")

if want_lstm:
    try:
        with st.spinner("Loading meteorological model…"):
            lstm_model = load_lstm_model_cached()
            scaler = load_scaler_cached()

        cell_lat = st.session_state["cell_lat"]
        cell_lon = st.session_state["cell_lon"]
        end_str = end_date.strftime("%Y-%m-%d")

        with st.spinner("Fetching ERA5-Land monthly means…"):
            era5_df = fetch_era5_sequence(cell_lat, cell_lon, end_str)
            st.session_state["era5_df"] = era5_df

        with st.spinner("Running LSTM inference…"):
            st.session_state["lstm_prob"] = run_lstm(lstm_model, scaler, era5_df)

        rerun_app()
    except Exception as e:
        st.error(f"Meteorological pipeline failed: {e}")


# ============================================================
# RESULTS
# ============================================================
st.subheader("2) Results")

if st.session_state["cv_df"] is None and st.session_state["lstm_prob"] is None:
    st.info("No prediction has been run yet.")
else:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown("**Computer vision result**")
        if st.session_state["cv_df"] is not None:
            cv_df = st.session_state["cv_df"].copy()
            likely_count, stars, emoji, msg = compute_fire_rating(cv_df)
            center_row = cv_df.loc[cv_df["point"] == "center", "p_wildfire"]
            center_p = center_row.iloc[0] if not center_row.empty else np.nan
            st.markdown(f"### {emoji}")
            st.write(
                f"Likely fire tiles: **{likely_count} / 5** (threshold ≥ {LIKELY_THRESHOLD:.1f})"
            )
            if pd.notna(center_p):
                st.metric("Center tile probability", f"{float(center_p):.3f}")
            st.write(msg)
        else:
            st.caption("Not run yet.")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown("**Meteorological result**")
        if st.session_state["lstm_prob"] is not None:
            p_lstm = float(st.session_state["lstm_prob"])
            label, emoji, _ = risk_info(p_lstm)
            st.markdown(f"### {emoji} {label}")
            st.metric("Fire probability", f"{p_lstm:.3f}")
            st.write(
                "Above threshold" if p_lstm >= LSTM_THRESHOLD else "Below threshold"
            )
            st.write(f"H3 cell: `{st.session_state['h3_cell']}`")
        else:
            st.caption("Not run yet.")
        st.markdown("</div>", unsafe_allow_html=True)

    tab_cv, tab_lstm = st.tabs(["Computer vision details", "Meteorological details"])

    with tab_cv:
        if st.session_state["cv_df"] is None:
            st.info("Run the computer-vision model to see details here.")
        else:
            df = st.session_state["cv_df"].copy()
            imgs = st.session_state.get("cv_imgs", [])

            st.dataframe(df, use_container_width=True)
            st.download_button(
                "Download CV CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="wildfire_predictions_cv_5points.csv",
                mime="text/csv",
            )

            st.markdown("**Satellite images**")
            if imgs:
                cols = st.columns(5)
                for i, (name, la, lo, p, img) in enumerate(imgs):
                    with cols[i]:
                        caption = f"{name}\n{lo:.6f}, {la:.6f}\n"
                        caption += f"p={p:.3f}" if p is not None else "p=None"
                        st.image(img, caption=caption, use_container_width=True)
            else:
                st.warning(
                    "No images were produced. Check the errors in the table above."
                )

    with tab_lstm:
        if st.session_state["lstm_prob"] is None:
            st.info("Run the meteorological model to see details here.")
        else:
            p_val = float(st.session_state["lstm_prob"])
            label, emoji, color = risk_info(p_val)
            st.markdown(
                f"<div class='result-card'><span style='font-size:2rem;color:{color};font-weight:800;'>"
                f"{emoji} {label} Risk</span><br><br>"
                f"H3 cell: <code>{st.session_state['h3_cell']}</code><br>"
                f"H3 centre: {st.session_state['cell_lat']:.6f}, {st.session_state['cell_lon']:.6f}</div>",
                unsafe_allow_html=True,
            )

            if st.session_state["era5_df"] is not None:
                df_show = st.session_state["era5_df"].copy()
                st.dataframe(df_show, use_container_width=True)
                st.download_button(
                    "Download ERA5 CSV",
                    data=df_show.to_csv(index=False).encode("utf-8"),
                    file_name=f"era5_{st.session_state['h3_cell']}_{end_date}.csv",
                    mime="text/csv",
                )

                st.markdown("**Feature trends**")
                cols = st.columns(3)
                chart_features = [
                    ("2m_temperature", "Temperature (K)"),
                    ("total_precipitation", "Precipitation (m)"),
                    ("wind_total", "Wind speed (m/s)"),
                    ("volumetric_soil_water_layer_1", "Soil water (m³/m³)"),
                    ("surface_solar_radiation_downwards", "Solar radiation (J/m²)"),
                    ("leaf_area_index_high_vegetation", "LAI high vegetation"),
                ]
                for i, (feat, title) in enumerate(chart_features):
                    with cols[i % 3]:
                        st.caption(title)
                        if feat in df_show.columns:
                            st.line_chart(
                                df_show.set_index("date")[feat],
                                height=140,
                                use_container_width=True,
                            )

st.caption("Streamlit app combining Mapbox + VGG16 and ERA5-Land + LSTM")
