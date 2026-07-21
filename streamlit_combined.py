import base64
import html
import io
import math
import os
import shutil
import tempfile
import time
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
from PIL import Image, ImageDraw
from folium import CircleMarker, Marker
from folium.plugins import FloatImage
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
LIKELY_THRESHOLD = 0.9
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


def end_of_month(d: datetime) -> datetime:
    """Return last day (UTC) of the month containing datetime d."""
    d = d.replace(hour=0, minute=0, second=0, microsecond=0)
    next_month = (d.replace(day=28) + timedelta(days=4)).replace(day=1)
    return next_month - timedelta(days=1)


def heuristic_latest_era5_date() -> datetime:
    """
    Safe fallback: assume last complete month is available.
    This avoids contacting CDS during page startup.
    """
    now = datetime.utcnow().replace(
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
    )
    first_this_month = now.replace(day=1)
    return first_this_month - timedelta(days=1)


def h3_polygon_coords(cell: str):
    boundary = h3.cell_to_boundary(cell)
    return [(lat, lon) for lat, lon in boundary]


def point_in_polygon(lat: float, lon: float, polygon):
    """
    Ray-casting point-in-polygon test.

    polygon is a list of latitude/longitude pairs.
    """
    x = lon
    y = lat
    inside = False
    n = len(polygon)

    for i in range(n):
        y1, x1 = polygon[i]
        y2, x2 = polygon[(i + 1) % n]

        intersects = ((y1 > y) != (y2 > y)) and (
            x
            < (x2 - x1)
            * (y - y1)
            / ((y2 - y1) + 1e-12)
            + x1
        )

        if intersects:
            inside = not inside

    return inside


@st.cache_data(show_spinner=False)
def cv_points_for_h3_cell(
    cell: str,
    spacing_km: float = SPACING_KM,
):
    """
    Generate extraction points spaced approximately three kilometres apart.

    The point grid is anchored on the H3-cell centre and clipped to the
    selected H3 polygon.
    """
    center_lat, center_lon = h3.cell_to_latlng(cell)
    center_lat = float(center_lat)
    center_lon = float(center_lon)

    polygon = h3_polygon_coords(cell)
    poly_lats = [point[0] for point in polygon]
    poly_lons = [point[1] for point in polygon]

    dlat = spacing_km / 110.574
    dlon = spacing_km / (
        111.320 * math.cos(math.radians(center_lat))
    )

    lat_span = max(
        abs(max(poly_lats) - center_lat),
        abs(center_lat - min(poly_lats)),
    )

    lon_span = max(
        abs(max(poly_lons) - center_lon),
        abs(center_lon - min(poly_lons)),
    )

    n_lat = max(
        1,
        int(math.ceil(lat_span / dlat)) + 1,
    )

    n_lon = max(
        1,
        int(math.ceil(lon_span / dlon)) + 1,
    )

    raw_points = []

    for i in range(-n_lat, n_lat + 1):
        for j in range(-n_lon, n_lon + 1):
            point_lat = center_lat + i * dlat
            point_lon = center_lon + j * dlon

            if point_in_polygon(
                point_lat,
                point_lon,
                polygon,
            ):
                raw_points.append(
                    (
                        i,
                        j,
                        point_lat,
                        point_lon,
                    )
                )

    # Ensure that the H3 centre is always included.
    if not any(
        i == 0 and j == 0
        for i, j, _, _ in raw_points
    ):
        raw_points.append(
            (
                0,
                0,
                center_lat,
                center_lon,
            )
        )

    raw_points.sort(
        key=lambda item: (
            item[0] ** 2 + item[1] ** 2,
            abs(item[0]),
            abs(item[1]),
            item[0],
            item[1],
        )
    )

    points = []
    counter = 1

    for i, j, point_lat, point_lon in raw_points:
        if i == 0 and j == 0:
            name = "center"
        else:
            name = f"p{counter:02d}"
            counter += 1

        points.append(
            (
                name,
                round(point_lat, 6),
                round(point_lon, 6),
            )
        )

    return points


@st.cache_data(show_spinner=False)
def build_legend_data_uri():
    width = 270
    height = 190

    image = Image.new(
        "RGBA",
        (width, height),
        (255, 255, 255, 235),
    )

    draw = ImageDraw.Draw(image)

    draw.rectangle(
        (
            0,
            0,
            width - 1,
            height - 1,
        ),
        outline=(190, 190, 190, 255),
        width=1,
    )

    x_start = 12
    y = 10

    draw.text(
        (x_start, y),
        "Legend",
        fill=(20, 20, 20, 255),
    )

    y += 24

    items = [
        ("#cc2222", "■", "LSTM high risk"),
        ("#dd8800", "■", "LSTM moderate risk"),
        ("#eecc00", "■", "LSTM low risk"),
        ("#33aa33", "■", "LSTM minimal risk"),
        (None, None, None),
        ("#ff0000", "o", "CV tile > 0.90"),
        ("#0000ff", "o", "CV tile < 0.90"),
        ("#808080", "o", "CV tile error"),
    ]

    for color, symbol, label in items:
        if symbol is None:
            draw.line(
                (
                    x_start,
                    y + 4,
                    width - 12,
                    y + 4,
                ),
                fill=(210, 210, 210, 255),
                width=1,
            )

            y += 14
            continue

        draw.text(
            (x_start, y),
            symbol,
            fill=color,
        )

        draw.text(
            (x_start + 18, y),
            label,
            fill=(50, 50, 50, 255),
        )

        y += 20

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("ascii")

    return f"data:image/png;base64,{encoded}"


def add_map_legend(map_object):
    legend_uri = build_legend_data_uri()

    FloatImage(
        legend_uri,
        bottom=3,
        left=74,
    ).add_to(map_object)


def render_result_card(
    title: str,
    body_html: str,
):
    st.markdown(
        f"""
<div class="result-card">
  <div style="font-weight:800; font-size:1.05rem; margin-bottom:8px;">
    {html.escape(title)}
  </div>
  {body_html}
</div>
""",
        unsafe_allow_html=True,
    )


# ============================================================
# CDS / ERA5 CONNECTION
# ============================================================
def get_cds_client():
    """Create a CDS client with bounded retries and a generous timeout."""
    url = secret_or_env("CDS_URL")
    key = secret_or_env("CDS_KEY")

    if not url or not key:
        raise RuntimeError(
            "CDS credentials are missing. Add CDS_URL and CDS_KEY "
            "to Streamlit secrets."
        )

    return cdsapi.Client(
        url=url,
        key=key,
        quiet=False,
        debug=False,
        timeout=600,
        retry_max=3,
        sleep_max=15,
    )


def _is_transient_cds_error(
    exception: Exception,
) -> bool:
    """Identify connection errors for which retrying is reasonable."""
    message = str(exception).lower()

    transient_markers = (
        "ssl",
        "unexpected_eof",
        "eof occurred",
        "connection reset",
        "connection aborted",
        "remote disconnected",
        "max retries exceeded",
        "temporarily unavailable",
        "bad gateway",
        "service unavailable",
        "gateway timeout",
        "timed out",
        "timeout",
        "http 429",
        "http 500",
        "http 502",
        "http 503",
        "http 504",
    )

    return any(
        marker in message
        for marker in transient_markers
    )


def retrieve_cds_with_retry(
    dataset: str,
    request: dict,
    target_path: str,
    attempts: int = 3,
) -> None:
    """
    Submit and download a CDS request.

    A new client is created for each application-level attempt so that a
    broken HTTP session is not reused.
    """
    last_error = None

    for attempt in range(1, attempts + 1):
        try:
            if os.path.exists(target_path):
                os.unlink(target_path)

            client = get_cds_client()

            client.retrieve(
                dataset,
                request,
                target_path,
            )

            if (
                not os.path.exists(target_path)
                or os.path.getsize(target_path) == 0
            ):
                raise RuntimeError(
                    "CDS returned an empty download."
                )

            return

        except Exception as exception:
            last_error = exception

            should_retry = (
                _is_transient_cds_error(exception)
                and attempt < attempts
            )

            if not should_retry:
                raise RuntimeError(
                    f"CDS request failed on attempt "
                    f"{attempt}/{attempts}: {exception}"
                ) from exception

            delay = min(
                5 * (2 ** (attempt - 1)),
                20,
            )

            time.sleep(delay)

    raise RuntimeError(
        f"CDS request failed after {attempts} attempts: "
        f"{last_error}"
    )


# ============================================================
# CDS NETCDF PARSING
# ============================================================
def _collect_netcdf_paths(
    download_path: str,
    extract_dir: str,
) -> list[Path]:
    """
    Return all NetCDF files contained in a CDS download.

    The service may return either one NetCDF file or a ZIP archive containing
    several NetCDF files.
    """
    source = Path(download_path)

    if zipfile.is_zipfile(source):
        extract_path = Path(extract_dir)

        extract_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        with zipfile.ZipFile(
            source,
            "r",
        ) as archive:
            archive.extractall(extract_path)

        candidates = sorted(
            extract_path.rglob("*.nc")
        )

        candidates.extend(
            sorted(
                extract_path.rglob("*.netcdf")
            )
        )

    else:
        candidates = [source]

    candidates = [
        path
        for path in candidates
        if path.is_file()
    ]

    if not candidates:
        raise RuntimeError(
            "The CDS download contained no readable NetCDF file."
        )

    return candidates


def load_cds_datasets(
    download_path: str,
    extract_dir: str,
) -> list[xr.Dataset]:
    """
    Load every NetCDF returned by CDS into memory.

    Datasets remain separate because different files can contain different
    variable groups or coordinate structures.
    """
    datasets: list[xr.Dataset] = []

    candidates = _collect_netcdf_paths(
        download_path,
        extract_dir,
    )

    for candidate in candidates:
        loaded_dataset = None
        last_error = None

        # Prefer netCDF4, but fall back to xarray's automatic engine.
        for engine in ("netcdf4", None):
            opened = None

            try:
                if engine is None:
                    opened = xr.open_dataset(candidate)
                else:
                    opened = xr.open_dataset(
                        candidate,
                        engine=engine,
                    )

                loaded_dataset = opened.load()
                break

            except Exception as exception:
                last_error = exception

            finally:
                if opened is not None:
                    opened.close()

        if loaded_dataset is None:
            raise RuntimeError(
                f"Could not open CDS NetCDF file "
                f"'{candidate.name}': {last_error}"
            ) from last_error

        datasets.append(loaded_dataset)

    if not datasets:
        raise RuntimeError(
            "CDS returned NetCDF files, but none could be opened."
        )

    return datasets


def _first_existing_name(
    names,
    candidates: tuple[str, ...],
) -> str | None:
    """Find a coordinate or variable name, ignoring letter case if needed."""
    names = [
        str(name)
        for name in names
    ]

    for candidate in candidates:
        if candidate in names:
            return candidate

    lower_lookup = {
        name.lower(): name
        for name in names
    }

    for candidate in candidates:
        match = lower_lookup.get(
            candidate.lower()
        )

        if match is not None:
            return match

    return None


def _month_key(value) -> str | None:
    """Convert a time-coordinate value to YYYY-MM."""
    text = str(value).strip()
    compact = text.split(".")[0]

    # Handle compact values such as 20250701 before pandas interprets an
    # integer as nanoseconds since 1970.
    if (
        compact.isdigit()
        and len(compact) in (
            6,
            8,
            10,
            12,
            14,
        )
    ):
        year = compact[:4]
        month = compact[4:6]

        if 1 <= int(month) <= 12:
            return f"{year}-{month}"

    try:
        timestamp = pd.to_datetime(value)

        if not pd.isna(timestamp):
            return timestamp.strftime(
                "%Y-%m"
            )

    except Exception:
        pass

    # Also handle ISO-like and cftime string representations.
    import re

    match = re.search(
        r"(?<!\d)(\d{4})[-/]?(\d{2})(?!\d)",
        text,
    )

    if match:
        year, month = match.groups()

        if 1 <= int(month) <= 12:
            return f"{year}-{month}"

    return None


def _normalise_longitude_for_dataset(
    longitude: float,
    values,
) -> float:
    """
    Match a longitude to the coordinate convention used by a dataset.

    Some datasets represent longitude as -180..180 and others as 0..360.
    """
    try:
        longitude_values = np.asarray(
            values,
            dtype=float,
        )

        longitude_values = longitude_values[
            np.isfinite(longitude_values)
        ]

        if longitude_values.size == 0:
            return float(longitude)

        minimum = float(
            longitude_values.min()
        )

        maximum = float(
            longitude_values.max()
        )

        if (
            minimum >= 0.0
            and maximum > 180.0
            and longitude < 0.0
        ):
            return float(
                longitude + 360.0
            )

        if (
            minimum < 0.0
            and longitude > 180.0
        ):
            return float(
                longitude - 360.0
            )

    except Exception:
        pass

    return float(longitude)


def _select_nearest_point(
    data_array: xr.DataArray,
    dataset: xr.Dataset,
    latitude: float,
    longitude: float,
) -> xr.DataArray:
    """Select the nearest available latitude and longitude grid point."""
    latitude_name = _first_existing_name(
        list(data_array.coords)
        + list(dataset.coords),
        (
            "latitude",
            "lat",
        ),
    )

    longitude_name = _first_existing_name(
        list(data_array.coords)
        + list(dataset.coords),
        (
            "longitude",
            "lon",
        ),
    )

    result = data_array

    if (
        latitude_name is not None
        and latitude_name in result.coords
    ):
        result = result.sel(
            {
                latitude_name: float(
                    latitude
                )
            },
            method="nearest",
        )

    if (
        longitude_name is not None
        and longitude_name in result.coords
    ):
        target_longitude = (
            _normalise_longitude_for_dataset(
                float(longitude),
                result[
                    longitude_name
                ].values,
            )
        )

        result = result.sel(
            {
                longitude_name: (
                    target_longitude
                )
            },
            method="nearest",
        )

    return result


def _scalar_from_data_array(
    data_array: xr.DataArray,
) -> float:
    """
    Return one finite scalar from a selected field.

    Extra dimensions such as expver or number are tolerated. If several
    finite values remain, their mean is used.
    """
    try:
        values = np.asarray(
            data_array.values,
            dtype=float,
        ).reshape(-1)

    except Exception:
        return np.nan

    finite_values = values[
        np.isfinite(values)
    ]

    if finite_values.size == 0:
        return np.nan

    return float(
        finite_values.mean()
    )


def _extract_monthly_series(
    datasets: list[xr.Dataset],
    variable_aliases: tuple[str, ...],
    latitude: float,
    longitude: float,
) -> dict[str, float]:
    """
    Extract a dictionary mapping YYYY-MM to values.

    This supports both short ERA5 variable names such as t2m and descriptive
    names such as 2m_temperature.
    """
    values_by_month: dict[
        str,
        float,
    ] = {}

    for dataset in datasets:
        variable_name = (
            _first_existing_name(
                dataset.data_vars,
                variable_aliases,
            )
        )

        if variable_name is None:
            continue

        data_array = (
            _select_nearest_point(
                dataset[variable_name],
                dataset,
                latitude,
                longitude,
            )
        )

        time_name = (
            _first_existing_name(
                list(data_array.coords)
                + list(data_array.dims),
                (
                    "valid_time",
                    "time",
                    "date",
                    "forecast_reference_time",
                ),
            )
        )

        if time_name is None:
            # A one-month file can expose its time only as a scalar dataset
            # coordinate rather than as a dimension of the variable.
            scalar_month = None

            for candidate in (
                "valid_time",
                "time",
                "date",
                "forecast_reference_time",
            ):
                if candidate not in dataset.coords:
                    continue

                coordinate_values = np.asarray(
                    dataset[
                        candidate
                    ].values
                ).reshape(-1)

                if coordinate_values.size:
                    scalar_month = _month_key(
                        coordinate_values[0]
                    )
                    break

            if scalar_month is not None:
                value = (
                    _scalar_from_data_array(
                        data_array
                    )
                )

                if np.isfinite(value):
                    values_by_month.setdefault(
                        scalar_month,
                        value,
                    )

            continue

        if time_name in data_array.coords:
            time_coordinate = (
                data_array[time_name]
            )

        elif time_name in dataset.coords:
            time_coordinate = (
                dataset[time_name]
            )

        else:
            continue

        time_values = np.asarray(
            time_coordinate.values
        ).reshape(-1)

        if time_name in data_array.dims:
            time_dimension = time_name

        elif time_coordinate.dims:
            matching_dimensions = [
                dimension
                for dimension
                in time_coordinate.dims
                if dimension
                in data_array.dims
            ]

            time_dimension = (
                matching_dimensions[0]
                if matching_dimensions
                else None
            )

        else:
            time_dimension = None

        if time_dimension is None:
            if time_values.size == 1:
                month = _month_key(
                    time_values[0]
                )

                value = (
                    _scalar_from_data_array(
                        data_array
                    )
                )

                if (
                    month is not None
                    and np.isfinite(value)
                ):
                    values_by_month.setdefault(
                        month,
                        value,
                    )

            continue

        count = min(
            data_array.sizes[
                time_dimension
            ],
            time_values.size,
        )

        for index in range(count):
            month = _month_key(
                time_values[index]
            )

            if month is None:
                continue

            monthly_slice = (
                data_array.isel(
                    {
                        time_dimension: index
                    }
                )
            )

            value = (
                _scalar_from_data_array(
                    monthly_slice
                )
            )

            if np.isfinite(value):
                values_by_month.setdefault(
                    month,
                    value,
                )

    return values_by_month


def _find_nearest_valid_era5_point(
    datasets: list[xr.Dataset],
    target_latitude: float,
    target_longitude: float,
    max_distance_km: float = 50.0,
) -> tuple[float, float, float]:
    """
    Find the nearest ERA5-Land cell containing valid temperature data.

    ERA5-Land masks ocean cells. A selected H3 centre can therefore be located
    on a water cell even when nearby land is visible. The same valid ERA5 cell
    is subsequently used for every meteorological feature.
    """
    temperature_aliases = (
        "t2m",
        "2m_temperature",
        "temperature_2m",
        "2t",
    )

    best_candidate = None

    for dataset in datasets:
        variable_name = (
            _first_existing_name(
                dataset.data_vars,
                temperature_aliases,
            )
        )

        if variable_name is None:
            continue

        data_array = dataset[
            variable_name
        ]

        latitude_name = (
            _first_existing_name(
                list(data_array.coords)
                + list(dataset.coords),
                (
                    "latitude",
                    "lat",
                ),
            )
        )

        longitude_name = (
            _first_existing_name(
                list(data_array.coords)
                + list(dataset.coords),
                (
                    "longitude",
                    "lon",
                ),
            )
        )

        if (
            latitude_name is None
            or longitude_name is None
        ):
            continue

        if (
            latitude_name
            not in data_array.dims
            or longitude_name
            not in data_array.dims
        ):
            continue

        non_spatial_dimensions = [
            dimension
            for dimension
            in data_array.dims
            if dimension
            not in (
                latitude_name,
                longitude_name,
            )
        ]

        valid_mask = np.isfinite(
            data_array
        )

        if non_spatial_dimensions:
            valid_mask = valid_mask.any(
                dim=non_spatial_dimensions
            )

        valid_mask = valid_mask.transpose(
            latitude_name,
            longitude_name,
        )

        latitude_values = np.asarray(
            valid_mask[
                latitude_name
            ].values,
            dtype=float,
        )

        longitude_values = np.asarray(
            valid_mask[
                longitude_name
            ].values,
            dtype=float,
        )

        mask_values = np.asarray(
            valid_mask.values,
            dtype=bool,
        )

        if (
            latitude_values.size == 0
            or longitude_values.size == 0
            or not mask_values.any()
        ):
            continue

        normalised_target_longitude = (
            _normalise_longitude_for_dataset(
                target_longitude,
                longitude_values,
            )
        )

        (
            latitude_grid,
            longitude_grid,
        ) = np.meshgrid(
            latitude_values,
            longitude_values,
            indexing="ij",
        )

        longitude_difference = (
            (
                longitude_grid
                - normalised_target_longitude
                + 180.0
            )
            % 360.0
        ) - 180.0

        latitude_difference = (
            latitude_grid
            - float(target_latitude)
        )

        north_south_km = (
            latitude_difference
            * 111.32
        )

        east_west_km = (
            longitude_difference
            * 111.32
            * math.cos(
                math.radians(
                    float(target_latitude)
                )
            )
        )

        distance_km = np.sqrt(
            north_south_km**2
            + east_west_km**2
        )

        distance_km[
            ~mask_values
        ] = np.inf

        flat_index = int(
            np.argmin(distance_km)
        )

        (
            row_index,
            column_index,
        ) = np.unravel_index(
            flat_index,
            distance_km.shape,
        )

        nearest_distance = float(
            distance_km[
                row_index,
                column_index,
            ]
        )

        if not np.isfinite(
            nearest_distance
        ):
            continue

        candidate = (
            float(
                latitude_values[
                    row_index
                ]
            ),
            float(
                longitude_values[
                    column_index
                ]
            ),
            nearest_distance,
        )

        if (
            best_candidate is None
            or candidate[2]
            < best_candidate[2]
        ):
            best_candidate = candidate

    if best_candidate is None:
        raise RuntimeError(
            "No valid ERA5-Land grid point was found in the downloaded "
            "area. The selected location may be over water or outside "
            "the ERA5-Land mask."
        )

    if (
        best_candidate[2]
        > max_distance_km
    ):
        raise RuntimeError(
            "The selected location does not have valid ERA5-Land data, "
            "and the nearest valid land grid point is "
            f"{best_candidate[2]:.1f} km away. Please select a location "
            "closer to land."
        )

    return best_candidate


def _dataset_diagnostics(
    datasets: list[xr.Dataset],
) -> str:
    """Build concise diagnostics for future CDS format changes."""
    details = []

    for index, dataset in enumerate(
        datasets,
        start=1,
    ):
        variables = ", ".join(
            map(
                str,
                dataset.data_vars,
            )
        )

        coordinates = ", ".join(
            map(
                str,
                dataset.coords,
            )
        )

        dimensions = ", ".join(
            f"{name}={size}"
            for name, size
            in dataset.sizes.items()
        )

        details.append(
            f"file {index}: "
            f"variables=[{variables}], "
            f"coordinates=[{coordinates}], "
            f"dimensions=[{dimensions}]"
        )

    return " | ".join(details)


# ============================================================
# COMPUTER VISION HELPERS
# ============================================================
def get_mapbox_token() -> str:
    token = secret_or_env(
        "MAPBOX_ACCESS_TOKEN"
    )

    if not token:
        raise RuntimeError(
            "MAPBOX_ACCESS_TOKEN is missing. Add it to Streamlit "
            "secrets or your environment."
        )

    return token


def build_mapbox_url(
    longitude: float,
    latitude: float,
    token: str,
) -> str:
    longitude = round(
        float(longitude),
        6,
    )

    latitude = round(
        float(latitude),
        6,
    )

    base = (
        f"https://api.mapbox.com/styles/v1/"
        f"{STYLE_USER}/{STYLE_ID}/static/"
    )

    coordinates = (
        f"{longitude},{latitude}"
    )

    remainder = (
        f",{ZOOM},{BEARING}/"
        f"{TILE_SIZE}x{TILE_SIZE}"
        f"?access_token={token}"
        f"&logo=false"
        f"&attribution=false"
    )

    return (
        base
        + coordinates
        + remainder
    )


def preprocess_pil(
    image: Image.Image,
) -> np.ndarray:
    values = (
        np.asarray(
            image,
            dtype=np.float32,
        )
        * RESCALE
    )

    return np.expand_dims(
        values,
        axis=0,
    )


def predict_wildfire_prob_cv(
    model,
    image: Image.Image,
) -> float:
    values = preprocess_pil(
        image
    )

    prediction = np.array(
        model.predict(
            values,
            verbose=0,
        )
    )

    if (
        prediction.ndim == 2
        and prediction.shape[1] == 2
    ):
        return float(
            prediction[
                0,
                CV_WILDFIRE_INDEX,
            ]
        )

    if (
        prediction.ndim == 2
        and prediction.shape[1] == 1
    ):
        return float(
            prediction[0, 0]
        )

    raise ValueError(
        "Unexpected CV model output shape: "
        f"{prediction.shape}"
    )


def compute_fire_rating(
    dataframe: pd.DataFrame,
    threshold: float = LIKELY_THRESHOLD,
):
    probabilities = pd.to_numeric(
        dataframe.get(
            "p_wildfire"
        ),
        errors="coerce",
    )

    likely_count = int(
        (
            probabilities >= threshold
        ).sum()
    )

    if likely_count in range(
        1,
        9,
    ):
        stars = 0
        message = (
            "A fire is unlikely in this environment."
        )

    elif likely_count in range(
        9,
        18,
    ):
        stars = 1
        message = (
            "The fire potential of this environment is low."
        )

    elif likely_count in range(
        18,
        27,
    ):
        stars = 2
        message = (
            "The fire potential of this environment is moderate. "
            "Check local safety precautions."
        )

    else:
        stars = 3
        message = (
            "The fire potential of this environment is high. "
            "Check local safety precautions."
        )

    emoji = (
        "🔥" * stars
        if stars > 0
        else "—"
    )

    return (
        likely_count,
        stars,
        emoji,
        message,
    )


@st.cache_resource(
    show_spinner=False
)
def load_cv_model_cached():
    import tensorflow as tf

    if not CV_MODEL_PATH.exists():
        raise FileNotFoundError(
            "CV model file not found: "
            f"{CV_MODEL_PATH.resolve()}"
        )

    return tf.keras.models.load_model(
        CV_MODEL_PATH
    )


@st.cache_data(
    ttl=3600,
    show_spinner=False,
)
def download_bytes(
    url: str,
) -> bytes:
    response = SESSION.get(
        url,
        timeout=30,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"HTTP {response.status_code}: "
            f"{response.text[:300]}"
        )

    return response.content


def fetch_tile(
    longitude: float,
    latitude: float,
    token: str,
) -> Image.Image:
    url = build_mapbox_url(
        longitude,
        latitude,
        token,
    )

    content = download_bytes(
        url
    )

    image = Image.open(
        io.BytesIO(content)
    ).convert("RGB")

    if image.size != (
        TILE_SIZE,
        TILE_SIZE,
    ):
        image = image.resize(
            (
                TILE_SIZE,
                TILE_SIZE,
            )
        )

    return image


# ============================================================
# LSTM / ERA5 HELPERS
# ============================================================
@st.cache_resource(
    show_spinner=False
)
def load_lstm_model_cached():
    import tensorflow as tf

    if not LSTM_MODEL_PATH.exists():
        raise FileNotFoundError(
            "LSTM model file not found: "
            f"{LSTM_MODEL_PATH.resolve()}"
        )

    return tf.keras.models.load_model(
        LSTM_MODEL_PATH
    )


@st.cache_resource(
    show_spinner=False
)
def load_scaler_cached():
    if not SCALER_PATH.exists():
        return None

    import joblib

    return joblib.load(
        SCALER_PATH
    )


@st.cache_data(
    ttl=3600,
    show_spinner=False,
)
def fetch_era5_sequence(
    latitude: float,
    longitude: float,
    end_date_string: str,
) -> pd.DataFrame:
    """
    Download exactly 12 monthly ERA5-Land observations for one location.

    Requests are split by calendar year. Returned NetCDF files are parsed
    separately. If the clicked grid cell is masked, the nearest valid
    ERA5-Land cell within 50 km is used consistently for all model features.
    """
    end_date = datetime.strptime(
        end_date_string,
        "%Y-%m-%d",
    )

    latest_date = (
        heuristic_latest_era5_date()
    )

    if end_date > latest_date:
        end_date = latest_date

    requested_months: list[
        datetime
    ] = []

    current_month = end_date.replace(
        day=1
    )

    for _ in range(SEQ_LEN):
        requested_months.append(
            current_month
        )

        if current_month.month == 1:
            current_month = (
                current_month.replace(
                    year=(
                        current_month.year
                        - 1
                    ),
                    month=12,
                )
            )

        else:
            current_month = (
                current_month.replace(
                    month=(
                        current_month.month
                        - 1
                    )
                )
            )

    requested_months = sorted(
        requested_months
    )

    months_by_year: dict[
        str,
        list[str],
    ] = {}

    dates_by_year: dict[
        str,
        list[datetime],
    ] = {}

    for month_date in requested_months:
        year = str(
            month_date.year
        )

        month = (
            f"{month_date.month:02d}"
        )

        months_by_year.setdefault(
            year,
            [],
        ).append(month)

        dates_by_year.setdefault(
            year,
            [],
        ).append(month_date)

    area = [
        round(
            latitude + 0.5,
            2,
        ),
        round(
            longitude - 0.5,
            2,
        ),
        round(
            latitude - 0.5,
            2,
        ),
        round(
            longitude + 0.5,
            2,
        ),
    ]

    variable_aliases = {
        "2m_temperature": (
            "t2m",
            "2m_temperature",
            "temperature_2m",
            "2t",
        ),
        "volumetric_soil_water_layer_1": (
            "swvl1",
            "volumetric_soil_water_layer_1",
        ),
        "surface_solar_radiation_downwards": (
            "ssrd",
            "surface_solar_radiation_downwards",
        ),
        "total_evaporation": (
            "e",
            "total_evaporation",
        ),
        "10m_u_component_of_wind": (
            "u10",
            "10m_u_component_of_wind",
            "u_component_of_wind_10m",
            "10u",
        ),
        "10m_v_component_of_wind": (
            "v10",
            "10m_v_component_of_wind",
            "v_component_of_wind_10m",
            "10v",
        ),
        "total_precipitation": (
            "tp",
            "total_precipitation",
        ),
        "leaf_area_index_high_vegetation": (
            "lai_hv",
            "leaf_area_index_high_vegetation",
            "high_vegetation_leaf_area_index",
        ),
    }

    records_by_month = {
        month_date.strftime(
            "%Y-%m"
        ): {
            "date": month_date.strftime(
                "%Y-%m"
            ),
            **{
                feature: np.nan
                for feature
                in FEATURE_NAMES
            },
        }
        for month_date
        in requested_months
    }

    diagnostics: list[str] = []

    selected_grid_points: list[
        tuple[
            float,
            float,
            float,
        ]
    ] = []

    temporary_root = tempfile.mkdtemp(
        prefix="era5_land_"
    )

    try:
        for year in sorted(
            months_by_year
        ):
            target_path = os.path.join(
                temporary_root,
                f"era5_{year}.download",
            )

            extract_directory = (
                os.path.join(
                    temporary_root,
                    f"era5_{year}_extracted",
                )
            )

            request = {
                "product_type": [
                    "monthly_averaged_reanalysis"
                ],
                "variable": ERA5_VARIABLES,
                "year": [year],
                "month": sorted(
                    months_by_year[year]
                ),
                "time": ["00:00"],
                "area": area,
                "data_format": "netcdf",
                "download_format": (
                    "unarchived"
                ),
            }

            retrieve_cds_with_retry(
                dataset=(
                    "reanalysis-era5-land-"
                    "monthly-means"
                ),
                request=request,
                target_path=target_path,
                attempts=3,
            )

            datasets = load_cds_datasets(
                target_path,
                extract_directory,
            )

            diagnostics.append(
                f"{year}: "
                f"{_dataset_diagnostics(datasets)}"
            )

            (
                era5_latitude,
                era5_longitude,
                era5_distance_km,
            ) = (
                _find_nearest_valid_era5_point(
                    datasets=datasets,
                    target_latitude=latitude,
                    target_longitude=longitude,
                    max_distance_km=50.0,
                )
            )

            selected_grid_points.append(
                (
                    era5_latitude,
                    era5_longitude,
                    era5_distance_km,
                )
            )

            diagnostics.append(
                f"{year}: requested point="
                f"({latitude:.6f}, "
                f"{longitude:.6f}), "
                f"ERA5 point="
                f"({era5_latitude:.6f}, "
                f"{era5_longitude:.6f}), "
                f"distance="
                f"{era5_distance_km:.1f} km"
            )

            temperature = (
                _extract_monthly_series(
                    datasets,
                    variable_aliases[
                        "2m_temperature"
                    ],
                    era5_latitude,
                    era5_longitude,
                )
            )

            soil_water = (
                _extract_monthly_series(
                    datasets,
                    variable_aliases[
                        "volumetric_soil_water_layer_1"
                    ],
                    era5_latitude,
                    era5_longitude,
                )
            )

            solar_radiation = (
                _extract_monthly_series(
                    datasets,
                    variable_aliases[
                        "surface_solar_radiation_downwards"
                    ],
                    era5_latitude,
                    era5_longitude,
                )
            )

            evaporation = (
                _extract_monthly_series(
                    datasets,
                    variable_aliases[
                        "total_evaporation"
                    ],
                    era5_latitude,
                    era5_longitude,
                )
            )

            u_wind = (
                _extract_monthly_series(
                    datasets,
                    variable_aliases[
                        "10m_u_component_of_wind"
                    ],
                    era5_latitude,
                    era5_longitude,
                )
            )

            v_wind = (
                _extract_monthly_series(
                    datasets,
                    variable_aliases[
                        "10m_v_component_of_wind"
                    ],
                    era5_latitude,
                    era5_longitude,
                )
            )

            precipitation = (
                _extract_monthly_series(
                    datasets,
                    variable_aliases[
                        "total_precipitation"
                    ],
                    era5_latitude,
                    era5_longitude,
                )
            )

            leaf_area = (
                _extract_monthly_series(
                    datasets,
                    variable_aliases[
                        "leaf_area_index_high_vegetation"
                    ],
                    era5_latitude,
                    era5_longitude,
                )
            )

            for month_date in (
                dates_by_year[year]
            ):
                month = (
                    month_date.strftime(
                        "%Y-%m"
                    )
                )

                record = (
                    records_by_month[
                        month
                    ]
                )

                record[
                    "2m_temperature"
                ] = temperature.get(
                    month,
                    np.nan,
                )

                record[
                    "volumetric_soil_water_layer_1"
                ] = soil_water.get(
                    month,
                    np.nan,
                )

                record[
                    "surface_solar_radiation_downwards"
                ] = solar_radiation.get(
                    month,
                    np.nan,
                )

                record[
                    "total_evaporation"
                ] = evaporation.get(
                    month,
                    np.nan,
                )

                record[
                    "total_precipitation"
                ] = precipitation.get(
                    month,
                    np.nan,
                )

                record[
                    "leaf_area_index_high_vegetation"
                ] = leaf_area.get(
                    month,
                    np.nan,
                )

                u_value = u_wind.get(
                    month,
                    np.nan,
                )

                v_value = v_wind.get(
                    month,
                    np.nan,
                )

                if (
                    np.isfinite(u_value)
                    and np.isfinite(v_value)
                ):
                    record[
                        "wind_total"
                    ] = math.hypot(
                        u_value,
                        v_value,
                    )

    finally:
        shutil.rmtree(
            temporary_root,
            ignore_errors=True,
        )

    result = (
        pd.DataFrame(
            records_by_month.values()
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    if len(result) != SEQ_LEN:
        raise RuntimeError(
            f"Expected {SEQ_LEN} monthly rows "
            f"from CDS, but received "
            f"{len(result)}."
        )

    missing_columns = [
        name
        for name in FEATURE_NAMES
        if name not in result.columns
    ]

    if missing_columns:
        raise RuntimeError(
            "CDS response is missing required "
            "variables: "
            + ", ".join(
                missing_columns
            )
        )

    missing_values = result[
        FEATURE_NAMES
    ].isna()

    if missing_values.any().any():
        affected = []

        for column in FEATURE_NAMES:
            bad_dates = result.loc[
                missing_values[column],
                "date",
            ].tolist()

            if bad_dates:
                affected.append(
                    f"{column}: "
                    + ", ".join(
                        bad_dates
                    )
                )

        raise RuntimeError(
            "CDS returned incomplete "
            "meteorological data. "
            + "; ".join(affected)
            + ". NetCDF diagnostics: "
            + " || ".join(diagnostics)
        )

    if selected_grid_points:
        result.attrs[
            "era5_latitude"
        ] = selected_grid_points[
            0
        ][0]

        result.attrs[
            "era5_longitude"
        ] = selected_grid_points[
            0
        ][1]

        result.attrs[
            "era5_distance_km"
        ] = selected_grid_points[
            0
        ][2]

    return result


def run_lstm(
    model,
    scaler,
    dataframe: pd.DataFrame,
) -> float:
    values = dataframe[
        FEATURE_NAMES
    ].values.astype(
        np.float32
    )

    if scaler is not None:
        flattened = values.reshape(
            -1,
            N_FEATURES,
        )

        flattened = (
            scaler.transform(
                flattened
            )
        )

        values = flattened.reshape(
            SEQ_LEN,
            N_FEATURES,
        )

    values = np.expand_dims(
        values,
        axis=0,
    )

    prediction = np.array(
        model.predict(
            values,
            verbose=0,
        )
    )

    if (
        prediction.ndim == 2
        and prediction.shape[1] == 2
    ):
        return float(
            prediction[
                0,
                LSTM_WILDFIRE_INDEX,
            ]
        )

    if (
        prediction.ndim == 2
        and prediction.shape[1] == 1
    ):
        return float(
            prediction[0, 0]
        )

    if prediction.ndim == 1:
        return float(
            prediction[0]
        )

    raise ValueError(
        "Unexpected LSTM model output shape: "
        f"{prediction.shape}"
    )


def risk_info(
    probability: float,
):
    if probability >= 0.75:
        return (
            "High",
            "🔥🔥🔥",
            "#cc2222",
        )

    if probability >= 0.5:
        return (
            "Moderate",
            "🔥🔥",
            "#dd8800",
        )

    if probability >= 0.25:
        return (
            "Low",
            "🔥",
            "#eecc00",
        )

    return (
        "Minimal",
        "—",
        "#33aa33",
    )


# ============================================================
# PAGE / SESSION STATE
# ============================================================
st.set_page_config(
    page_title=APP_TITLE,
    layout="wide",
)

st.title(APP_TITLE)

st.caption(
    "Click a location. The point snaps to the center of the selected "
    "H3 cell. That same H3 cell center is used for both workflows."
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

[data-testid="stMetricValue"],
[data-testid="stMetricLabel"] {
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
    "cv_point_count": 0,
}

for key, value in (
    state_defaults.items()
):
    if key not in st.session_state:
        st.session_state[key] = value

selected_latitude, selected_longitude = (
    st.session_state[
        "selected_center"
    ]
)

# Use a local date heuristic so the page never contacts CDS during startup.
latest_end_datetime = (
    heuristic_latest_era5_date()
)

latest_end_date = (
    latest_end_datetime.date()
)


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header(
        "Selected H3 cell"
    )

    if st.session_state[
        "h3_cell"
    ]:
        st.write(
            f"H3 r{H3_RES} cell: "
            f"`{st.session_state['h3_cell']}`"
        )

        st.write(
            "Snapped centre: "
            f"`{st.session_state['cell_lat']:.6f}, "
            f"{st.session_state['cell_lon']:.6f}`"
        )

        planned_points = (
            cv_points_for_h3_cell(
                st.session_state[
                    "h3_cell"
                ],
                SPACING_KM,
            )
        )

        st.write(
            "CV extraction points: "
            f"`{len(planned_points)}`"
        )

    else:
        st.info(
            "Click the map to select a cell."
        )

    st.divider()

    st.header(
        "Meteorological settings"
    )

    end_date = st.date_input(
        "Sequence end date",
        value=latest_end_date,
        max_value=latest_end_date,
        help=(
            "The app uses the last complete calendar month as the "
            "maximum. ERA5-Land can occasionally be published later; "
            "choose an earlier month if CDS reports that data is not "
            "available yet."
        ),
    )

    st.caption(
        f"This fetches the last "
        f"{SEQ_LEN} monthly time steps."
    )

    st.divider()

    run_cv = st.button(
        "Prediction Using Computervision",
        type="primary",
        use_container_width=True,
    )

    run_lstm_button = st.button(
        "Prediction Using Meteorological Data",
        use_container_width=True,
    )

    clear = st.button(
        "Clear results",
        use_container_width=True,
    )

if clear:
    for key in [
        "cv_df",
        "cv_imgs",
        "lstm_prob",
        "era5_df",
        "cv_point_count",
    ]:
        st.session_state[key] = (
            state_defaults[key]
        )

    rerun_app()


# ============================================================
# SINGLE MAP
# ============================================================
st.subheader(
    "1) Pick a location"
)

main_map = folium.Map(
    location=[
        selected_latitude,
        selected_longitude,
    ],
    zoom_start=DEFAULT_ZOOM_PICK,
    tiles=ESRI_TILE_URL,
    attr=ESRI_ATTR,
    control_scale=True,
    width="100%",
    height="540px",
)

# Do not render the default Leaflet marker before the first map click.
# The default marker icon can display as an empty square on the initial
# streamlit-folium iframe load.
if (
    st.session_state[
        "h3_cell"
    ]
    is not None
):
    Marker(
        location=[
            st.session_state[
                "cell_lat"
            ],
            st.session_state[
                "cell_lon"
            ],
        ],
        popup="Snapped H3 centre",
    ).add_to(main_map)

if st.session_state[
    "h3_cell"
]:
    polygon_color = "#4a8a4a"

    if (
        st.session_state[
            "lstm_prob"
        ]
        is not None
    ):
        (
            _,
            _,
            polygon_color,
        ) = risk_info(
            float(
                st.session_state[
                    "lstm_prob"
                ]
            )
        )

    folium.Polygon(
        locations=h3_polygon_coords(
            st.session_state[
                "h3_cell"
            ]
        ),
        color=polygon_color,
        fill=True,
        fill_color=polygon_color,
        fill_opacity=0.28,
        weight=2,
        popup=(
            f"H3: "
            f"{st.session_state['h3_cell']}"
            + (
                "<br>p_lstm="
                f"{float(st.session_state['lstm_prob']):.3f}"
                if (
                    st.session_state[
                        "lstm_prob"
                    ]
                    is not None
                )
                else ""
            )
        ),
    ).add_to(main_map)

# Show planned CV extraction points before prediction and coloured
# prediction points after the CV workflow has run.
if st.session_state[
    "h3_cell"
]:
    planned_points = (
        cv_points_for_h3_cell(
            st.session_state[
                "h3_cell"
            ],
            SPACING_KM,
        )
    )

    if (
        st.session_state[
            "cv_df"
        ]
        is None
    ):
        for (
            name,
            point_latitude,
            point_longitude,
        ) in planned_points:
            CircleMarker(
                location=(
                    point_latitude,
                    point_longitude,
                ),
                radius=(
                    4
                    if name != "center"
                    else 6
                ),
                color="white",
                fill=True,
                fill_color="white",
                fill_opacity=0.8,
                popup=name,
            ).add_to(main_map)

    else:
        for _, row in (
            st.session_state[
                "cv_df"
            ].iterrows()
        ):
            point_latitude = float(
                row["lat"]
            )

            point_longitude = float(
                row["lon"]
            )

            probability = row.get(
                "p_wildfire",
                None,
            )

            if (
                probability is None
                or (
                    isinstance(
                        probability,
                        float,
                    )
                    and np.isnan(
                        probability
                    )
                )
            ):
                color = "gray"
                popup = (
                    f"{row['point']}: error"
                )

            else:
                probability = float(
                    probability
                )

                color = (
                    "red"
                    if (
                        probability
                        >= LIKELY_THRESHOLD
                    )
                    else "blue"
                )

                popup = (
                    f"{row['point']}: "
                    f"p={probability:.3f}"
                )

            CircleMarker(
                location=(
                    point_latitude,
                    point_longitude,
                ),
                radius=(
                    5
                    if (
                        row["point"]
                        != "center"
                    )
                    else 7
                ),
                color=color,
                fill=True,
                fill_opacity=0.85,
                popup=popup,
            ).add_to(main_map)

add_map_legend(
    main_map
)

picked = st_folium(
    main_map,
    height=540,
    key="main_map",
    use_container_width=True,
    returned_objects=[
        "last_clicked"
    ],
)

if (
    picked
    and picked.get(
        "last_clicked"
    )
):
    clicked_latitude = float(
        picked[
            "last_clicked"
        ]["lat"]
    )

    clicked_longitude = float(
        picked[
            "last_clicked"
        ]["lng"]
    )

    new_cell = h3.latlng_to_cell(
        clicked_latitude,
        clicked_longitude,
        H3_RES,
    )

    cell_center = (
        h3.cell_to_latlng(
            new_cell
        )
    )

    snapped_center = (
        round(
            float(
                cell_center[0]
            ),
            6,
        ),
        round(
            float(
                cell_center[1]
            ),
            6,
        ),
    )

    current_signature = (
        st.session_state[
            "selected_center"
        ],
        st.session_state[
            "h3_cell"
        ],
    )

    new_signature = (
        snapped_center,
        new_cell,
    )

    if (
        new_signature
        != current_signature
    ):
        st.session_state[
            "selected_center"
        ] = snapped_center

        st.session_state[
            "h3_cell"
        ] = new_cell

        st.session_state[
            "cell_lat"
        ] = snapped_center[0]

        st.session_state[
            "cell_lon"
        ] = snapped_center[1]

        st.session_state[
            "cv_df"
        ] = None

        st.session_state[
            "cv_imgs"
        ] = []

        st.session_state[
            "lstm_prob"
        ] = None

        st.session_state[
            "era5_df"
        ] = None

        st.session_state[
            "cv_point_count"
        ] = len(
            cv_points_for_h3_cell(
                new_cell,
                SPACING_KM,
            )
        )

        rerun_app()

if (
    st.session_state[
        "h3_cell"
    ]
    is None
):
    st.info(
        "Click on the map to choose a location "
        "before running a prediction."
    )


# ============================================================
# RUN MODELS
# ============================================================
want_cv = run_cv
want_lstm = run_lstm_button

if want_cv:
    try:
        if (
            st.session_state[
                "h3_cell"
            ]
            is None
        ):
            st.error(
                "Please click on the map first "
                "to select an H3 cell."
            )

        else:
            mapbox_token = (
                get_mapbox_token()
            )

            with st.spinner(
                "Loading computer-vision model…"
            ):
                cv_model = (
                    load_cv_model_cached()
                )

            points = (
                cv_points_for_h3_cell(
                    st.session_state[
                        "h3_cell"
                    ],
                    SPACING_KM,
                )
            )

            rows = []
            images = []

            with st.spinner(
                "Downloading satellite images and "
                "running CV predictions…"
            ):
                for (
                    name,
                    point_latitude,
                    point_longitude,
                ) in points:
                    try:
                        image = fetch_tile(
                            point_longitude,
                            point_latitude,
                            mapbox_token,
                        )

                        probability = (
                            predict_wildfire_prob_cv(
                                cv_model,
                                image,
                            )
                        )

                        rows.append(
                            {
                                "point": name,
                                "lat": (
                                    point_latitude
                                ),
                                "lon": (
                                    point_longitude
                                ),
                                "p_wildfire": (
                                    probability
                                ),
                            }
                        )

                        images.append(
                            (
                                name,
                                point_latitude,
                                point_longitude,
                                probability,
                                image,
                            )
                        )

                    except Exception as exception:
                        rows.append(
                            {
                                "point": name,
                                "lat": (
                                    point_latitude
                                ),
                                "lon": (
                                    point_longitude
                                ),
                                "p_wildfire": None,
                                "error": str(
                                    exception
                                ),
                            }
                        )

            st.session_state[
                "cv_df"
            ] = pd.DataFrame(
                rows
            )

            st.session_state[
                "cv_imgs"
            ] = images

            st.session_state[
                "cv_point_count"
            ] = len(points)

            rerun_app()

    except Exception as exception:
        st.error(
            "Computer-vision pipeline failed: "
            f"{exception}"
        )

if want_lstm:
    try:
        if (
            st.session_state[
                "h3_cell"
            ]
            is None
        ):
            st.error(
                "Please click on the map first "
                "to select an H3 cell."
            )

        else:
            with st.spinner(
                "Loading meteorological model…"
            ):
                lstm_model = (
                    load_lstm_model_cached()
                )

                scaler = (
                    load_scaler_cached()
                )

            cell_latitude = (
                st.session_state[
                    "cell_lat"
                ]
            )

            cell_longitude = (
                st.session_state[
                    "cell_lon"
                ]
            )

            end_date_string = (
                end_date.strftime(
                    "%Y-%m-%d"
                )
            )

            with st.spinner(
                "Fetching ERA5-Land monthly means…"
            ):
                era5_dataframe = (
                    fetch_era5_sequence(
                        cell_latitude,
                        cell_longitude,
                        end_date_string,
                    )
                )

                st.session_state[
                    "era5_df"
                ] = era5_dataframe

            with st.spinner(
                "Running LSTM inference…"
            ):
                st.session_state[
                    "lstm_prob"
                ] = run_lstm(
                    lstm_model,
                    scaler,
                    era5_dataframe,
                )

            rerun_app()

    except Exception as exception:
        st.error(
            "Meteorological pipeline failed: "
            f"{exception}"
        )


# ============================================================
# RESULTS
# ============================================================
st.subheader(
    "2) Results"
)

if (
    st.session_state[
        "cv_df"
    ]
    is None
    and st.session_state[
        "lstm_prob"
    ]
    is None
):
    st.info(
        "No prediction has been run yet."
    )

else:
    column_one, column_two = (
        st.columns(2)
    )

    with column_one:
        if (
            st.session_state[
                "cv_df"
            ]
            is not None
        ):
            cv_dataframe = (
                st.session_state[
                    "cv_df"
                ].copy()
            )

            (
                likely_count,
                stars,
                emoji,
                message,
            ) = compute_fire_rating(
                cv_dataframe
            )

            center_rows = cv_dataframe.loc[
                cv_dataframe[
                    "point"
                ]
                == "center",
                "p_wildfire",
            ]

            center_probability = (
                center_rows.iloc[0]
                if not center_rows.empty
                else np.nan
            )

            center_line = (
                "<div>Center tile "
                "probability: "
                f"<b>{float(center_probability):.3f}"
                "</b></div>"
                if pd.notna(
                    center_probability
                )
                else (
                    "<div>Center tile "
                    "probability: "
                    "<b>—</b></div>"
                )
            )

            body = f"""
<div style="font-size:2rem; font-weight:900; margin:0 0 10px 0;">
    {html.escape(emoji)}
</div>
<div>
    Likely fire tiles:
    <b>{likely_count} / {len(cv_dataframe)}</b>
    (threshold ≥ {LIKELY_THRESHOLD:.2f})
</div>
<div>
    Downloaded images:
    <b>{len(cv_dataframe)}</b>
</div>
{center_line}
<div style="margin-top:8px;">
    {html.escape(message)}
</div>
"""

            render_result_card(
                "Computer vision result",
                body,
            )

        else:
            render_result_card(
                "Computer vision result",
                "<div>Not run yet.</div>",
            )

    with column_two:
        if (
            st.session_state[
                "lstm_prob"
            ]
            is not None
        ):
            lstm_probability = float(
                st.session_state[
                    "lstm_prob"
                ]
            )

            (
                label,
                emoji,
                color,
            ) = risk_info(
                lstm_probability
            )

            threshold_status = (
                "Above threshold"
                if (
                    lstm_probability
                    >= LSTM_THRESHOLD
                )
                else "Below threshold"
            )

            body = f"""
<div style="font-size:1.8rem; font-weight:900; color:{html.escape(color)}; margin:0 0 8px 0;">
    {html.escape(emoji)} {html.escape(label)}
</div>
<div>
    Fire probability:
    <b>{lstm_probability:.3f}</b>
</div>
<div>
    {html.escape(threshold_status)}
    (threshold = {LSTM_THRESHOLD:.2f})
</div>
<div style="margin-top:8px;">
    H3 cell:
    <code>{html.escape(str(st.session_state["h3_cell"]))}</code>
</div>
"""

            render_result_card(
                "Meteorological result",
                body,
            )

        else:
            render_result_card(
                "Meteorological result",
                "<div>Not run yet.</div>",
            )

    (
        computer_vision_tab,
        meteorological_tab,
    ) = st.tabs(
        [
            "Computer vision details",
            "Meteorological details",
        ]
    )

    with computer_vision_tab:
        if (
            st.session_state[
                "cv_df"
            ]
            is None
        ):
            st.info(
                "Run the computer-vision model "
                "to see details here."
            )

        else:
            dataframe = (
                st.session_state[
                    "cv_df"
                ].copy()
            )

            images = (
                st.session_state.get(
                    "cv_imgs",
                    [],
                )
            )

            st.write(
                "Extraction points used: "
                f"**{len(dataframe)}**"
            )

            st.dataframe(
                dataframe,
                use_container_width=True,
            )

            st.download_button(
                "Download CV CSV",
                data=(
                    dataframe.to_csv(
                        index=False
                    ).encode("utf-8")
                ),
                file_name=(
                    "wildfire_predictions_"
                    "cv_h3cell.csv"
                ),
                mime="text/csv",
            )

            st.markdown(
                "**Satellite images**"
            )

            if images:
                column_count = 4

                for start in range(
                    0,
                    len(images),
                    column_count,
                ):
                    columns = st.columns(
                        column_count
                    )

                    for column, item in zip(
                        columns,
                        images[
                            start:
                            start
                            + column_count
                        ],
                    ):
                        (
                            name,
                            point_latitude,
                            point_longitude,
                            probability,
                            image,
                        ) = item

                        with column:
                            caption = (
                                f"{name}\n"
                                f"{point_longitude:.6f}, "
                                f"{point_latitude:.6f}\n"
                            )

                            caption += (
                                f"p={probability:.3f}"
                                if (
                                    probability
                                    is not None
                                )
                                else "p=None"
                            )

                            st.image(
                                image,
                                caption=caption,
                                use_container_width=True,
                            )

            else:
                st.warning(
                    "No images were produced. "
                    "Check the errors in the "
                    "table above."
                )

    with meteorological_tab:
        if (
            st.session_state[
                "lstm_prob"
            ]
            is None
        ):
            st.info(
                "Run the meteorological model "
                "to see details here."
            )

        else:
            probability = float(
                st.session_state[
                    "lstm_prob"
                ]
            )

            (
                label,
                emoji,
                color,
            ) = risk_info(
                probability
            )

            st.markdown(
                (
                    "<div class='result-card'>"
                    f"<span style='font-size:2rem;"
                    f"color:{color};font-weight:800;'>"
                    f"{emoji} {label} Risk"
                    "</span><br><br>"
                    "H3 cell: "
                    f"<code>{st.session_state['h3_cell']}</code>"
                    "<br>"
                    "H3 centre: "
                    f"{st.session_state['cell_lat']:.6f}, "
                    f"{st.session_state['cell_lon']:.6f}"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )

            if (
                st.session_state[
                    "era5_df"
                ]
                is not None
            ):
                dataframe = (
                    st.session_state[
                        "era5_df"
                    ].copy()
                )

                st.dataframe(
                    dataframe,
                    use_container_width=True,
                )

                st.download_button(
                    "Download ERA5 CSV",
                    data=(
                        dataframe.to_csv(
                            index=False
                        ).encode("utf-8")
                    ),
                    file_name=(
                        f"era5_"
                        f"{st.session_state['h3_cell']}_"
                        f"{end_date}.csv"
                    ),
                    mime="text/csv",
                )

                st.markdown(
                    "**Feature trends**"
                )

                columns = st.columns(3)

                chart_features = [
                    (
                        "2m_temperature",
                        "Temperature (K)",
                    ),
                    (
                        "total_precipitation",
                        "Precipitation (m)",
                    ),
                    (
                        "wind_total",
                        "Wind speed (m/s)",
                    ),
                    (
                        "volumetric_soil_water_layer_1",
                        "Soil water (m³/m³)",
                    ),
                    (
                        "surface_solar_radiation_downwards",
                        "Solar radiation (J/m²)",
                    ),
                    (
                        "leaf_area_index_high_vegetation",
                        "LAI high vegetation",
                    ),
                ]

                for index, (
                    feature,
                    title,
                ) in enumerate(
                    chart_features
                ):
                    with columns[
                        index % 3
                    ]:
                        st.caption(
                            title
                        )

                        if (
                            feature
                            in dataframe.columns
                        ):
                            st.line_chart(
                                dataframe.set_index(
                                    "date"
                                )[feature],
                                height=140,
                                use_container_width=True,
                            )


st.caption(
    "Streamlit app combining Mapbox + VGG16 "
    "and ERA5-Land + LSTM"
)
