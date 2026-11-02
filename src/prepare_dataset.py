import random
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import requests
import requests_cache
import torch
from PIL import Image, ExifTags
from scipy.ndimage import minimum_filter
from torch.utils.data import Dataset
from torchvision import transforms

# PATHS
ROOT = Path("pm25dataset")
TRAIN_CSV = ROOT / "train" / "metadata_enriched.csv"
TEST_CSV = ROOT / "test" / "metadata_enriched.csv"

TRAIN_IMG_DIR = ROOT / "train" / "images"
TEST_IMG_DIR = ROOT / "test" / "images"

TRAIN_ENRICHED_CSV = ROOT / "train" / "metadata_enriched.csv"
TEST_ENRICHED_CSV = ROOT / "test" / "metadata_enriched.csv"

CACHE_DIR = ROOT / "cache"
OPEN_METEO_CACHE = CACHE_DIR / "open_meteo_cache"
DCP_DIR_TRAIN = ROOT / "train" / "images_dehazed"
DCP_DIR_TEST = ROOT / "test" / "images_dehazed"

IMAGE_SIZE = 300

IMAGE_COL_CANDIDATES = ["image", "filename", "file_name", "img", "image_name", "path"]
TARGET_REG_COL_CANDIDATES = ["aqi", "pm25", "pm2.5", "pm25_aqi", "label", "target"]
TARGET_CLASS_COL_CANDIDATES = ["aqi_level", "class", "label_cls", "category"]

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "apparent_temperature",
    "precipitation",
    "rain",
    "snowfall",
    "snow_depth",
    "cloud_cover",
    "visibility",
    "surface_pressure",
    "weather_code",
]

WEATHER_NUMERIC_COLS = [
    "temperature_2m",
    "relative_humidity_2m",
    "apparent_temperature",
    "precipitation",
    "rain",
    "snowfall",
    "snow_depth",
    "cloud_cover",
    "visibility",
    "surface_pressure",
    "weather_code",
]

WEATHER_BINARY_COLS = [
    "is_fog",
    "is_rain",
    "is_snow",
    "is_thunderstorm",
    "is_freezing_precip",
    "is_cloudy",
]

DCP_FEATURE_COLS = [
    "dcp_dark_mean",
    "dcp_dark_std",
    "dcp_atmospheric_light_r",
    "dcp_atmospheric_light_g",
    "dcp_atmospheric_light_b",
    "dcp_atmospheric_light_mean",
    "dcp_transmission_mean",
    "dcp_transmission_std",
    "dcp_haze_score",
]

ALL_EXTRA_FEATURE_COLS = WEATHER_NUMERIC_COLS + WEATHER_BINARY_COLS + DCP_FEATURE_COLS

# TRANSFORMS
train_tfms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

test_tfms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# UTILS
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def find_first_existing_column(df: pd.DataFrame, candidates: List[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"No column founf in {candidates}. Column: {list(df.columns)}"
    )

def _safe_float(x):
    try:
        if pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan

# TIMESTAMP PARSING
def _extract_exif_datetime(img_path: Path) -> Optional[pd.Timestamp]:
    try:
        img = Image.open(img_path)
        exif = img.getexif()
        if not exif:
            return None

        exif_map = {}
        for tag_id, value in exif.items():
            tag = ExifTags.TAGS.get(tag_id, tag_id)
            exif_map[tag] = value

        for k in ["DateTimeOriginal", "DateTimeDigitized", "DateTime"]:
            if k in exif_map:
                dt_str = str(exif_map[k])
                dt = pd.to_datetime(dt_str, errors="coerce", format="%Y:%m:%d %H:%M:%S")
                if pd.notna(dt):
                    return dt
    except Exception:
        return None

    return None

def parse_capture_timestamp(row: pd.Series, img_path: Optional[Path] = None) -> pd.Timestamp:
    candidate_cols = [
        "captured_at",
        "captured_datetime",
        "datetime",
        "timestamp",
        "capture_time",
        "captured_time",
    ]

    parsed = None
    for col in candidate_cols:
        if col in row.index:
            val = row[col]
            if pd.notna(val):
                parsed = pd.to_datetime(val, errors="coerce")
                if pd.notna(parsed):
                    break

    if parsed is not None and pd.notna(parsed):
        return parsed

    if img_path is not None and img_path.exists():
        exif_dt = _extract_exif_datetime(img_path)
        if exif_dt is not None:
            return exif_dt

    if "captured_at" in row.index and pd.notna(row["captured_at"]):
        d = pd.to_datetime(str(row["captured_at"]), errors="coerce")
        if pd.notna(d):
            return d.normalize() + pd.Timedelta(hours=12)

    raise ValueError("Timestamp invalid")


# OPEN-METEO CACHE + MAPPING WEATHER_CODE
def create_cached_session(expire_after_seconds: int = 60 * 60 * 24 * 365):
    ensure_dir(CACHE_DIR)
    session = requests_cache.CachedSession(
        cache_name=str(OPEN_METEO_CACHE),
        backend="sqlite",
        expire_after=expire_after_seconds,
        allowable_methods=("GET",),
    )
    return session

def map_weather_code_features(code) -> Dict[str, float]:
    if pd.isna(code):
        return {
            "is_fog": np.nan,
            "is_rain": np.nan,
            "is_snow": np.nan,
            "is_thunderstorm": np.nan,
            "is_freezing_precip": np.nan,
            "is_cloudy": np.nan,
            "cloud_class": np.nan,
        }

    code = int(code)

    fog_codes = {45, 48}
    rain_codes = {51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82}
    snow_codes = {71, 73, 75, 77, 85, 86}
    thunder_codes = {95, 96, 99}
    freezing_precip_codes = {56, 57, 66, 67}

    if code == 0:
        cloud_class = 0  # clear
    elif code == 1:
        cloud_class = 1  # mainly clear
    elif code == 2:
        cloud_class = 2  # partly cloudy
    elif code == 3:
        cloud_class = 3  # overcast
    else:
        cloud_class = 4  # other

    return {
        "is_fog": float(code in fog_codes),
        "is_rain": float(code in rain_codes),
        "is_snow": float(code in snow_codes),
        "is_thunderstorm": float(code in thunder_codes),
        "is_freezing_precip": float(code in freezing_precip_codes),
        "is_cloudy": float(code in {1, 2, 3}),
        "cloud_class": float(cloud_class),
    }

def fetch_weather_for_timestamp(lat: float, lon: float, ts: pd.Timestamp, session=None, ) -> Dict[str, float]:

    if session is None:
        session = create_cached_session()

    if pd.isna(lat) or pd.isna(lon) or pd.isna(ts):
        return {col: np.nan for col in WEATHER_NUMERIC_COLS + WEATHER_BINARY_COLS + ["cloud_class"]}

    day_str = ts.strftime("%Y-%m-%d")

    params = {
        "latitude": float(lat),
        "longitude": float(lon),
        "start_date": day_str,
        "end_date": day_str,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "auto",
    }

    resp = session.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    if not times:
        return {col: np.nan for col in WEATHER_NUMERIC_COLS + WEATHER_BINARY_COLS + ["cloud_class"]}

    time_index = pd.to_datetime(times, errors="coerce")
    if len(time_index) == 0:
        return {col: np.nan for col in WEATHER_NUMERIC_COLS + WEATHER_BINARY_COLS + ["cloud_class"]}

    deltas = np.abs((time_index - ts).astype("timedelta64[s]").astype(np.int64))
    nearest_idx = int(np.argmin(deltas))

    feats = {}
    for var in HOURLY_VARS:
        arr = hourly.get(var, [])
        if arr is None or len(arr) <= nearest_idx:
            feats[var] = np.nan
        else:
            feats[var] = _safe_float(arr[nearest_idx])

    code_feats = map_weather_code_features(feats.get("weather_code", np.nan))
    feats.update(code_feats)

    return feats
def fetch_weather_for_timestamp_with_retry(lat, lon, ts, session, max_retries=8, base_sleep=3.0):
    last_err = None

    for attempt in range(max_retries):
        try:
            return fetch_weather_for_timestamp(
                lat=lat,
                lon=lon,
                ts=ts,
                session=session,
            )

        except requests.HTTPError as e:
            status = None
            retry_after = None

            if hasattr(e, "response") and e.response is not None:
                status = e.response.status_code
                retry_after = e.response.headers.get("Retry-After")

            if status == 429:
                if retry_after is not None:
                    wait_s = float(retry_after)
                else:
                    wait_s = base_sleep * (2 ** attempt) + random.uniform(0.0, 1.0)

                print(f"Too Many Requests -> waiting {wait_s:.1f}s")
                time.sleep(wait_s)
                last_err = e
                continue

            raise

        except Exception as e:
            last_err = e
            if attempt == max_retries - 1:
                break

            wait_s = base_sleep * (2 ** attempt) + random.uniform(0.0, 1.0)
            print(f"Error -> waiting {wait_s:.1f}s ({e})")
            time.sleep(wait_s)

    raise last_err if last_err is not None else RuntimeError("weather unknown")


def fill_missing_weather_only(enriched_csv: Path,image_dir: Path,save_every: int = 20,throttle_seconds: float = 0.5,) -> pd.DataFrame:
    if not enriched_csv.exists():
        raise FileNotFoundError(f"File non trovato: {enriched_csv}")

    df = pd.read_csv(enriched_csv)
    image_col = find_first_existing_column(df, IMAGE_COL_CANDIDATES)

    session = create_cached_session()
    total = len(df)

    print("\n" + "=" * 60)
    print("WEATHER FILLING")
    print(f"FILE: {enriched_csv}")
    print(f"TOTAL IMAGES: {total}")
    print("=" * 60)

    for loop_idx, (idx, row) in enumerate(df.iterrows(), start=1):
        img_name = str(row[image_col])
        img_path = image_dir / img_name

        print(f"\nIMG {loop_idx}/{total} -> {img_name}")

        weather_done = pd.notna(row.get("weather_timestamp_used", np.nan))
        if weather_done:
            continue

        ts = None
        try:
            if pd.notna(row.get("capture_timestamp", np.nan)):
                ts = pd.to_datetime(row["capture_timestamp"]).to_pydatetime()
            else:
                ts = parse_capture_timestamp(row, img_path=img_path)
                df.at[idx, "capture_timestamp"] = ts.isoformat()
        except Exception:
            ts = None
            df.at[idx, "capture_timestamp"] = None

        print(f"WEATHER START [{loop_idx}/{total}]")

        try:
            feats = fetch_weather_for_timestamp_with_retry(lat=row["latitude"],lon=row["longitude"],ts=ts,session=session,max_retries=8,base_sleep=3.0,)

            for k, v in feats.items():
                df.at[idx, k] = v

            if ts is not None:
                df.at[idx, "weather_timestamp_used"] = ts.isoformat()

            print(f"WEATHER END [{loop_idx}/{total}]")

            if throttle_seconds > 0:
                time.sleep(throttle_seconds)

        except Exception as e:
            for c in WEATHER_NUMERIC_COLS + WEATHER_BINARY_COLS + ["cloud_class"]:
                df.at[idx, c] = np.nan
            df.at[idx, "weather_timestamp_used"] = None
            print(f"WEATHER ERROR [{loop_idx}/{total}] -> {e}")

        if loop_idx % save_every == 0:
            ensure_dir(enriched_csv.parent)
            df.to_csv(enriched_csv, index=False)

    ensure_dir(enriched_csv.parent)
    df.to_csv(enriched_csv, index=False)

    print("\n" + "=" * 60)
    print("END WEATHER FILLING")
    print(f"SAVED IN: {enriched_csv}")
    print("=" * 60)

    return df

# DCP
def _load_rgb_float(img_path: Path) -> np.ndarray:
    img = Image.open(img_path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr

def compute_dark_channel(img_rgb: np.ndarray, patch_size: int = 15) -> np.ndarray:
    min_rgb = np.min(img_rgb, axis=2)
    dark = minimum_filter(min_rgb, size=patch_size)
    return dark

def estimate_atmospheric_light(img_rgb: np.ndarray, dark_channel: np.ndarray, top_percent: float = 0.001) -> np.ndarray:
    h, w = dark_channel.shape
    n_pixels = h * w
    n_search = max(1, int(n_pixels * top_percent))

    dark_vec = dark_channel.reshape(-1)
    img_vec = img_rgb.reshape(-1, 3)

    indices = np.argpartition(dark_vec, -n_search)[-n_search:]
    candidate_pixels = img_vec[indices]
    candidate_intensity = candidate_pixels.sum(axis=1)
    A = candidate_pixels[np.argmax(candidate_intensity)]
    return A

def estimate_transmission(img_rgb: np.ndarray, A: np.ndarray, patch_size: int = 15, omega: float = 0.95) -> np.ndarray:
    normed = img_rgb / np.maximum(A.reshape(1, 1, 3), 1e-6)
    dark_normed = compute_dark_channel(normed, patch_size=patch_size)
    t = 1.0 - omega * dark_normed
    return np.clip(t, 0.0, 1.0)

def refine_transmission_guided(gray_img: np.ndarray,transmission: np.ndarray,radius: int = 40,eps: float = 1e-3) -> np.ndarray:
    I = gray_img.astype(np.float32)
    p = transmission.astype(np.float32)

    I = np.clip(I, 0.0, 1.0)
    p = np.nan_to_num(p, nan=0.0, posinf=1.0, neginf=0.0)
    p = np.clip(p, 0.0, 1.0)

    ksize = (2 * radius + 1, 2 * radius + 1)

    ones = np.ones_like(I, dtype=np.float32)
    N = cv2.boxFilter(ones, ddepth=-1, ksize=ksize, borderType=cv2.BORDER_REFLECT)

    mean_I = cv2.boxFilter(I, ddepth=-1, ksize=ksize, borderType=cv2.BORDER_REFLECT) / N
    mean_p = cv2.boxFilter(p, ddepth=-1, ksize=ksize, borderType=cv2.BORDER_REFLECT) / N
    mean_Ip = cv2.boxFilter(I * p, ddepth=-1, ksize=ksize, borderType=cv2.BORDER_REFLECT) / N
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(I * I, ddepth=-1, ksize=ksize, borderType=cv2.BORDER_REFLECT) / N
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, ddepth=-1, ksize=ksize, borderType=cv2.BORDER_REFLECT) / N
    mean_b = cv2.boxFilter(b, ddepth=-1, ksize=ksize, borderType=cv2.BORDER_REFLECT) / N

    q = mean_a * I + mean_b
    q = np.nan_to_num(q, nan=0.0, posinf=1.0, neginf=0.0)
    q = np.clip(q, 0.0, 1.0)

    return q.astype(np.float32)

def recover_radiance(img_rgb: np.ndarray, A: np.ndarray, transmission: np.ndarray, t0: float = 0.1) -> np.ndarray:
    t = np.maximum(transmission, t0)[..., None]
    J = (img_rgb - A.reshape(1, 1, 3)) / t + A.reshape(1, 1, 3)
    return np.clip(J, 0.0, 1.0)

def compute_dcp_advanced(img_path: Path,save_dehazed: bool = True,dehazed_output_path: Optional[Path] = None,patch_size: int = 15,omega: float = 0.95,t0: float = 0.1,) -> Dict[str, float]:
    img = Image.open(img_path).convert("RGB")
    img_rgb = np.asarray(img).astype(np.float32) / 255.0

    # Dark channel
    dark = compute_dark_channel(img_rgb, patch_size=patch_size)

    # Atmospheric light
    A = estimate_atmospheric_light(img_rgb, dark, top_percent=0.001)
    A = np.asarray(A, dtype=np.float32)
    A = np.clip(A, 1e-6, 1.0)

    # Rough transmission
    t_est = estimate_transmission(
        img_rgb,
        A,
        patch_size=patch_size,
        omega=omega,
    )
    t_est = np.asarray(t_est, dtype=np.float32)
    t_est = np.nan_to_num(t_est, nan=t0, posinf=1.0, neginf=t0)
    t_est = np.clip(t_est, t0, 1.0)

    # Refinement
    gray = cv2.cvtColor((img_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    t_ref = refine_transmission_guided(
        gray_img=gray,
        transmission=t_est,
        radius=40,
        eps=1e-3,
    )
    t_ref = np.asarray(t_ref, dtype=np.float32)
    t_ref = np.nan_to_num(t_ref, nan=t0, posinf=1.0, neginf=t0)
    t_ref = np.clip(t_ref, t0, 1.0)

    # Scene radiance
    t_broadcast = np.repeat(t_ref[:, :, None], 3, axis=2)
    J = (img_rgb - A.reshape(1, 1, 3)) / np.maximum(t_broadcast, t0) + A.reshape(1, 1, 3)
    J = np.nan_to_num(J, nan=0.0, posinf=1.0, neginf=0.0)
    J = np.clip(J, 0.0, 1.0)

    # Save image
    if save_dehazed and dehazed_output_path is not None:
        ensure_dir(dehazed_output_path.parent)
        out = (J * 255.0).astype(np.uint8)
        Image.fromarray(out).save(dehazed_output_path)

    # DCP features
    dark_mean = float(np.mean(dark))
    dark_std = float(np.std(dark))
    transmission_mean = float(np.mean(t_ref))
    transmission_std = float(np.std(t_ref))
    atmospheric_r = float(A[0])
    atmospheric_g = float(A[1])
    atmospheric_b = float(A[2])
    atmospheric_mean = float(np.mean(A))

    gray_in = gray
    gray_out = cv2.cvtColor((J * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    contrast_in = float(np.std(gray_in))
    contrast_out = float(np.std(gray_out))
    contrast_gain = float(contrast_out - contrast_in)

    dehazed_mean = float(np.mean(J))
    dehazed_std = float(np.std(J))

    feats = {
        "dcp_dark_mean": dark_mean,
        "dcp_dark_std": dark_std,
        "dcp_transmission_mean": transmission_mean,
        "dcp_transmission_std": transmission_std,
        "dcp_atmospheric_r": atmospheric_r,
        "dcp_atmospheric_g": atmospheric_g,
        "dcp_atmospheric_b": atmospheric_b,
        "dcp_atmospheric_mean": atmospheric_mean,
        "dcp_contrast_gain": contrast_gain,
        "dcp_dehazed_mean": dehazed_mean,
        "dcp_dehazed_std": dehazed_std,
    }

    return feats

# BUILD ENRICHED METADATA
def build_enriched_metadata(input_csv: Path,image_dir: Path,output_csv: Path,save_dehazed: bool = True,dehazed_dir: Optional[Path] = None,use_weather: bool = True, use_dcp: bool = True,overwrite: bool = False,save_every: int = 20,) -> pd.DataFrame:
    if output_csv.exists() and not overwrite:
        return pd.read_csv(output_csv)

    df = pd.read_csv(input_csv)
    image_col = find_first_existing_column(df, IMAGE_COL_CANDIDATES)

    session = create_cached_session()
    total = len(df)

    numeric_cols = WEATHER_NUMERIC_COLS + WEATHER_BINARY_COLS + ["cloud_class"] + DCP_FEATURE_COLS
    for c in numeric_cols:
        if c not in df.columns:
            df[c] = np.nan

    string_cols = ["capture_timestamp", "weather_timestamp_used", "dehazed_image_path"]
    for c in string_cols:
        if c not in df.columns:
            df[c] = None

    print("\n" + "=" * 60)
    print("BUILD ENRICHED METADATA")
    print(f"TOTAL IMAGES: {total}")
    print("=" * 60)

    for loop_idx, (idx, row) in enumerate(df.iterrows(), start=1):
        img_name = str(row[image_col])
        img_path = image_dir / img_name

        print(f"\nIMG {loop_idx}/{total} -> {img_name}")

        try:
            ts = parse_capture_timestamp(row, img_path=img_path)
            df.at[idx, "capture_timestamp"] = ts.isoformat()
        except Exception:
            ts = None
            df.at[idx, "capture_timestamp"] = None

        # meteo row-by-row con cache
        if use_weather:
            print(f"WEATHER START [{loop_idx}/{total}]")
            try:
                feats = fetch_weather_for_timestamp(
                    lat=row["latitude"],
                    lon=row["longitude"],
                    ts=ts,
                    session=session,
                )
                for k, v in feats.items():
                    df.at[idx, k] = v

                if ts is not None:
                    df.at[idx, "weather_timestamp_used"] = ts.isoformat()

                print(f"WEATHER END[{loop_idx}/{total}]")
            except Exception as e:
                for c in WEATHER_NUMERIC_COLS + WEATHER_BINARY_COLS + ["cloud_class"]:
                    df.at[idx, c] = np.nan
                df.at[idx, "weather_timestamp_used"] = None
                print(f"WEATHER ERROR [{loop_idx}/{total}] -> {e}")

        if use_dcp:
            print(f"DCP START [{loop_idx}/{total}]")
            try:
                if dehazed_dir is not None:
                    out_path = dehazed_dir / img_name
                else:
                    out_path = None

                dcp_feats = compute_dcp_advanced(
                    img_path=img_path,
                    save_dehazed=save_dehazed,
                    dehazed_output_path=out_path,
                    patch_size=15,
                    omega=0.95,
                    t0=0.1,
                )
                for k, v in dcp_feats.items():
                    df.at[idx, k] = v

                if out_path is not None:
                    df.at[idx, "dehazed_image_path"] = str(out_path)

                print(f"DCP END [{loop_idx}/{total}]")
            except Exception as e:
                for c in DCP_FEATURE_COLS:
                    df.at[idx, c] = np.nan
                df.at[idx, "dehazed_image_path"] = None
                print(f"DCP ERROR [{loop_idx}/{total}] -> {e}")

        if loop_idx % save_every == 0:
            ensure_dir(output_csv.parent)
            df.to_csv(output_csv, index=False)

    ensure_dir(output_csv.parent)
    df.to_csv(output_csv, index=False)

    print("\n" + "=" * 60)
    print("END BUILD ENRICHED METADATA")
    print(f"SAVED IN: {output_csv}")
    print("=" * 60)

    return df

# EXTRA FEATURES
def fit_extra_feature_stats(df_train: pd.DataFrame, feature_cols: List[str]) -> Dict[str, Dict[str, float]]:
    stats = {}
    for col in feature_cols:
        x = pd.to_numeric(df_train[col], errors="coerce")

        is_binary = col in WEATHER_BINARY_COLS
        if is_binary:
            stats[col] = {"kind": "binary","fillna": 0.0,}
        else:
            med = float(np.nanmedian(x.values)) if np.isfinite(np.nanmedian(x.values)) else 0.0
            filled = np.where(np.isnan(x.values), med, x.values).astype(np.float32)
            mean = float(np.mean(filled))
            std = float(np.std(filled))
            if std < 1e-8:
                std = 1.0

            stats[col] = {"kind": "continuous","fillna": med,"mean": mean,"std": std,}
    return stats

def apply_extra_feature_stats(df: pd.DataFrame, stats: Dict[str, Dict[str, float]], inplace: bool = False) -> pd.DataFrame:
    if not inplace:
        df = df.copy()

    for col, cfg in stats.items():
        x = pd.to_numeric(df[col], errors="coerce")

        if cfg["kind"] == "binary":
            x = x.fillna(cfg["fillna"]).astype(np.float32)
            df[col] = x
        else:
            x = x.fillna(cfg["fillna"]).astype(np.float32)
            x = (x - cfg["mean"]) / cfg["std"]
            df[col] = x

    return df

# DATASET
class PM25Dataset(Dataset):
    def __init__(self,
        csv_path,
        image_dir,
        task="regression",
        transform=None,
        use_weather=False,
        use_dcp=False,
        use_dehazed_images=False,
    ):
        self.df = pd.read_csv(csv_path)
        self.image_dir = Path(image_dir)
        self.task = task
        self.transform = transform
        self.use_weather = use_weather
        self.use_dcp = use_dcp
        self.use_dehazed_images = use_dehazed_images

        self.image_col = find_first_existing_column(self.df, IMAGE_COL_CANDIDATES)

        if task == "regression":
            self.target_col = find_first_existing_column(self.df, TARGET_REG_COL_CANDIDATES)
        else:
            class_col = None
            for c in TARGET_CLASS_COL_CANDIDATES:
                if c in self.df.columns:
                    class_col = c
                    break
            if class_col is not None:
                self.target_col = class_col
            else:
                reg_col = find_first_existing_column(self.df, TARGET_REG_COL_CANDIDATES)
                self.target_col = reg_col

        self.weather_cols = [
            "temperature_2m",
            "relative_humidity_2m",
            "apparent_temperature",
            "precipitation",
            "rain",
            "snowfall",
            "snow_depth",
            "cloud_cover",
            "visibility",
            "surface_pressure",
            "weather_code",
            "is_fog",
            "is_rain",
            "is_snow",
            "is_thunderstorm",
            "is_freezing_precip",
            "is_cloudy",
            "cloud_class",
        ]

        self.dcp_cols = [
            "dcp_dark_mean",
            "dcp_dark_std",
            "dcp_atmospheric_light_r",
            "dcp_atmospheric_light_g",
            "dcp_atmospheric_light_b",
            "dcp_atmospheric_light_mean",
            "dcp_transmission_mean",
            "dcp_transmission_std",
            "dcp_haze_score",
            "dcp_atmospheric_r",
            "dcp_atmospheric_g",
            "dcp_atmospheric_b",
            "dcp_atmospheric_mean",
            "dcp_contrast_gain",
            "dcp_dehazed_mean",
            "dcp_dehazed_std",
        ]

        if self.use_weather:
            missing_weather = [c for c in self.weather_cols if c not in self.df.columns]
            if len(missing_weather) > 0:
                raise ValueError(f"use_weather=True but weather column missing: {missing_weather}")

        if self.use_dcp:
            missing_dcp = [c for c in self.dcp_cols if c not in self.df.columns]
            if len(missing_dcp) > 0:
                raise ValueError(f"use_dcp=True but DCP column missing: {missing_dcp}")

        if self.use_dehazed_images and "dehazed_image_path" not in self.df.columns:
            raise ValueError("use_dehazed_images=True but dehazed path column missing")

    def __len__(self):
        return len(self.df)

    def _resolve_image_path(self, row):
        if self.use_dehazed_images:
            p = str(row["dehazed_image_path"])
            p = p.replace("\\", "/")
            return Path(p)

        img_name = str(row[self.image_col])
        return self.image_dir / img_name

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = self._resolve_image_path(row)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.task == "regression":
            y = torch.tensor(np.float32(row[self.target_col]), dtype=torch.float32)
        else:
            val = row[self.target_col]
            if self.target_col in TARGET_CLASS_COL_CANDIDATES and pd.notna(val):
                y = int(val)
            else:
                y = int(val)
            y = torch.tensor(y, dtype=torch.long)

        extras = []

        if self.use_weather:
            for col in self.weather_cols:
                v = row[col]
                if pd.isna(v):
                    v = 0.0
                extras.append(float(v))

        if self.use_dcp:
            for col in self.dcp_cols:
                v = row[col]
                if pd.isna(v):
                    v = 0.0
                extras.append(float(v))

        if len(extras) > 0:
            x_extra = torch.tensor(extras, dtype=torch.float32)
        else:
            x_extra = torch.empty(0, dtype=torch.float32)

        return image, x_extra, y

# HELPER
def prepare_train_test_enriched_metadata(overwrite: bool = False, save_dehazed: bool = True,):
    train_df = build_enriched_metadata(
        input_csv=TRAIN_CSV,
        image_dir=TRAIN_IMG_DIR,
        output_csv=TRAIN_ENRICHED_CSV,
        save_dehazed=save_dehazed,
        dehazed_dir=DCP_DIR_TRAIN,
        use_weather=True,
        use_dcp=True,
        overwrite=overwrite,
    )

    test_df = build_enriched_metadata(
        input_csv=TEST_CSV,
        image_dir=TEST_IMG_DIR,
        output_csv=TEST_ENRICHED_CSV,
        save_dehazed=save_dehazed,
        dehazed_dir=DCP_DIR_TEST,
        use_weather=True,
        use_dcp=True,
        overwrite=overwrite,
    )

    return train_df, test_df

def main():
    print("=" * 80)
    print("PM2.5 DATASET PREPROCESSING START")
    print("=" * 80)

    required_paths = [
        TRAIN_CSV,
        TEST_CSV,
        TRAIN_IMG_DIR,
        TEST_IMG_DIR,
    ]

    for p in required_paths:
        if not Path(p).exists():
            raise FileNotFoundError(f"Path mancante: {p}")

    print(f"[OK] TRAIN_CSV      : {TRAIN_CSV}")
    print(f"[OK] TEST_CSV       : {TEST_CSV}")
    print(f"[OK] TRAIN_IMG_DIR  : {TRAIN_IMG_DIR}")
    print(f"[OK] TEST_IMG_DIR   : {TEST_IMG_DIR}")

    print("\n[1/2] Building enriched TRAIN metadata...")
    train_df = build_enriched_metadata(
        input_csv=TRAIN_CSV,
        image_dir=TRAIN_IMG_DIR,
        output_csv=TRAIN_ENRICHED_CSV,
        save_dehazed=True,
        dehazed_dir=DCP_DIR_TRAIN,
        use_weather=True,
        use_dcp=True,
        overwrite=False,
    )
    print(f"[DONE] Train enriched CSV saved in: {TRAIN_ENRICHED_CSV}")
    print(f"[INFO] Train rows: {len(train_df)}")

    print("\n[2/2] Building enriched TEST metadata...")
    test_df = build_enriched_metadata(
        input_csv=TEST_CSV,
        image_dir=TEST_IMG_DIR,
        output_csv=TEST_ENRICHED_CSV,
        save_dehazed=True,
        dehazed_dir=DCP_DIR_TEST,
        use_weather=True,
        use_dcp=True,
        overwrite=False,
    )
    print(f"[DONE] Test enriched CSV saved in: {TEST_ENRICHED_CSV}")
    print(f"[INFO] Test rows: {len(test_df)}")

    print("\n--- TRAIN enriched columns preview ---")
    preview_cols = [
        c for c in [
            "capture_timestamp",
            "temperature_2m",
            "relative_humidity_2m",
            "visibility",
            "weather_code",
            "is_fog",
            "is_rain",
            "is_snow",
            "is_thunderstorm",
            "dcp_dark_mean",
            "dcp_transmission_mean",
            "dcp_haze_score",
            "dehazed_image_path",
        ] if c in train_df.columns
    ]
    print(train_df[preview_cols].head(3))

    print("\n--- TEST enriched columns preview ---")
    preview_cols_test = [c for c in preview_cols if c in test_df.columns]
    print(test_df[preview_cols_test].head(3))

    print("\n" + "=" * 80)
    print("PM2.5 DATASET PREPROCESSING COMPLETED")
    print("=" * 80)
    print(f"Train enriched file: {TRAIN_ENRICHED_CSV}")
    print(f"Test enriched file : {TEST_ENRICHED_CSV}")
    print(f"Train dehazed dir  : {DCP_DIR_TRAIN}")
    print(f"Test dehazed dir   : {DCP_DIR_TEST}")

if __name__ == "__main__":
    main()