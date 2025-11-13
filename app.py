import logging
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import requests
from flask import Flask, jsonify, render_template

try:
    import xgboost as xgb  # type: ignore
except Exception:  # pragma: no cover - xgboost may not be available in some environments
    xgb = None  # type: ignore


BASE_DIR = Path(__file__).resolve().parent
XGBOOST_MODEL_PATH = BASE_DIR / "earthquake_xgb_model.json"
XGBOOST_BASE_MODEL_PATH = BASE_DIR / "earthquake_xgb_base_model.json"
XGBOOST_TRANSFER_MODEL_PATH = BASE_DIR / "earthquake_xgb_transfer_model.json"
SCALER_PATH = BASE_DIR / "earthquake_scaler.pkl"
HISTORICAL_CSV_PATH = BASE_DIR / "earthquake_1995-2023.csv"
DEFAULT_MIN_MAGNITUDE = 4.0
RECENT_LOOKBACK_HOURS = 48

app = Flask(__name__)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class EarthquakePredictor:
    """
    Wrapper around the earthquake prediction logic that originated in the
    multi-modal notebook (a.ipynb). The predictor is responsible for:
        - Loading the trained transfer learning model + scaler (if available)
        - Fetching and caching recent earthquake data from USGS
        - Producing the next magnitude prediction based on recent statistics
    """

    def __init__(self) -> None:
        (
            self.base_model,
            self.transfer_model,
            self.base_model_path,
            self.transfer_model_path,
        ) = self._load_models()
        self.model = self.transfer_model or self.base_model
        self.base_booster = self._extract_booster(self.base_model, self.base_model_path)
        self.transfer_booster = self._extract_booster(self.transfer_model, self.transfer_model_path)
        self.scaler = self._load_scaler()
        self.recent_events: Optional[pd.DataFrame] = None
        self.last_refreshed: Optional[datetime] = None
        self.historical_data: Optional[pd.DataFrame] = None
        self._load_historical_data()

    def _load_models(self) -> Tuple[Optional[object], Optional[object], Optional[Path], Optional[Path]]:
        if xgb is None:
            logger.warning("XGBoost is not available. Falling back to statistical predictions.")
            return None, None, None, None

        base_model, base_path = self._load_single_model(XGBOOST_BASE_MODEL_PATH)
        transfer_model, transfer_path = self._load_single_model(XGBOOST_TRANSFER_MODEL_PATH)

        if transfer_model is None:
            transfer_model, transfer_path = self._load_single_model(XGBOOST_MODEL_PATH)

        if base_model is None:
            base_model, base_path = self._load_single_model(XGBOOST_MODEL_PATH)

        if transfer_model is None and base_model is not None:
            transfer_model = self._duplicate_model(base_model, base_path)
            transfer_path = base_path

        return base_model, transfer_model, base_path, transfer_path

    def _load_single_model(self, path: Path) -> Tuple[Optional[object], Optional[Path]]:
        if xgb is None or not path.exists():
            return None, None

        try:
            if path.suffix.lower() in {".json", ".ubj", ".bin"}:
                try:
                    model = xgb.XGBRegressor()
                    model.load_model(str(path))
                    logger.info("Loaded XGBoost regressor from %s.", path.name)
                    return model, path
                except Exception:
                    booster = xgb.Booster()
                    booster.load_model(str(path))
                    logger.info("Loaded XGBoost booster from %s.", path.name)
                    return booster, path

            model = joblib.load(path)
            if hasattr(model, "predict"):
                logger.info("Loaded XGBoost model via joblib from %s.", path.name)
                return model, path
            logger.warning("Loaded object from %s lacks predict(); ignoring.", path)
        except Exception as exc:  # pragma: no cover - safety net
            logger.warning("Failed to load XGBoost model from %s. %s", path, exc)
        return None, None

    def _duplicate_model(self, model: object, source_path: Optional[Path]) -> Optional[object]:
        if xgb is None or model is None:
            return None
        if isinstance(model, xgb.Booster):
            try:
                return model.copy()
            except Exception:
                pass
        if hasattr(model, "save_model"):
            temp_booster = self._extract_booster(model, source_path)
            if temp_booster is not None:
                return temp_booster.copy()
        if source_path and source_path.exists():
            duplicate, _ = self._load_single_model(source_path)
            return duplicate
        return None

    def _extract_booster(self, model: Optional[object], source_path: Optional[Path]):
        if xgb is None or model is None:
            return None
        if isinstance(model, xgb.Booster):
            return model
        if hasattr(model, "get_booster"):
            try:
                return model.get_booster()
            except Exception:
                logger.debug("Unable to extract booster from model type %s", type(model))
        if source_path and source_path.exists():
            try:
                booster = xgb.Booster()
                booster.load_model(str(source_path))
                return booster
            except Exception as exc:
                logger.debug("Unable to load booster from %s: %s", source_path, exc)
        return None

    def _load_base_booster_copy(self) -> Optional["xgb.Booster"]:
        if xgb is None:
            return None
        if self.base_model_path and self.base_model_path.exists():
            try:
                booster = xgb.Booster()
                booster.load_model(str(self.base_model_path))
                return booster
            except Exception as exc:
                logger.debug("Failed to reload base booster from %s: %s", self.base_model_path, exc)
        if self.base_booster is not None:
            try:
                return self.base_booster.copy()
            except Exception:
                logger.debug("Unable to clone in-memory base booster.")
        return None

    def _fine_tune_from_base(
        self,
        df: pd.DataFrame,
        feature_order: List[str],
        feature_defaults: Dict[str, float],
        minimum_samples: int = 5,
    ) -> "xgb.Booster":
        if xgb is None:
            raise ValueError("XGBoost is not available.")
        if "magnitude" not in df.columns:
            raise ValueError("Transfer learning input missing 'magnitude' column.")

        df_valid = df.dropna(subset=["magnitude"])
        if len(df_valid) < minimum_samples:
            raise ValueError("Not enough samples for fine-tuning.")

        feature_rows: List[List[float]] = []
        labels: List[float] = []

        for _, row in df_valid.iterrows():
            row_values: List[float] = []
            for key in feature_order:
                value = row.get(key)
                if value is None or pd.isna(value):
                    value = feature_defaults.get(key, feature_defaults.get("depth", 10.0))
                row_values.append(float(value))
            feature_rows.append(row_values)
            labels.append(float(row["magnitude"]))

        feature_matrix = np.array(feature_rows, dtype=np.float32)
        if self.scaler is not None:
            feature_matrix = self.scaler.transform(feature_matrix)
        label_array = np.array(labels, dtype=np.float32)

        booster = self._load_base_booster_copy()
        if booster is None:
            raise ValueError("Unable to clone base booster for fine-tuning.")

        params = {
            "objective": "reg:squarederror",
            "eta": 0.02,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "lambda": 1.0,
        }
        rounds = min(50, max(10, len(label_array) * 2))
        dtrain = xgb.DMatrix(feature_matrix, label=label_array, feature_names=feature_order)
        return xgb.train(params, dtrain, num_boost_round=rounds, xgb_model=booster, verbose_eval=False)

    @staticmethod
    def _load_scaler():
        if SCALER_PATH.exists():
            try:
                scaler = joblib.load(SCALER_PATH)
                logger.info("Loaded feature scaler.")
                return scaler
            except Exception as exc:  # pragma: no cover
                logger.warning("Failed to load scaler. Proceeding without scaling. %s", exc)
        return None

    def _load_historical_data(self):
        """Load historical earthquake data from CSV file."""
        if HISTORICAL_CSV_PATH.exists():
            try:
                self.historical_data = pd.read_csv(HISTORICAL_CSV_PATH)
                logger.info(f"Loaded historical data: {len(self.historical_data)} records.")
            except Exception as exc:
                logger.warning("Failed to load historical CSV. Location-based prediction will be limited. %s", exc)
                self.historical_data = None
        else:
            logger.warning("Historical CSV file not found. Location-based prediction will be limited.")
            self.historical_data = None

    @staticmethod
    def _extract_location_keywords(place: str) -> List[str]:
        """
        Extract location keywords (city and country) from place string.
        Example: "101 km ESE of Yamada, Japan" -> ["yamada", "japan"]
        Example: "southeast of Easter Island" -> ["easter island", "easter", "island"]
        """
        if not place or pd.isna(place):
            return []
        
        keywords = []
        # Remove distance/direction prefixes like "101 km ESE of", "southeast of", etc.
        place_clean = place.lower()
        
        # Remove common directional prefixes
        directional_prefixes = [
            "north of", "south of", "east of", "west of",
            "northeast of", "northwest of", "southeast of", "southwest of",
            "nne of", "ene of", "ese of", "sse of", "ssw of", "wsw of", "wnw of", "nnw of",
            "km n of", "km s of", "km e of", "km w of",
            "km ne of", "km nw of", "km se of", "km sw of"
        ]
        
        for prefix in directional_prefixes:
            if place_clean.startswith(prefix):
                place_clean = place_clean[len(prefix):].strip()
                break
            elif f" {prefix} " in place_clean:
                place_clean = place_clean.split(f" {prefix} ", 1)[-1]
                break
        
        # Remove distance prefixes like "101 km", "50 km", etc.
        place_clean = re.sub(r'^\d+\s*km\s+', '', place_clean).strip()
        
        # Split by comma to get city and country
        parts = [p.strip() for p in place_clean.split(",")]
        
        # Add all parts as keywords
        for part in parts:
            if part:
                keywords.append(part.lower())
                # Also add individual words for multi-word locations like "Easter Island"
                words = part.lower().split()
                if len(words) > 1:
                    keywords.extend(words)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for kw in keywords:
            if kw and kw not in seen:
                seen.add(kw)
                unique_keywords.append(kw)
        
        return unique_keywords

    def _filter_historical_by_location(self, location_place: str) -> pd.DataFrame:
        """
        Filter historical earthquake data by matching location keywords.
        Returns filtered DataFrame with earthquakes from the same city/country.
        Handles both 'place' (USGS format) and 'location'/'country'/'continent' (CSV format) columns.
        """
        if self.historical_data is None or self.historical_data.empty:
            return pd.DataFrame()
        
        location_keywords = self._extract_location_keywords(location_place)
        if not location_keywords:
            return pd.DataFrame()
        
        # Determine which column to use for filtering
        location_column = None
        if 'place' in self.historical_data.columns:
            location_column = 'place'
        elif 'location' in self.historical_data.columns:
            location_column = 'location'
        else:
            logger.warning("Historical data does not have 'place' or 'location' column. Trying 'country' and 'continent'.")
            # Try using country and continent columns
            if 'country' in self.historical_data.columns or 'continent' in self.historical_data.columns:
                mask = pd.Series([False] * len(self.historical_data))
                for keyword in location_keywords:
                    if 'country' in self.historical_data.columns:
                        mask |= self.historical_data['country'].astype(str).str.lower().str.contains(keyword, na=False, regex=False)
                    if 'continent' in self.historical_data.columns:
                        mask |= self.historical_data['continent'].astype(str).str.lower().str.contains(keyword, na=False, regex=False)
                
                filtered = self.historical_data[mask].copy()
                if len(filtered) > 0:
                    logger.info(f"Filtered {len(filtered)} historical earthquakes for location: {location_place}")
                return filtered
            else:
                logger.warning("Historical data does not have location-related columns. Cannot filter by location.")
                return pd.DataFrame()
        
        # Filter historical data where location column contains any of the location keywords
        mask = pd.Series([False] * len(self.historical_data))
        for keyword in location_keywords:
            mask |= self.historical_data[location_column].astype(str).str.lower().str.contains(keyword, na=False, regex=False)
        
        # Also try matching with country and continent if available
        if 'country' in self.historical_data.columns:
            for keyword in location_keywords:
                mask |= self.historical_data['country'].astype(str).str.lower().str.contains(keyword, na=False, regex=False)
        
        if 'continent' in self.historical_data.columns:
            for keyword in location_keywords:
                mask |= self.historical_data['continent'].astype(str).str.lower().str.contains(keyword, na=False, regex=False)
        
        filtered = self.historical_data[mask].copy()
        logger.info(f"Filtered {len(filtered)} historical earthquakes for location: {location_place}")
        
        return filtered

    def refresh_recent_events(self) -> pd.DataFrame:
        """Fetch latest earthquakes from USGS and cache them."""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=RECENT_LOOKBACK_HOURS)

        params = {
            "format": "geojson",
            "starttime": start_time.strftime("%Y-%m-%dT%H:%M:%S"),
            "endtime": end_time.strftime("%Y-%m-%dT%H:%M:%S"),
            "minmagnitude": DEFAULT_MIN_MAGNITUDE,
            "orderby": "time",
        }

        try:
            response = requests.get(
                "https://earthquake.usgs.gov/fdsnws/event/1/query", params=params, timeout=10
            )
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            logger.error("Failed to fetch recent earthquake data: %s", exc)
            if self.recent_events is not None:
                return self.recent_events
            return pd.DataFrame()

        events: List[Dict[str, float]] = []
        for feature in data.get("features", []):
            props = feature.get("properties", {})
            geometry = feature.get("geometry", {}) or {}
            coords = geometry.get("coordinates", [None, None, None])

            events.append(
                {
                    "place": props.get("place", "Unknown Location"),
                    "magnitude": props.get("mag"),
                    "depth": coords[2],
                    "time": pd.to_datetime(props.get("time"), unit="ms"),
                    "latitude": coords[1],
                    "longitude": coords[0],
                    "sig": props.get("sig"),
                    "mmi": props.get("mmi"),
                    "cdi": props.get("cdi"),
                    "nst": props.get("nst"),
                    "gap": props.get("gap"),
                    "dmin": props.get("dmin"),
                }
            )

        df = pd.DataFrame(events).dropna(subset=["magnitude", "depth", "time"])
        df = df.sort_values("time", ascending=False).reset_index(drop=True)
        df = df.replace({np.nan: None})

        self.recent_events = df
        self.last_refreshed = datetime.utcnow()
        return df

    def get_recent_events(self) -> pd.DataFrame:
        if self.recent_events is None or self.last_refreshed is None:
            return self.refresh_recent_events()

        if datetime.utcnow() - self.last_refreshed > timedelta(minutes=10):
            return self.refresh_recent_events()
        return self.recent_events

    def predict_next_magnitude(self) -> Dict[str, object]:
        df = self.get_recent_events()
        if df.empty:
            logger.warning("No recent data available for prediction. Returning fallback values.")
            return {
                "previous_events": [],
                "predicted_magnitude": None,
                "predicted_location": "Unavailable",
                "method": "fallback",
            }

        # Get the most recent earthquake location
        most_recent_location = df.iloc[0]["place"] if not df.empty else "Unknown Location"
        
        # Filter historical data by location
        historical_filtered = self._filter_historical_by_location(most_recent_location)
        
        # Use historical filtered data if available, otherwise use recent events
        if not historical_filtered.empty:
            logger.info(f"Using {len(historical_filtered)} historical earthquakes for location-based prediction.")
            prediction_df = historical_filtered
            data_source = "historical_location"
        else:
            logger.info("No historical data found for location. Using recent events.")
            prediction_df = df
            data_source = "recent"

        previous_events = (
            df[["place", "magnitude", "time", "depth"]]
            .head(5)
            .to_dict("records")
        )
        previous_events = self._prepare_for_json(previous_events)
        for event in previous_events:
            if isinstance(event.get("time"), pd.Timestamp):
                event["time"] = event["time"].isoformat()

        # Aggregate features from the prediction dataset (historical filtered or recent)
        features = {}
        for key in ["cdi", "mmi", "sig", "nst", "dmin", "gap", "depth", "latitude", "longitude"]:
            if key in prediction_df.columns:
                features[key] = prediction_df[key].mean() if not prediction_df[key].dropna().empty else None
            else:
                features[key] = None

        feature_defaults = {
            "cdi": 3.0,
            "mmi": 3.0,
            "sig": 400,
            "nst": 40,
            "dmin": 2.0,
            "gap": 45,
            "depth": float(prediction_df["depth"].median()) if "depth" in prediction_df.columns and not prediction_df["depth"].dropna().empty else 10.0,
            "latitude": float(prediction_df["latitude"].median()) if "latitude" in prediction_df.columns and not prediction_df["latitude"].dropna().empty else 0.0,
            "longitude": float(prediction_df["longitude"].median()) if "longitude" in prediction_df.columns and not prediction_df["longitude"].dropna().empty else 0.0,
        }

        for key, default in feature_defaults.items():
            value = features.get(key)
            if value is None or pd.isna(value):
                features[key] = default

        feature_order = ["cdi", "mmi", "sig", "nst", "dmin", "gap", "depth", "latitude", "longitude"]
        feature_array = np.array([[features[key] for key in feature_order]], dtype=np.float32)

        if self.scaler is not None:
            feature_array = self.scaler.transform(feature_array)

        predicted_mag: Optional[float] = None
        method: str = "fallback"
        base_candidate = self.transfer_model or self.base_model

        if data_source == "historical_location" and xgb is not None:
            try:
                tuned_model = self._fine_tune_from_base(historical_filtered, feature_order, feature_defaults)
                predicted_mag = float(self._predict_with_model(feature_array, feature_order, tuned_model))
                method = f"xgboost_transfer_{data_source}"
            except Exception as exc:
                logger.debug("Fine-tuning skipped; using base model. %s", exc)

        if predicted_mag is None and base_candidate is not None and xgb is not None:
            try:
                predicted_mag = float(self._predict_with_model(feature_array, feature_order, base_candidate))
                method = f"xgboost_{data_source}"
            except Exception as exc:
                logger.warning("XGBoost prediction failed; reverting to statistical fallback. %s", exc)
                predicted_mag = None

        if predicted_mag is None:
            predicted_mag, method = self._statistical_fallback(df, historical_filtered)

        precautions = self._precautions_for(predicted_mag)

        return {
            "previous_events": previous_events,
            "predicted_magnitude": round(predicted_mag, 2),
            "predicted_location": most_recent_location,
            "method": method,
            "precaution_level": precautions["level"],
            "precautions": precautions["actions"],
        }

    @staticmethod
    def _precautions_for(magnitude: float) -> Dict[str, object]:
        tiers = [
            {
                "level": "Low Awareness",
                "min": 0.0,
                "max": 4.0,
                "actions": [
                    "Secure loose household items that could fall.",
                    "Keep a basic emergency kit stocked (water, flashlight, batteries).",
                    "Review safe spots indoors, such as under a study table.",
                ],
            },
            {
                "level": "Preparedness",
                "min": 4.0,
                "max": 5.0,
                "actions": [
                    "Move heavy objects to lower shelves and anchor tall furniture.",
                    "Ensure family members know drop-cover-hold-on basics.",
                    "Keep mobile phones charged and flashlights accessible.",
                ],
            },
            {
                "level": "Heightened Alert",
                "min": 5.0,
                "max": 6.0,
                "actions": [
                    "Assemble essential supplies and medications near exit points.",
                    "Identify quick evacuation routes and outdoor meeting points.",
                    "Stay tuned to local emergency broadcasts for updates.",
                ],
            },
            {
                "level": "High Alert",
                "min": 6.0,
                "max": 7.0,
                "actions": [
                    "Review gas, electricity, and water shut-off procedures.",
                    "Secure fragile items; reinforce windows if possible.",
                    "Prepare to evacuate vulnerable family members or pets.",
                ],
            },
            {
                "level": "Maximum Readiness",
                "min": 7.0,
                "max": 11.0,
                "actions": [
                    "Follow all official evacuation orders immediately.",
                    "Shelter in reinforced structures or open safe zones.",
                    "Expect strong aftershocks; keep emergency kits on hand.",
                    "Assist neighbors and check on those needing extra help.",
                ],
            },
        ]

        for tier in tiers:
            if tier["min"] <= magnitude < tier["max"]:
                return {"level": tier["level"], "actions": tier["actions"]}
        # fallback
        return {
            "level": "Preparedness",
            "actions": [
                "Maintain a three-day supply of food, water, and medications.",
                "Know emergency contacts and local shelter locations.",
                "Keep vehicle fuel tank at least half full.",
            ],
        }

    @staticmethod
    def _prepare_for_json(records: List[Dict[str, object]]) -> List[Dict[str, object]]:
        cleaned: List[Dict[str, object]] = []
        for record in records:
            converted: Dict[str, object] = {}
            for key, value in record.items():
                if isinstance(value, (np.floating, float)):
                    if np.isnan(value) or np.isinf(value):
                        converted[key] = None
                    else:
                        converted[key] = float(value)
                else:
                    converted[key] = value
            cleaned.append(converted)
        return cleaned

    def _predict_with_model(
        self,
        feature_array: np.ndarray,
        feature_names: List[str],
        model: Optional[object] = None,
    ) -> float:
        target_model = model or self.transfer_model or self.base_model
        if target_model is None:
            raise ValueError("Model is not loaded.")

        if xgb is None:
            raise ValueError("XGBoost is not available.")

        if isinstance(target_model, xgb.XGBRegressor):
            predictions = target_model.predict(feature_array)
        elif isinstance(target_model, xgb.Booster):
            dmatrix = xgb.DMatrix(feature_array, feature_names=feature_names)
            predictions = target_model.predict(dmatrix)
        elif hasattr(target_model, "predict"):
            predictions = target_model.predict(feature_array)
        else:
            raise TypeError(f"Unsupported model type for prediction: {type(target_model)}")

        if predictions is None or len(predictions) == 0:
            raise ValueError("Model returned no predictions.")

        value = float(np.asarray(predictions).reshape(-1)[0])
        if np.isnan(value) or np.isinf(value):
            raise ValueError("Model prediction is not a finite number.")
        return value

    @staticmethod
    def _statistical_fallback(
        recent_df: pd.DataFrame, historical_filtered: pd.DataFrame
    ) -> Tuple[float, str]:
        if not historical_filtered.empty and "magnitude" in historical_filtered.columns:
            mean_mag = historical_filtered["magnitude"].mean()
            std_mag = historical_filtered["magnitude"].std() if len(historical_filtered) > 1 else 0.2
            predicted_mag = float(np.clip(mean_mag + 0.1 * std_mag, 0, 10))
            method = "statistical_historical_location"
        else:
            mean_mag = recent_df["magnitude"].mean()
            std_mag = recent_df["magnitude"].std() if len(recent_df) > 1 else 0.2
            predicted_mag = float(np.clip(mean_mag + 0.1 * std_mag, 0, 10))
            method = "statistical_recent"
        return predicted_mag, method


predictor = EarthquakePredictor()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict")
def predict():
    result = predictor.predict_next_magnitude()
    return jsonify(
        {
            "previous_events": result["previous_events"],
            "predicted_magnitude": result["predicted_magnitude"],
            "predicted_location": result["predicted_location"],
            "method": result["method"],
            "precaution_level": result["precaution_level"],
            "precautions": result["precautions"],
            "generated_at": datetime.utcnow().isoformat() + "Z",
        }
    )


@app.route("/recent")
def recent():
    df = predictor.get_recent_events()
    events = predictor._prepare_for_json(df.to_dict("records")) if not df.empty else []
    for event in events:
        if isinstance(event.get("time"), pd.Timestamp):
            event["time"] = event["time"].isoformat()
    return jsonify(
        {
            "events": events,
            "last_refreshed": predictor.last_refreshed.isoformat() + "Z"
            if predictor.last_refreshed
            else None,
        }
    )


@app.route("/refresh", methods=["POST"])
def refresh():
    df = predictor.refresh_recent_events()
    events = predictor._prepare_for_json(df.to_dict("records")) if not df.empty else []
    for event in events:
        if isinstance(event.get("time"), pd.Timestamp):
            event["time"] = event["time"].isoformat()
    return jsonify(
        {
            "events": events,
            "last_refreshed": predictor.last_refreshed.isoformat() + "Z"
            if predictor.last_refreshed
            else None,
        }
    )


if __name__ == "__main__":  # pragma: no cover
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

