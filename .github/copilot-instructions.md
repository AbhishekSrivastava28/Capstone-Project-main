# Earthquake Magnitude Prediction System - AI Agent Guide

## Project Overview
This is a **real-time earthquake prediction web application** combining machine learning models with live USGS seismic data. It provides magnitude predictions, location-based historical analysis, and safety precaution guidance.

### Core Architecture
- **ML Backend**: Transfer learning model trained on historical 1995-2023 earthquake data (saves predictions to JSON endpoints)
- **Data Pipeline**: Fetches real-time earthquakes from USGS API (~10 min cache) and filters by location using historical CSV
- **Web Frontend**: Flask + Bootstrap dashboard displaying recent events and predictions
- **Model Strategy**: Base CNN → Transfer Learning fine-tuning (see notebook cells for training)

---

## Key Components & Data Flow

### 1. **Model & Feature Engineering** (`app.py`, `a.ipynb`)
- **Input Features (9 total)**: `cdi`, `mmi`, `sig`, `nst`, `dmin`, `gap`, `depth`, `latitude`, `longitude`
- **Output**: Single magnitude prediction value
- **Scaler**: StandardScaler fitted on training data (must be applied before predictions)
- **Transfer Learning Pattern**: Freeze base model layers, add dense layers on top, use low learning rate (0.0001)
- **Fallback Logic**: If model/scaler unavailable → statistical mean of recent/historical magnitudes

### 2. **Location-Based Filtering** (critical pattern in `EarthquakePredictor` class)
```python
# Extract keywords from USGS place strings: "101 km ESE of Yamada, Japan" → ["yamada", "japan"]
# Filter historical CSV using multi-column matching:
#   - Primary: 'place' column (USGS format)
#   - Secondary: 'country', 'continent' columns (for alternate CSV formats)
# Always prefer historical filtered data over recent for predictions if available
```

### 3. **USGS API Integration** (`refresh_recent_events()`)
- Endpoint: `https://earthquake.usgs.gov/fdsnws/event/1/query`
- Default lookback: **48 hours**, **magnitude ≥ 4.0**
- Response format: GeoJSON with properties: `mag`, `depth`, `cdi`, `mmi`, `sig`, etc.
- Error handling: Returns cached data if API fails (graceful degradation)

### 4. **Web Routes** (Flask)
| Route | Method | Purpose |
|-------|--------|---------|
| `/` | GET | Render dashboard HTML |
| `/predict` | GET | Returns JSON with `predicted_magnitude`, `precaution_level`, `previous_events` |
| `/recent` | GET | Returns cached recent earthquakes + last_refreshed timestamp |
| `/refresh` | POST | Force API fetch and update cache |

---

## Developer Workflows

### **Building & Running**
```bash
# Install dependencies (Flask, TensorFlow, pandas, requests)
pip install -r requirements.txt  # Create if missing

# Run development server (auto-reloads)
python app.py  # Starts on http://localhost:5000

# Run notebook (for model training/exploration)
jupyter notebook a.ipynb
```

### **Testing Model Predictions**
```bash
# Via curl
curl http://localhost:5000/predict | python -m json.tool

# Check recent events
curl http://localhost:5000/recent
```

### **Data Refresh Strategy**
- Frontend: Auto-refresh every 10 minutes via JavaScript interval
- Backend: 10-minute cache to avoid USGS rate limits
- Manual refresh: "Refresh" button in UI triggers POST `/refresh`

---

## Code Patterns & Conventions

### **Missing Data Handling**
- **Feature defaults** in `EarthquakePredictor.predict_next_magnitude()` when columns absent:
  ```python
  feature_defaults = {
      "cdi": 3.0, "mmi": 3.0, "sig": 400, "nst": 40,
      "dmin": 2.0, "gap": 45, "depth": 10.0, ...
  }
  ```
- **Null value replacement**: Use median/mean of available data, fallback to defaults
- **NaN handling in JSON**: `np.nan` and `np.inf` converted to `None` via `_prepare_for_json()`

### **Logging & Error Recovery**
- Logger configured at module level; use `logger.info()`, `logger.warning()`, `logger.error()`
- All external API calls wrapped in try-except with fallback to cached/statistical data
- No hard failures—graceful degradation preferred (e.g., no TensorFlow → statistical predictions)

### **Location Matching Algorithm**
- Removes directional prefixes: "north of", "southeast of", "101 km ESE of", etc.
- Splits by comma; creates keywords from each part (handles multi-word places like "Easter Island")
- Case-insensitive substring matching on historical 'place', 'country', 'continent' columns
- Returns empty DataFrame if no matches (prediction falls back to recent events)

### **Prediction Method String**
- Format: `{model_or_statistical}_{data_source}`
  - Model-based: `model_recent`, `model_historical_location`
  - Statistical: `statistical_recent`, `statistical_historical_location`
- Rendered in frontend badge

### **Precaution Tier System**
- 5 magnitude ranges → 5 precaution levels (Low Awareness → Maximum Readiness)
- Each tier has specific actionable steps (e.g., "Move heavy objects to lower shelves")
- Bounded by magnitude thresholds: `[0–4)`, `[4–5)`, `[5–6)`, `[6–7)`, `[7–11)`

---

## Integration Points & External Dependencies

### **Machine Learning Stack**
- **TensorFlow/Keras**: Model architecture stored in JSON, weights in HDF5
- **Scikit-learn**: StandardScaler for feature normalization
- **Joblib**: Scaler serialization (pickle alternative for sklearn objects)
- **NumPy/Pandas**: Numerical operations, DataFrame manipulation

### **Data Sources**
- **USGS Earthquake API**: Real-time seismic data (public, no auth required)
- **CSV File** (`earthquake_1995-2023.csv`): Historical training data for location filtering
- **Model Artifacts**: 4 files required:
  - `earthquake_transfer_model_architecture.json`
  - `earthquake_transfer_model.weights.h5`
  - `earthquake_base_model_architecture.json` (optional, for reference)
  - `earthquake_scaler.pkl`

### **Frontend Dependencies**
- **Bootstrap 5.3.3**: CSS framework (CDN)
- **Chart.js 4.4.7**: Magnitude trend visualization
- Vanilla JavaScript (no framework)

---

## Common Extension Points

### **Adding New Features**
1. Add column to CSV → Include in feature list `['cdi', 'mmi', ...]`
2. Update `EarthquakePredictor.__init__()` to load column
3. Add default value in `feature_defaults` dict
4. Update model input dimension if retraining

### **Changing Prediction Logic**
- Modify `predict_next_magnitude()` method
- Ensure feature array maintains order: `["cdi", "mmi", "sig", "nst", "dmin", "gap", "depth", "latitude", "longitude"]`
- Test fallback: Set `self.model = None` to verify statistical branch

### **Adjusting Cache Strategy**
- Change `RECENT_LOOKBACK_HOURS` (line ~25 in `app.py`) to fetch different time windows
- Modify `timedelta(minutes=10)` in `get_recent_events()` to adjust cache TTL

### **Frontend Customization**
- Magnitude chart: Modify Chart.js config in `app.js` `renderPrediction()` function
- Precaution UI: Update precautions list HTML generation in same function
- Auto-refresh interval: Change `setInterval(refreshData, 10 * 60 * 1000)` (in ms)

---

## Troubleshooting Guide

| Issue | Diagnosis | Fix |
|-------|-----------|-----|
| "No recent data" in dropdown | USGS API down or rate-limited | Check network; wait 5 min; manual refresh |
| `predicted_magnitude: null` | Model/scaler missing or failed to load | Verify all 4 model artifacts exist and permissions correct |
| NaN values in predictions | Missing columns in historical CSV | Add column or update defaults dict |
| Location filter returns 0 results | Place string parsing failed | Check USGS format; add debug logs to `_extract_location_keywords()` |
| Frontend not updating | Cache not refreshed | Click "Refresh" button; check browser console for fetch errors |

---

## File Structure Reference

```
project/
├── app.py                                    # Flask backend + EarthquakePredictor class
├── a.ipynb                                   # Model training notebook
├── earthquake_1995-2023.csv                  # Historical data
├── earthquake_transfer_model_architecture.json  # Model JSON
├── earthquake_transfer_model.weights.h5      # Model weights
├── earthquake_scaler.pkl                     # Feature scaler
├── templates/
│   └── index.html                            # Dashboard HTML
├── static/
│   ├── js/app.js                             # Client-side logic + auto-refresh
│   └── css/styles.css                        # Dashboard styling
└── .github/
    └── copilot-instructions.md               # This file
```

---

## Key Takeaways for AI Agents

1. **Always preserve graceful fallback paths**: Model → Statistical → Defaults
2. **Location filtering is brittle**: Test with varied USGS place strings; handle missing CSV columns
3. **Feature order matters**: Model expects exact order during prediction
4. **Cache management is implicit**: Frontend auto-refreshes; backend respects 10-min TTL
5. **No database**: All data is ephemeral; CSV is source of truth for historical analysis
