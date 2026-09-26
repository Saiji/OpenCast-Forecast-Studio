import io
import math
import numpy as np
import pandas as pd
import streamlit as st

# --- Page Config ---
st.set_page_config(
    page_title="OpenCast Forecast Studio",
    page_icon="📈",
    layout="wide",
)

# --- Helper Functions & Math ---


def safe_mean(arr):
  return np.mean(arr) if len(arr) > 0 else np.nan


def safe_median(arr):
  if len(arr) == 0:
    return np.nan
  b = np.sort(arr)
  k = len(b) >> 1
  return b[k] if len(b) % 2 else (b[k - 1] + b[k]) / 2


def safe_std(arr):
  return np.std(arr, ddof=1) if len(arr) >= 2 else 0.0


def hampel(y, w=3):
  n = len(y)
  out = y.copy()
  idx = []
  if n < 7:
    return out, idx
  k = max(3, min(w, n // 2))
  for t in range(n):
    lo = max(0, t - k)
    hi = min(n, t + k + 1)
    win = y[lo:hi]
    md = safe_median(win)
    mad = 1.4826 * safe_median(np.abs(win - md))
    if (
        mad > 0
        and abs(y[t] - md) > 3 * mad
        and abs(y[t] - md) > 0.25 * abs(md)
    ):
      out[t] = md
      idx.append(t)
  return out, idx


def classify_demand(y):
  nz = y[y > 0]
  n = len(y)
  adi = n / len(nz) if len(nz) > 0 else np.inf
  cv2 = (safe_std(nz) / safe_mean(nz)) ** 2 if len(nz) > 1 else 0.0
  if len(nz) == 0:
    pattern = "No demand"
  else:
    pattern = (
        "Smooth"
        if adi < 1.32 and cv2 < 0.49
        else "Erratic"
        if adi < 1.32
        else "Intermittent"
        if cv2 < 0.49
        else "Lumpy"
    )
  m = safe_mean(y)
  cv = safe_std(y) / m if m > 0 else np.inf
  return {
      "adi": adi,
      "cv2": cv2,
      "pattern": pattern,
      "cv": cv,
      "zeroShare": 1 - len(nz) / n if n > 0 else 0,
  }


def lin_fit(y):
  n = len(y)
  t_arr = np.arange(n)
  sx, sy = np.sum(t_arr), np.sum(y)
  sxx, sxy = np.sum(t_arr**2), np.sum(t_arr * y)
  denom = n * sxx - sx**2
  b = (n * sxy - sx * sy) / denom if denom != 0 else 0
  a = (sy - b * sx) / n
  return a, b


# --- Model Implementations ---
def fit_ses(y):
  best_sse = np.inf
  best_alpha, best_l, best_fitted = 0.2, y[0], np.full(len(y), np.nan)
  for alpha in np.arange(0.05, 0.96, 0.05):
    l = y[0]
    sse = 0
    fitted = np.zeros(len(y))
    fitted[0] = np.nan
    for t in range(1, len(y)):
      fitted[t] = l
      e = y[t] - l
      sse += e**2
      l += alpha * e
    if sse < best_sse:
      best_sse = sse
      best_alpha = alpha
      best_l = l
      best_fitted = fitted
  return {"l": best_l, "fitted": best_fitted, "sse": best_sse, "alpha": best_alpha}


def fit_holt(y):
  best = None
  n = len(y)
  k = min(n - 1, 3)
  init_tr = (y[k] - y[0]) / k if k > 0 else 0
  for a in [0.1, 0.2, 0.4, 0.6, 0.8]:
    for b in [0.02, 0.05, 0.1, 0.2]:
      for phi in [0.8, 0.9, 0.98]:
        l = y[0]
        tr = init_tr
        sse = 0
        fitted = np.full(n, np.nan)
        for t in range(1, n):
          f = l + phi * tr
          fitted[t] = f
          e = y[t] - f
          sse += e**2
          ln = a * y[t] + (1 - a) * (l + phi * tr)
          tr = b * (ln - l) + (1 - b) * phi * tr
          l = ln
        if best is None or sse < best["sse"]:
          best = {
              "l": l,
              "tr": tr,
              "fitted": fitted,
              "sse": sse,
              "a": a,
              "phi": phi,
          }
  return best


def run_model(id, y, h, m_seas, seasonal):
  n = len(y)
  if id == "naive":
    fitted = np.insert(y[:-1], 0, np.nan)
    fc = np.full(h, y[-1])
    return {"fc": fc, "fitted": fitted}
  elif id == "snaive" and m_seas > 1 and n > m_seas:
    fitted = np.array([y[t - m_seas] if t >= m_seas else np.nan for t in range(n)])
    fc = np.array([y[n - m_seas + (i % m_seas)] for i in range(h)])
    return {"fc": fc, "fitted": fitted}
  elif id == "ma":
    fitted = np.array(
        [
            np.mean(y[t - 3 : t]) if t >= 3 else np.nan
            for t in range(n)
        ]
    )
    val = np.mean(y[-3:])
    return {"fc": np.full(h, val), "fitted": fitted}
  elif id == "ses":
    res = fit_ses(y)
    return {"fc": np.full(h, res["l"]), "fitted": res["fitted"]}
  elif id == "holt":
    res = fit_holt(y)
    fc = [res["l"] + sum(res["phi"] ** (i + 1) for i in range(j + 1)) * res["tr"] for j in range(h)]
    return {"fc": np.array(fc), "fitted": res["fitted"]}
  elif id == "reg":
    a, b = lin_fit(y)
    fitted = np.array([a + b * t for t in range(n)])
    fc = np.array([a + b * (n + i) for i in range(h)])
    return {"fc": fc, "fitted": fitted}
  else:
    # Fallback to Naive
    return run_model("naive", y, h, m_seas, seasonal)


# --- Streamlit UI App ---
st.title("📈 OpenCast Forecast Studio")
st.markdown("Open-source demand forecasting deployed locally via Streamlit.")

# Sidebar Controls
st.sidebar.header("Forecast Settings")
horizon = st.sidebar.number_input("Horizon (periods)", 1, 104, 12)
holdout = st.sidebar.number_input("Backtest holdout", 0, 52, 6)
confidence = st.sidebar.selectbox("Prediction interval", [80, 90, 95], index=1)
metric_choice = st.sidebar.selectbox(
    "Pick best model by", ["WAPE", "RMSE", "MASE", "MAPE"]
)
cleanse_outliers = st.sidebar.checkbox("Cleanse outliers", value=True)
non_negative = st.sidebar.checkbox("No negative forecasts", value=True)

# Data input options
st.subheader("1. Load Data")
data_option = st.radio(
    "Choose data source", ["Use Sample Data", "Upload CSV", "Paste CSV Data"]
)

df = None
if data_option == "Use Sample Data":
  # Generate synthetic sample dataset
  dates = pd.date_range(start="2024-01-01", periods=24, freq="ME")
  np.random.seed(42)
  sample_data = []
  for sku in ["SKU-100", "SKU-200", "SKU-300"]:
    base = np.random.randint(100, 500)
    for d in dates:
      val = max(10, int(base + np.random.normal(0, 50)))
      sample_data.append(
          {"date": d.strftime("%Y-%m-%d"), "item": sku, "quantity": val}
      )
  df = pd.DataFrame(sample_data)
elif data_option == "Upload CSV":
  uploaded_file = st.file_uploader("Upload your CSV file", type=["csv", "txt"])
  if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
  pasted = st.text_area(
      "Paste CSV Data here",
      "date,item,quantity\n2025-01,SKU-100,420\n2025-02,SKU-100,455",
  )
  if pasted:
    try:
      df = pd.read_csv(io.StringIO(pasted))
    except Exception as e:
      st.error(f"Error parsing CSV text: {e}")

if df is not None:
  st.success("Data loaded successfully!")
  st.dataframe(df.head())

  # Standardize column names
  df.columns = [c.lower().strip() for c in df.columns]

  # Identify columns
  date_col = next(
      (c for c in df.columns if c in ["date", "period", "month", "time", "day"]),
      df.columns[0],
  )
  item_col = next(
      (
          c
          for c in df.columns
          if c in ["item", "sku", "product", "material", "part", "series"]
      ),
      df.columns[1] if len(df.columns) > 1 else df.columns[0],
  )
  qty_col = next(
      (
          c
          for c in df.columns
          if c in ["quantity", "qty", "demand", "sales", "units", "value"]
      ),
      df.columns[-1],
  )

  # Process items
  items = df[item_col].unique()
  selected_item = st.selectbox("Select Item for Detailed View", items)

  item_df = df[df[item_col] == selected_item].sort_values(by=date_col)
  y_raw = item_df[qty_col].to_numpy(dtype=float)

  if cleanse_outliers:
    y, _ = hampel(y_raw)
  else:
    y = y_raw

  # Run basic forecasting for the selected item
  models_to_test = ["naive", "snaive", "ma", "ses", "holt", "reg"]
  results = {}
  for m_id in models_to_test:
    if len(y) >= 3:
      res = run_model(m_id, y, horizon, 12, True)
      if non_negative:
        res["fc"] = np.clip(res["fc"], 0, None)
      results[m_id] = res

  st.subheader(f"Forecast for: {selected_item}")

  # Display chart of historical + forecast
  if results:
    best_model_id = list(results.keys())[0]
    fc_values = results[best_model_id]["fc"]

    # Build future dates
    last_date = pd.to_datetime(item_df[date_col].iloc[-1], errors="coerce")
    if pd.isna(last_date):
      future_dates = [f"Period {len(item_df)+i}" for i in range(horizon)]
    else:
      future_dates = [
          (last_date + pd.DateOffset(months=i + 1)).strftime("%Y-%m")
          for i in range(horizon)
      ]

    fc_df = pd.DataFrame(
        {
            "Period": future_dates,
            "Forecast": fc_values,
        }
    )

    st.line_chart(
        pd.DataFrame(
            {"Actual": y},
            index=item_df[date_col].values
            if not pd.isna(last_date)
            else range(len(y)),
        )
    )

    st.markdown("### Future Forecast Values")
    st.dataframe(fc_df)

  # Portfolio ABC-XYZ Summary
  st.subheader("📊 Portfolio Segmentation & Summary")
  summary_rows = []
  for itm in items:
    sub = df[df[item_col] == itm]
    val_arr = sub[qty_col].to_numpy(dtype=float)
    cls = classify_demand(val_arr)
    summary_rows.append(
        {
            "Item": itm,
            "Total Demand": np.sum(val_arr),
            "Pattern": cls["pattern"],
            "CV": round(cls["cv"], 2),
        }
    )

  summary_df = pd.DataFrame(summary_rows)
  st.dataframe(summary_df)
else:
  st.info("Please load or select data to begin forecasting.")