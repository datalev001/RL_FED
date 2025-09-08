import os
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from urllib.parse import quote

# -------- Optional dependencies (graceful detection) --------
try:
    import statsmodels.api as sm
    HAVE_SM = True
except Exception:
    HAVE_SM = False

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    HAVE_SK = True
except Exception:
    HAVE_SK = False

try:
    import yfinance as yf
    HAVE_YF = True
except Exception:
    HAVE_YF = False

try:
    from pandas_datareader import data as pdr
    HAVE_PDR = True
except Exception:
    HAVE_PDR = False

try:
    import requests
    HAVE_REQ = True
except Exception:
    HAVE_REQ = False


# ==========================
# ======= CONFIG AREA ======
# ==========================
USE_REAL_DATA        = True    # Try real downloads; will fallback if unavailable
REQUIRE_REAL_DATA    = False   # If True and real pull fails -> raise error (no fallback)
USE_TE_CPI_SURPRISES = False   # TE US calendar often blocked on free plans; fallback is robust
SAVE_ARTIFACTS       = False   # If True, also save tables/figs into SAVE_DIR

# --- Keys (ENV first, then inline defaults) ---
FRED_KEY_DEFAULT = "####"      # <- your FRED key
TE_API_KEY_DEFAULT = "###"      # <- your TE key (key:secret)


FRED_KEY = os.getenv("FRED_API_KEY", FRED_KEY_DEFAULT)
TE_API_KEY = os.getenv("TE_API_KEY", TE_API_KEY_DEFAULT)

SAVE_DIR = "###"
if SAVE_ARTIFACTS:
    os.makedirs(SAVE_DIR, exist_ok=True)

# For reproducibility of synthetic pieces
NP_SEED_MAIN = 42
NP_SEED_PRICES = 321


# ==========================
# ===== Helper: plotting ===
# ==========================
def show_fig(fig, name=None):
    
    try:
        fig.tight_layout()
    except Exception:
        pass
    plt.show()
    if SAVE_ARTIFACTS and name:
        path = os.path.join(SAVE_DIR, name)
        try:
            fig.savefig(path, dpi=120)
            print(f"[Saved] {path}")
        except Exception as e:
            print(f"[WARN] Failed to save figure {name}: {e}")


# ==========================
# ===== Data download  =====
# ==========================
def fetch_fred_series(series_ids, start="2010-01-01", end=None):
    """Download FRED series using pandas_datareader (if available)."""
    if not HAVE_PDR:
        print("[WARN] pandas_datareader is not installed.")
        return None
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")
    frames = []
    for sid in series_ids:
        try:
            if FRED_KEY:
                df = pdr.DataReader(sid, "fred", start, end, api_key=FRED_KEY)
            else:
                df = pdr.DataReader(sid, "fred", start, end)
            df.columns = [sid]
            frames.append(df)
        except Exception as e:
            print(f"[WARN] FRED fetch failed for {sid}: {e}")
            return None
    if not frames:
        return None
    out = pd.concat(frames, axis=1)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    try:
        if out.index.tz is not None:
            out.index = out.index.tz_localize(None)
    except Exception:
        pass
    return out


def fetch_prices(tickers=("SPY", "QQQ"), start="2010-01-01", end=None):
    """Download adjusted close prices via yfinance (if available)."""
    if not HAVE_YF:
        print("[WARN] yfinance is not installed.")
        return None
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")
    try:
        data = yf.download(list(tickers), start=start, end=end, auto_adjust=True, progress=False)
        if data is None or data.empty:
            return None
        if isinstance(data.columns, pd.MultiIndex):
            px = data["Close"].copy()
        else:
            px = data[["Close"]].copy()
            px.columns = [tickers[0]]
        px = px[~px.index.duplicated(keep="last")].sort_index()
        try:
            if px.index.tz is not None:
                px.index = px.index.tz_localize(None)
        except Exception:
            pass
        return px
    except Exception as e:
        print(f"[WARN] yfinance download failed: {e}")
        return None


# ==========================
# ==== Synthetic fallback ==
# ==========================
def synthesize_dataset():
    """Synthetic data for pipeline validation only (not for real conclusions)."""
    rng = np.random.default_rng(NP_SEED_MAIN)
    start = pd.Timestamp("2014-01-01")
    end   = pd.Timestamp(datetime.today().strftime("%Y-%m-%d"))
    dates_d = pd.bdate_range(start, end, freq="C")
    dates_m = pd.date_range(start, end, freq="M")

    # Monthly macro proxies
    m = len(dates_m)
    cpi = pd.Series(100 + np.cumsum(0.18 + 0.45*np.sin(2*np.pi*np.arange(m)/12) + rng.normal(0,0.2,m)), index=dates_m)
    unemp = pd.Series(5.6 + 0.25*np.cos(2*np.pi*np.arange(m)/12) + rng.normal(0,0.12, m), index=dates_m).clip(3.4, 10.5)
    vix = pd.Series(15 + 6*(np.sin(2*np.pi*np.arange(m)/11)>0) + 5* rng.random(m), index=dates_m)
    dgs2 = pd.Series(1.4 + 0.6*np.sin(2*np.pi*np.arange(m)/17) + rng.normal(0,0.1,m), index=dates_m)
    dgs10= pd.Series(2.0 + 0.5*np.sin(2*np.pi*np.arange(m)/13) + rng.normal(0,0.1,m), index=dates_m)

    macro_m = pd.concat({"CPILFESL": cpi, "UNRATE": unemp}, axis=1)
    macro_d = pd.concat({"VIXCLS": vix, "DGS2": dgs2, "DGS10": dgs10}, axis=1)

    vix_d = vix.reindex(dates_d).ffill().fillna(vix.iloc[0])
    spy = np.empty(len(dates_d)); spy[0] = 250.0
    rng_p = np.random.default_rng(NP_SEED_PRICES)
    for i in range(1, len(dates_d)):
        vt = float(vix_d.iloc[i])
        mu = 0.0002 - 0.00001*(vt - 20)
        sigma = max(1e-4, 0.008 + 0.002*(vt - 15)/10)
        r = rng_p.normal(mu, sigma)
        spy[i] = spy[i-1] * (1 + r)
    prices = pd.DataFrame({"SPY": spy, "QQQ": spy*1.02}, index=dates_d)
    return prices, macro_m, macro_d


# ==========================
# == TradingEconomics CPI ==
# ==========================
def fetch_te_cpi_surprises(country="United States", start="2010-01-01"):
    """
    Fetch CPI surprise z-scores from TradingEconomics calendar.
    Surprise = (Actual - Forecast) / std(Actual - Forecast)
    Auto-fallback to None on any permission/network error.
    """
    if not (USE_TE_CPI_SURPRISES and HAVE_REQ and TE_API_KEY):
        return None

    api = quote(TE_API_KEY, safe="")
    country_q = quote(country, safe="")
    url = f"https://api.tradingeconomics.com/calendar/country/{country_q}?c={api}&format=json"

    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        js = r.json()
    except Exception as e:
        print(f"[INFO] TE unavailable ({e}); falling back to CPI surprise fallback.")
        return None

    rows = []
    for x in js:
        if "CPI" in str(x.get("Category","")).upper():
            dt = pd.to_datetime(x.get("Date", None))
            if dt is None or str(dt.date()) < start:
                continue
            actual = pd.to_numeric(x.get("Actual"), errors="coerce")
            forecast = pd.to_numeric(x.get("Forecast"), errors="coerce")
            if pd.notna(actual) and pd.notna(forecast):
                rows.append({"ts": dt.normalize(), "actual": actual, "cons": forecast})
    df = pd.DataFrame(rows).dropna().drop_duplicates(subset=["ts"]).sort_values("ts")
    if df.empty:
        return None
    sigma = (df["actual"] - df["cons"]).std(ddof=1)
    if pd.isna(sigma) or sigma == 0:
        return None
    df["CPI_surp"] = (df["actual"] - df["cons"]) / sigma
    # add month_key for robust merge
    df["month_key"] = df["ts"].dt.to_period("M")
    return df[["month_key", "CPI_surp"]]


# ==========================
# ======= FOMC dates =======
# ==========================
FOMC_DATES = pd.to_datetime([
    # 2023
    "2023-02-01","2023-03-22","2023-05-03","2023-06-14","2023-07-26","2023-09-20","2023-11-01","2023-12-13",
    # 2024
    "2024-01-31","2024-03-20","2024-05-01","2024-06-12","2024-07-31",
    # 2025 (placeholders)
    "2025-01-29","2025-03-19","2025-05-07","2025-06-18","2025-07-30"
])

def build_fomc_shock_from_dgs2(dgs2_daily):
    """Approximate policy shock with 2y yield daily change on meeting dates; tz-naive index."""
    if dgs2_daily is None or dgs2_daily.dropna().empty:
        return None
    d = dgs2_daily.dropna().copy()
    try:
        if d.index.tz is not None:
            d.index = d.index.tz_localize(None)
    except Exception:
        pass
    ret = d.diff()
    shocks = ret.reindex(FOMC_DATES.tz_localize(None), method="nearest").dropna()
    if shocks.empty:
        return None
    out = pd.DataFrame({"fomc_shock_dgs2": shocks})
    try:
        if out.index.tz is not None:
            out.index = out.index.tz_localize(None)
    except Exception:
        pass
    return out


# ==========================
# ====== Event panel =======
# ==========================
def make_event_panel(macro_m, macro_d, prices_d, prefer_real_cpi_surprise=True):
    """
    Build event+state panel:
      - Events: CPI release timestamps (TE) or monthly CPI timestamps (fallback)
      - Align to next trading day
      - Compute +1/+3/+5 day cumulative returns
      - CPI surprise: TE (preferred) or AR(1)/rolling innovation z-score (fallback)
      - State context: core_cpi_yoy, unemp, vix (merged by month key)
      - Optional: FOMC shock proxy (2y daily change on FOMC)
    """
    prices = prices_d.sort_index().copy()
    try:
        if prices.index.tz is not None:
            prices.index = prices.index.tz_localize(None)
    except Exception:
        pass
    pr = prices["SPY"]
    ret = pr.pct_change()

    # 1) Event timestamps (per month)
    te = fetch_te_cpi_surprises("United States", start="2010-01-01") if prefer_real_cpi_surprise else None
    if te is not None and not te.empty:
        # TE already has month_key
        month_idx = te["month_key"]
        month_key_series = month_idx
    else:
        # fallback: use CPI monthly index
        idx = pd.to_datetime(macro_m.index)
        month_key_series = idx.to_period("M")

    # 2) Build base event rows (align to first trading day ON/AFTER each month)
    events = []
    for mk in month_key_series.unique():
        # choose a representative timestamp for the month (1st day)
        ts = pd.Period(mk, "M").to_timestamp()
        pos_arr = prices.index.get_indexer([ts], method="backfill")
        if len(pos_arr) == 0 or pos_arr[0] == -1:
            continue
        d0 = prices.index[pos_arr[0]]
        seg = ret.loc[d0:]
        r1 = seg.iloc[1:2].sum() if len(seg) > 1 else np.nan
        r3 = seg.iloc[1:4].sum() if len(seg) > 3 else np.nan
        r5 = seg.iloc[1:6].sum() if len(seg) > 5 else np.nan
        events.append({
            "date": d0,
            "event_month": ts.normalize(),
            "month_key": mk,
            "ret_h1": float(r1) if pd.notna(r1) else np.nan,
            "ret_h3": float(r3) if pd.notna(r3) else np.nan,
            "ret_h5": float(r5) if pd.notna(r5) else np.nan,
        })
    panel = pd.DataFrame(events).dropna(subset=["ret_h1","ret_h3","ret_h5"], how="any")
    if panel.empty:
        return panel

    # 3) CPI surprise
    if te is not None and not te.empty:
        panel = panel.merge(te, on="month_key", how="left")
        panel.rename(columns={"CPI_surp": "CPI_surp"}, inplace=True)

    if ("CPI_surp" not in panel.columns) or panel["CPI_surp"].isna().all():
        # compute from CPILFESL even without statsmodels
        if ("CPILFESL" in macro_m.columns) and not macro_m["CPILFESL"].dropna().empty:
            x = macro_m["CPILFESL"].dropna().copy()
            x_month = pd.DataFrame({"CPILFESL": x})
            x_month["month_key"] = x_month.index.to_period("M")
            # AR(1) if available; else rolling mean fallback
            use_ar1 = HAVE_SM and (len(x) >= 24)
            if use_ar1:
                try:
                    x_lag = x.shift(1).dropna(); y = x.loc[x_lag.index]
                    X = sm.add_constant(x_lag.values)
                    mfit = sm.OLS(y.values, X, missing='drop').fit()
                    yhat = mfit.predict(sm.add_constant(x.shift(1))).reindex(x.index)
                    innov = x - yhat
                except Exception:
                    use_ar1 = False
            if not use_ar1:
                innov = x - x.rolling(12, min_periods=6).mean()
            z = (innov - innov.mean())/innov.std(ddof=1)
            z = pd.DataFrame({"CPI_surp": z})
            z["month_key"] = z.index.to_period("M")
            panel = panel.merge(z[["month_key","CPI_surp"]], on="month_key", how="left")

    # 4) State context by month_key
    if "CPILFESL" in macro_m.columns:
        cpi_yoy = macro_m["CPILFESL"].pct_change(12)
        df_cpi = pd.DataFrame({"core_cpi_yoy": cpi_yoy})
        df_cpi["month_key"] = df_cpi.index.to_period("M")
        panel = panel.merge(df_cpi[["month_key","core_cpi_yoy"]], on="month_key", how="left")

    if "UNRATE" in macro_m.columns:
        df_un = pd.DataFrame({"unemp": macro_m["UNRATE"]})
        df_un["month_key"] = df_un.index.to_period("M")
        panel = panel.merge(df_un[["month_key","unemp"]], on="month_key", how="left")

    if "VIXCLS" in macro_d.columns:
        vmap = macro_d["VIXCLS"].dropna().copy()
        try:
            if vmap.index.tz is not None:
                vmap.index = vmap.index.tz_localize(None)
        except Exception:
            pass
        vix_m = vmap.resample("M").last()
        df_vix = pd.DataFrame({"vix": vix_m})
        df_vix["month_key"] = df_vix.index.to_period("M")
        panel = panel.merge(df_vix[["month_key","vix"]], on="month_key", how="left")

    # Drop rows missing any state/surprise, keep one per month_key
    panel = panel.dropna(subset=["CPI_surp","core_cpi_yoy","unemp","vix"], how="any")
    panel = panel.sort_values("event_month").drop_duplicates(subset=["month_key"], keep="last")

    # 5) Optional: FOMC shock proxy (use .map; index is date)
    if "DGS2" in macro_d.columns and not macro_d["DGS2"].dropna().empty and not panel.empty:
        shocks = build_fomc_shock_from_dgs2(macro_d["DGS2"])
        if shocks is not None and not shocks.empty:
            try:
                if shocks.index.tz is not None:
                    shocks.index = shocks.index.tz_localize(None)
            except Exception:
                pass
            panel["fomc_shock_dgs2"] = panel["date"].map(shocks["fomc_shock_dgs2"])

    return panel


# ==========================
# === Local Projections  ===
# ==========================
def local_projection(panel_df, horizon=1, shock_col="CPI_surp",
                     state_cols=("core_cpi_yoy","unemp","vix"), regime_q=0.7):
    """
    OLS local projection with a regime interaction and HAC SEs.
    r_{t+h} = alpha + beta*Shock_t + gamma' State_t + delta' (Shock x HighRegime) + eps
    """
    if not HAVE_SM:
        print("[WARN] statsmodels is not installed; skipping local projections.")
        return None
    use_cols = [f"ret_h{horizon}", shock_col] + list(state_cols)
    df = panel_df.dropna(subset=use_cols).copy()
    if df.empty:
        print(f"[WARN] Local projection h=+{horizon}: not enough data.")
        return None
    y = df[f"ret_h{horizon}"].values
    X = df[[shock_col]+list(state_cols)].copy()
    if "core_cpi_yoy" not in df.columns:
        print("[WARN] Missing core_cpi_yoy; skipping local projection.")
        return None
    hi = (df["core_cpi_yoy"] > df["core_cpi_yoy"].quantile(regime_q)).astype(int)
    X["surp_x_hiCPI"] = df[shock_col] * hi
    X = sm.add_constant(X, has_constant='add')
    try:
        model = sm.OLS(y, X, missing='drop').fit(cov_type="HAC", cov_kwds={"maxlags":2})
    except Exception as e:
        print(f"[WARN] OLS failed (h=+{horizon}): {e}")
        return None
    return model


# ==========================
# === Rolling Graph/DAG  ===
# ==========================
def rolling_partial_corr(df, window=36):
    """Return dict of date -> partial correlation matrix (precision-normalized)."""
    out = {}
    if df is None or df.dropna().empty:
        return out
    cols = df.columns.tolist()
    for i in range(window, len(df)):
        sub = df.iloc[i-window:i].copy()
        sub = sub.dropna()
        if len(sub) < window//2:
            continue
        sub = (sub - sub.mean())/sub.std(ddof=1)
        cov = np.cov(sub.values.T)
        try:
            prec = np.linalg.pinv(cov)
        except Exception:
            continue
        D = np.sqrt(np.diag(prec))
        with np.errstate(divide='ignore', invalid='ignore'):
            P = -prec / (D[:,None]*D[None,:])
        np.fill_diagonal(P, 1.0)
        out[df.index[i]] = pd.DataFrame(P, index=cols, columns=cols)
    return out


# ==========================
# === IRL-like (logit)  ====
# ==========================
def irl_like_logistic(panel_df):
    """
    Proxy 'preference' learning: logistic regression for +1d up-move (or +3d median split).
    Returns coef Series or None.
    """
    if not HAVE_SK:
        print("[WARN] scikit-learn not installed; skipping IRL-like step.")
        return None
    dfc = panel_df.copy()
    Xcols = [c for c in ["CPI_surp","core_cpi_yoy","unemp","vix","fomc_shock_dgs2"] if c in dfc.columns]
    if "ret_h1" not in dfc.columns:
        print("[WARN] Missing ret_h1; skipping IRL-like step.")
        return None
    dfc = dfc.dropna(subset=Xcols + ["ret_h1"])
    if dfc.empty or len(Xcols) == 0:
        print("[INFO] IRL-like skipped (not enough data/features).")
        return None
    y = (dfc["ret_h1"] > 0).astype(int).values
    if len(np.unique(y)) < 2:
        if "ret_h3" not in dfc.columns:
            print("[INFO] IRL-like skipped (no label variability).")
            return None
        y = (dfc["ret_h3"] > dfc["ret_h3"].median()).astype(int).values
        if len(np.unique(y)) < 2:
            print("[INFO] IRL-like skipped (no label variability after fallback).")
            return None
    X = dfc[Xcols].values
    try:
        clf = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=500, solver="lbfgs"))])
        clf.fit(X, y)
    except Exception as e:
        print(f"[WARN] Logistic regression failed: {e}")
        return None
    coefs = pd.Series(clf.named_steps["lr"].coef_.ravel(), index=Xcols).sort_values(key=np.abs, ascending=False)
    return coefs


# ==========================
# ===== RL: Q-learning =====
# ==========================
def qlearning_backtest(prices, macro_m, macro_d):
    """Simple 3-action Q-learning using macro bins as state; compare to Buy&Hold."""
    feat = pd.DataFrame(index=prices.index)
    feat["ret"] = prices["SPY"].pct_change()
    if "VIXCLS" in macro_d.columns:
        feat["vix"] = macro_d["VIXCLS"].reindex(feat.index, method="ffill")
    if "UNRATE" in macro_m.columns:
        feat["unemp"] = macro_m["UNRATE"].reindex(feat.index, method="ffill")
    if "CPILFESL" in macro_m.columns:
        feat["core_cpi_yoy"] = macro_m["CPILFESL"].pct_change(12).reindex(feat.index, method="ffill")
    feat["unemp_delta"] = feat["unemp"].diff(21)
    feat = feat.dropna()
    if feat.empty:
        print("[WARN] RL skipped (not enough features).")
        return ({}, {}, pd.Series(dtype=int), pd.DataFrame())

    def bin_series(s, qs=(0.33, 0.66)):
        q1, q2 = s.quantile(qs[0]), s.quantile(qs[1])
        return pd.Series(np.where(s<=q1,0, np.where(s<=q2,1,2)), index=s.index)

    state_df = pd.DataFrame(index=feat.index)
    state_df["cpi_bin"]   = bin_series(feat["core_cpi_yoy"])
    state_df["unemp_bin"] = bin_series(feat["unemp_delta"])
    state_df["vix_bin"]   = bin_series(feat["vix"])
    state_df["sid"] = state_df["cpi_bin"]*9 + state_df["unemp_bin"]*3 + state_df["vix_bin"]  # 27 states

    action_expo = np.array([0.4, 1.0, 1.2])  # risk-off, neutral, risk-on
    n_states, n_actions = 27, 3
    Q = np.zeros((n_states, n_actions))
    alpha, gamma, eps0 = 0.1, 0.9, 0.2

    rets, dates = feat["ret"], feat.index
    policy_actions, policy_daily_ret, bh_daily_ret = [], [], []

    for t in range(1, len(dates)-1):
        d = dates[t]; d_next = dates[t+1]
        s = int(state_df.loc[d, "sid"])
        eps = eps0 * (1 - t/len(dates))
        a = np.random.randint(n_actions) if (np.random.random() < eps) else int(np.argmax(Q[s]))
        r_next = float(rets.loc[d_next]) * action_expo[a] - 0.00005*(action_expo[a]**2)
        policy_daily_ret.append(r_next); policy_actions.append(a); bh_daily_ret.append(float(rets.loc[d_next]))
        s_next = int(state_df.loc[d_next, "sid"])
        td_target = r_next + gamma * np.max(Q[s_next])
        Q[s, a] = (1 - alpha)*Q[s, a] + alpha * td_target

    policy_daily_ret = pd.Series(policy_daily_ret, index=dates[1:len(policy_daily_ret)+1])
    bh_daily_ret     = pd.Series(bh_daily_ret,     index=dates[1:len(bh_daily_ret)+1])
    policy_actions   = pd.Series(policy_actions,   index=policy_daily_ret.index)

    def perf_stats(r):
        r = r.dropna()
        if len(r) == 0:
            return {"ann_ret":np.nan,"ann_vol":np.nan,"sharpe":np.nan,"mdd":np.nan}
        cum = (1+r).cumprod()
        ann = cum.iloc[-1]**(252/len(r)) - 1
        vol = r.std()*np.sqrt(252)
        sharpe = ann/vol if vol>0 else np.nan
        peak = cum.cummax()
        mdd = (cum/peak - 1).min()
        return {"ann_ret":ann, "ann_vol":vol, "sharpe":sharpe, "mdd":mdd}

    pstats = perf_stats(policy_daily_ret); bstats = perf_stats(bh_daily_ret)

    fig = plt.figure(figsize=(8,4))
    (1 + policy_daily_ret).cumprod().plot(label="RL Policy")
    (1 + bh_daily_ret).cumprod().plot(label="Buy&Hold SPY", alpha=0.85)
    plt.title("Cumulative Return")
    plt.legend()
    show_fig(fig, "rl_cum_return.png" if SAVE_ARTIFACTS else None)

    print("=== RL Policy Performance ===")
    print({k: round(v,4) for k,v in pstats.items()})
    print("=== Buy&Hold SPY Performance ===")
    print({k: round(v,4) for k,v in bstats.items()})

    return pstats, bstats, policy_actions, state_df


# ==========================
# ========= Main ===========
# ==========================
def main():
    # ---- Pull data ----
    prices = macro_all = None
    if USE_REAL_DATA:
        fred_ids = ["CPILFESL","UNRATE","VIXCLS","DGS2","DGS10","PCEPI","FEDFUNDS"]
        macro_all = fetch_fred_series(fred_ids, start="2010-01-01")
        prices    = fetch_prices(("SPY","QQQ"), start="2010-01-01")

    need_fallback = (not USE_REAL_DATA) or (macro_all is None) or (prices is None) \
                    or (macro_all.dropna().empty if macro_all is not None else True) \
                    or (prices.dropna().empty if prices is not None else True)

    if REQUIRE_REAL_DATA and need_fallback:
        raise RuntimeError("Failed to download real data. Check internet and API keys (FRED_API_KEY, TE_API_KEY).")

    if need_fallback:
        prices, macro_m, macro_d = synthesize_dataset()
        print("[INFO] Using synthetic dataset fallback.")
    else:
        # Split macro into monthly (CPILFESL, UNRATE) and daily (VIXCLS, DGS2, DGS10, FEDFUNDS)
        macro_m = pd.DataFrame(index=macro_all.index)
        if "CPILFESL" in macro_all.columns: macro_m["CPILFESL"] = macro_all["CPILFESL"].dropna()
        if "UNRATE"   in macro_all.columns: macro_m["UNRATE"]   = macro_all["UNRATE"].dropna()
        macro_m = macro_m[~macro_m.index.duplicated(keep="last")].sort_index()

        macro_d = pd.DataFrame(index=macro_all.index)
        for c in ["VIXCLS","DGS2","DGS10","FEDFUNDS"]:
            if c in macro_all.columns:
                macro_d[c] = macro_all[c]
        macro_d = macro_d[~macro_d.index.duplicated(keep="last")].sort_index()

    # ---- Quick visuals ----
    fig = plt.figure(figsize=(9,4))
    prices["SPY"].dropna().plot(label="SPY")
    if "QQQ" in prices.columns:
        prices["QQQ"].dropna().plot(label="QQQ", alpha=0.8)
    plt.title("Prices")
    plt.legend()
    show_fig(fig, "prices.png" if SAVE_ARTIFACTS else None)

    if "CPILFESL" in macro_m.columns:
        fig = plt.figure(figsize=(9,3))
        (macro_m["CPILFESL"].pct_change(12)*100).dropna().plot()
        plt.title("Core CPI YoY (%)")
        show_fig(fig, "core_cpi_yoy.png" if SAVE_ARTIFACTS else None)

    if "VIXCLS" in macro_d.columns:
        fig = plt.figure(figsize=(9,3))
        macro_d["VIXCLS"].dropna().plot()
        plt.title("VIX (daily)")
        show_fig(fig, "vix.png" if SAVE_ARTIFACTS else None)

    if "DGS2" in macro_d.columns and "DGS10" in macro_d.columns:
        fig = plt.figure(figsize=(9,3))
        macro_d[["DGS2","DGS10"]].dropna().plot()
        plt.title("Treasury Yields (2y, 10y)")
        show_fig(fig, "yields.png" if SAVE_ARTIFACTS else None)

    # ---- Build event panel ----
    panel = make_event_panel(macro_m, macro_d, prices, prefer_real_cpi_surprise=USE_TE_CPI_SURPRISES)
    print("[INFO] Event panel head:\n", panel.head())
    print(f"[INFO] Panel rows: {len(panel)}")

    # ---- Local Projections (h=1,3,5) ----
    if "CPI_surp" in panel.columns and HAVE_SM and not panel.dropna().empty:
        for h in (1,3,5):
            model = local_projection(panel, h, "CPI_surp", ("core_cpi_yoy","unemp","vix"))
            if model is not None:
                print(f"\n[Local Projection] Horizon +{h}: N={int(model.nobs)} | adj R^2={model.rsquared_adj:.3f}")
                coef_tab = pd.DataFrame({"beta": model.params, "pval": model.pvalues}).round(4)
                print(coef_tab)

        fig = plt.figure(figsize=(6,4))
        plt.scatter(panel["CPI_surp"], panel["ret_h1"], s=18)
        plt.axhline(0, ls="--", lw=0.8, c="k"); plt.axvline(0, ls="--", lw=0.8, c="k")
        plt.xlabel("CPI Surprise (z)")
        plt.ylabel("+1d SPY Return")
        plt.title("CPI Surprise vs +1d Return")
        show_fig(fig, "scatter_cpi_surp_ret1.png" if SAVE_ARTIFACTS else None)
    else:
        print("[INFO] Skipping local projections (missing CPI_surp or statsmodels).")

    # ---- Graph view (rolling partial correlations on monthly features) ----
    feat_m = pd.DataFrame(index=macro_m.index)
    if "CPILFESL" in macro_m.columns: feat_m["core_cpi_yoy"] = macro_m["CPILFESL"].pct_change(12)
    if "UNRATE"   in macro_m.columns: feat_m["unemp"] = macro_m["UNRATE"]
    if "VIXCLS"   in macro_d.columns: feat_m = feat_m.join(macro_d["VIXCLS"].resample("M").last().rename("vix"), how="left")
    if "DGS2"     in macro_d.columns: feat_m = feat_m.join(macro_d["DGS2"].resample("M").last().rename("dgs2"), how="left")
    if "DGS10"    in macro_d.columns: feat_m = feat_m.join(macro_d["DGS10"].resample("M").last().rename("dgs10"), how="left")
    feat_m = feat_m.dropna()

    if len(feat_m) > 20:
        win = min(36, len(feat_m)) if len(feat_m) > 36 else max(12, len(feat_m)//2)
        graphs = rolling_partial_corr(feat_m, window=win)
        if graphs:
            last_dt = sorted(graphs.keys())[-1]
            G = graphs[last_dt].copy()
            np.fill_diagonal(G.values, 0.0)
            centrality = G.abs().sum(axis=1).sort_values(ascending=False)
            print("\n[Graph centrality | latest window]")
            print(centrality.round(3))

            fig = plt.figure(figsize=(7,4))
            centrality.plot(kind="bar")
            plt.title("Graph Centrality (abs edge strength) - latest window")
            plt.ylabel("sum(|partial corr|)")
            show_fig(fig, "graph_centrality.png" if SAVE_ARTIFACTS else None)

    # ---- IRL-like logistic (preference weights) ----
    coefs = irl_like_logistic(panel)
    if coefs is not None:
        print("\n[IRL-like preference weights] (logistic coefficients)")
        print(coefs.round(3))

        fig = plt.figure(figsize=(6,3))
        coefs.sort_values().plot(kind="bar")
        plt.title("IRL-like Coefficients")
        show_fig(fig, "irl_like_coefs.png" if SAVE_ARTIFACTS else None)

    # ---- RL backtest ----
    pstats, bstats, policy_actions, state_df = qlearning_backtest(prices, macro_m, macro_d)

    print("\n===== SUMMARY =====")
    print(f"Data source: {'REAL' if not need_fallback else 'SYNTHETIC'}")
    if not panel.empty:
        print(f"Panel rows: {len(panel)} | Date span: {panel['date'].min().date()} → {panel['date'].max().date()}")
    else:
        print("Panel rows: 0 | Date span: N/A")
    print("Outputs: charts shown in Plots pane; details printed above.")


if __name__ == "__main__":
    main()
