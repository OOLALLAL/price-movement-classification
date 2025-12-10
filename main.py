import numpy as np
import yfinance as yf
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


# ---------------------------------------------
# 1. Data Download
# ---------------------------------------------
def download_btc(start="2000-01-01"):
    btc = yf.download("BTC-USD", start=start)
    close = btc["Close"]
    high = btc["High"]
    low = btc["Low"]
    volume = btc["Volume"]
    return close, high, low, volume


# ---------------------------------------------
# 2. Utility Functions
# ---------------------------------------------
def to_log(close: pd.Series) -> pd.Series:
    """Compute log returns from price series."""
    return np.log(close / close.shift(1)).dropna()


def compute_down_streak(window: pd.Series) -> int:
    """Maximum consecutive negative returns in the window."""
    cnt = 0
    max_cnt = 0
    for r in window.values:
        if r < 0:
            cnt += 1
        else:
            max_cnt = max(max_cnt, cnt)
            cnt = 0
    return max_cnt


def compute_up_streak(window: pd.Series) -> int:
    """Maximum consecutive positive returns in the window."""
    cnt = 0
    max_cnt = 0
    for r in window.values:
        if r > 0:
            cnt += 1
        else:
            max_cnt = max(max_cnt, cnt)
            cnt = 0
    return max_cnt


def max_drawdown(ret_window: pd.Series) -> float:
    """Compute the maximum drawdown of cumulative returns."""
    cum = ret_window.cumsum()
    peak = cum.cummax()
    dd = cum - peak
    return dd.min()


def compute_ATR(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window).mean()
    return atr


# ---------------------------------------------
# 3. Feature Engineering
# ---------------------------------------------
def build_feature_dataset(close: pd.Series,
                          high: pd.Series,
                          low: pd.Series,
                          volume: pd.Series,
                          N: int,
                          K: int,
                          atr_window: int):
    """
    Build supervised-learning dataset of engineered features
    from rolling BTC windows.
    """
    rets = to_log(close)

    close = close.reindex(rets.index)
    high = high.reindex(rets.index)
    low = low.reindex(rets.index)
    volume = volume.reindex(rets.index)

    atr = compute_ATR(high, low, close, window=atr_window)
    atr = atr.reindex(rets.index)

    ma7 = close.rolling(7).mean()
    ma7 = ma7.reindex(rets.index)

    feature_list = []
    label_list = []
    index_list = []

    for i in range(N, len(rets) - K):
        past_window = rets.iloc[i-N:i]
        future_window = rets.iloc[i:i+K]
        volume_window = volume.iloc[i-N:i]

        feat = {}

        # Return-based indicators
        feat["ret_3d"] = past_window[-3:].sum()
        feat["ret_7d"] = past_window[-7:].sum()
        feat["ret_14d"] = past_window[-14:].sum()
        feat["ret_21d"] = past_window[-21:].sum()
        feat["ret_N"] = past_window.sum()

        feat["down_ratio"] = (past_window < 0).mean()
        feat["up_ratio"] = (past_window > 0).mean()

        feat["down_streak"] = compute_down_streak(past_window)
        feat["up_streak"] = compute_up_streak(past_window)

        # Volatility indicators
        feat["vol_7"] = past_window[-7:].std()
        feat["vol_30"] = past_window[-30:].std()
        feat["vol_of_vol"] = past_window.rolling(7).std().std()

        # ATR
        feat["atr"] = atr.iloc[i-1]

        # Volume-based indicators
        last_volume = volume_window.iloc[-1]
        vol_7ago = volume_window.iloc[-7]
        feat["vol_change"] = (last_volume - volume_window.mean()) / volume_window.mean()
        feat["vol_mom"] = np.log(last_volume / vol_7ago)

        # Price vs MA
        feat["price_over_ma7"] = close.iloc[i-1] / ma7.iloc[i-1]
        feat["ma7_slope"] = ma7.iloc[i-1] - ma7.iloc[i-2]

        # Pattern indicators
        feat["maxdd_N"] = max_drawdown(past_window)
        feat["skew"] = past_window.skew()
        feat["kurt"] = past_window.kurtosis()

        # Classification target
        future_ret = float(future_window.sum())
        if future_ret >= 0.03:
            y = 2      # strong upward movement
        elif future_ret <= -0.02:
            y = 0      # strong downward movement
        else:
            y = 1      # neutral / sideways

        feature_list.append(feat)
        label_list.append(y)
        index_list.append(rets.index[i-1])

    X = pd.DataFrame(feature_list, index=index_list)
    y = np.array(label_list)

    return X, y


# ---------------------------------------------
# 4. Main training script
# ---------------------------------------------
if __name__ == "__main__":
    close, high, low, volume = download_btc("2000-01-01")

    N = 30
    K = 3
    ATR_WINDOW = 14

    X, y = build_feature_dataset(
        close=close,
        high=high,
        low=low,
        volume=volume,
        N=N,
        K=K,
        atr_window=ATR_WINDOW,
    )

    print("X samples:", X.shape[0], " / y samples:", y.shape[0])
    print("Number of features:", X.shape[1])
    print("\nFeature columns:\n", X.columns)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    print("\nTrain size:", X_train.shape[0], " / Test size:", X_test.shape[0])

    # RandomForest model
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    clf.fit(X_train, y_train)

    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    print(f"\nTrain Accuracy: {train_acc:.3f}")
    print(f"Test  Accuracy: {test_acc:.3f}")

    y_pred = clf.predict(X_test)

    print("\nClassification report (0=down, 1=neutral, 2=up):")
    print(classification_report(y_test, y_pred))

    print("\nConfusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_test, y_pred))

    # Feature importances
    importances = pd.Series(clf.feature_importances_, index=X.columns)
    importances = importances.sort_values(ascending=False)

    print("\nTop 10 Feature Importances:")
    print(importances.head(10))

    # Example predictions on last few samples
    proba = clf.predict_proba(X_test.tail(5))
    print("\nPredicted class probabilities for last 5 samples:")
    print(proba)

    print("\nTrue labels for last 5 samples:", y_test[-5:])
