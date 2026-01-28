import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import mean_squared_error



# --------------------------------------------------
# PLOT SETTINGS
# --------------------------------------------------
color_pal = sns.color_palette()
plt.style.use("fivethirtyeight")

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
df = pd.read_csv("PJME_hourly.csv")
df = df.set_index("Datetime")
df.index = pd.to_datetime(df.index)
df = df.sort_index()

df.plot(style=".", figsize=(15, 5), title="PJME Energy Use in MW", color=color_pal[0])
plt.show()

# --------------------------------------------------
# TRAIN / TEST SPLIT
# --------------------------------------------------
train = df.loc[df.index < "2015-01-01"]
test = df.loc[df.index >= "2015-01-01"]

fig, ax = plt.subplots(figsize=(15, 5))
train.plot(ax=ax, label="Training Set")
test.plot(ax=ax, label="Test Set")
ax.axvline(pd.to_datetime("2015-01-01"), color="black", ls="--")
plt.legend()
plt.title("Train/Test Split")
plt.show()

# --------------------------------------------------
# VISUALIZE WEEK DATA SAMPLE
# --------------------------------------------------
df.loc[(df.index > "2010-01-01") & (df.index < "2010-01-08")] \
    .plot(figsize=(15, 5), title="Week Of Data")
plt.show()

# --------------------------------------------------
# FEATURE ENGINEERING FUNCTION
# --------------------------------------------------
def create_features(df):
    df = df.copy()
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["quarter"] = df.index.quarter
    df["month"] = df.index.month
    df["year"] = df.index.year
    df["dayofyear"] = df.index.dayofyear
    return df

df = create_features(df)

# --------------------------------------------------
# BOX PLOT ANALYSIS
# --------------------------------------------------
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="hour", y="PJME_MW", hue="hour", palette="Blues", legend=False)
plt.title("MW by Hour")
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="month", y="PJME_MW", hue="month", palette="Blues", legend=False)
plt.title("MW by Month")
plt.show()

# --------------------------------------------------
# TRAIN MODEL
# --------------------------------------------------
train = create_features(train)
test = create_features(test)

FEATURES = ["dayofyear", "hour", "dayofweek", "quarter", "month", "year"]
TARGET = "PJME_MW"

X_train = train[FEATURES]
y_train = train[TARGET]

X_test = test[FEATURES]
y_test = test[TARGET]

reg = xgb.XGBRegressor(
    n_estimators=1000,
    early_stopping_rounds=50,
    objective="reg:squarederror",
    max_depth=3,
    learning_rate=0.01
)

reg.fit(
    X_train,
    y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=100
)

# --------------------------------------------------
# FEATURE IMPORTANCE (FIXED)
# --------------------------------------------------
fi = pd.DataFrame(
    data=reg.feature_importances_,
    index=FEATURES,   # ✅ FIXED HERE
    columns=["importance"]
)

fi.sort_values("importance").plot(kind="barh", figsize=(6, 4), title="Feature Importance")
plt.show()

# --------------------------------------------------
# MAKE FORECAST
# --------------------------------------------------
test["prediction"] = reg.predict(X_test)

df = df.merge(test[["prediction"]], left_index=True, right_index=True, how="left")

ax = df[["PJME_MW"]].plot(figsize=(15, 5))
df["prediction"].plot(ax=ax, style=".")
plt.title("Prediction vs Real Value (Full Range)")
plt.show()

# --------------------------------------------------
# ZOOM INTO ONE WEEK FORECAST
# --------------------------------------------------
week_slice = df.loc["2018-04-01":"2018-04-08"]

week_slice[["PJME_MW"]].plot(figsize=(15, 5), title="True vs Predicted – One Week")
week_slice["prediction"].plot(style=".")
plt.legend(["Truth Data", "Prediction"])
plt.show()

# --------------------------------------------------
# RMSE SCORE
# --------------------------------------------------
score = np.sqrt(mean_squared_error(test["PJME_MW"], test["prediction"]))
print(f"RMSE Score on Test Set: {score:.2f}")

# --------------------------------------------------
# BEST / WORST PREDICTED DAYS
# --------------------------------------------------
test["error"] = np.abs(test[TARGET] - test["prediction"])
test["date"] = test.index.date

print("\n🔴 Worst 10 Days (Highest Prediction Error)")
print(test.groupby("date")["error"].mean().sort_values(ascending=False).head(10))

print("\n🟢 Best 10 Days (Lowest Prediction Error)")
print(test.groupby("date")["error"].mean().sort_values().head(10))
