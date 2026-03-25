# =========================
# ITA105 - LAB 2 FULL CODE
# =========================

# Nếu Colab thiếu thư viện thì mở dòng dưới:
# !pip install pandas numpy matplotlib seaborn scipy scikit-learn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import zscore

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)

# =========================
# 1. LOAD DATA
# =========================
housing = pd.read_csv("ITA105_Lab_2_Housing.csv")
iot = pd.read_csv("ITA105_Lab_2_Iot.csv")
ecom = pd.read_csv("ITA105_Lab_2_Ecommerce.csv")

print("Housing shape:", housing.shape)
print("IoT shape:", iot.shape)
print("Ecommerce shape:", ecom.shape)

print("\nHousing columns:", housing.columns.tolist())
print("IoT columns:", iot.columns.tolist())
print("Ecommerce columns:", ecom.columns.tolist())


# =========================
# HÀM HỖ TRỢ
# =========================
def iqr_bounds(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return Q1, Q3, IQR, lower, upper

def get_iqr_outliers(df, col):
    _, _, _, lower, upper = iqr_bounds(df[col])
    return df[(df[col] < lower) | (df[col] > upper)]

def add_zscore_columns(df, cols):
    df = df.copy()
    for col in cols:
        df[f"z_{col}"] = zscore(df[col], nan_policy="omit")
    return df

def clip_by_iqr(df, cols):
    df = df.copy()
    for col in cols:
        _, _, _, lower, upper = iqr_bounds(df[col])
        df[col] = df[col].clip(lower, upper)
    return df

def describe_full(df, numeric_cols):
    result = df[numeric_cols].agg(['mean', 'median', 'std', 'min', 'max']).T
    return result


# =========================================================
# BÀI 1: KHÁM PHÁ DỮ LIỆU HOUSING
# =========================================================
print("\n" + "="*60)
print("BÀI 1: HOUSING")
print("="*60)

# 1. Nạp dữ liệu, kiểm tra shape, missing values
print("\n1. Shape và missing values")
print("Shape:", housing.shape)
print(housing.isnull().sum())

# 2. In thống kê mô tả
housing_num = ["dien_tich", "gia", "so_phong"]
print("\n2. Thống kê mô tả")
print(describe_full(housing, housing_num))

# 3. Vẽ boxplot cho từng biến numeric
print("\n3. Boxplot các biến numeric")
for col in housing_num:
    plt.figure()
    sns.boxplot(x=housing[col])
    plt.title(f"Boxplot - {col}")
    plt.show()

# 4. Scatterplot diện tích và giá
print("\n4. Scatterplot diện tích và giá")
plt.figure()
sns.scatterplot(data=housing, x="dien_tich", y="gia")
plt.title("Scatterplot: dien_tich vs gia")
plt.show()

# 5. Tính IQR, xác định ngoại lệ theo công thức
print("\n5. Ngoại lệ theo IQR")
housing_iqr_summary = {}
for col in housing_num:
    outliers = get_iqr_outliers(housing, col)
    housing_iqr_summary[col] = len(outliers)
    print(f"{col}: {len(outliers)} outliers")

# 6. Tính Z-score cho từng biến numeric và xác định ngoại lệ (|Z| > 3)
print("\n6. Ngoại lệ theo Z-score")
housing_z = add_zscore_columns(housing, housing_num)
housing_z_summary = {}
for col in housing_num:
    outliers = housing_z[np.abs(housing_z[f"z_{col}"]) > 3]
    housing_z_summary[col] = len(outliers)
    print(f"{col}: {len(outliers)} outliers")

# 7. So sánh số lượng ngoại lệ IQR, Z-score, boxplot
print("\n7. So sánh số lượng ngoại lệ")
housing_compare = pd.DataFrame({
    "IQR": housing_iqr_summary,
    "Z-score": housing_z_summary,
    "Boxplot (gần tương đương IQR)": housing_iqr_summary
})
print(housing_compare)

# 8. Phân tích nguyên nhân
print("\n8. Nhận xét gợi ý:")
print("- Giá trị quá cao/thấp có thể là nhà đặc biệt, diện tích bất thường hoặc lỗi nhập liệu.")
print("- Boxplot/IQR nhạy với phân phối lệch, Z-score phù hợp hơn khi dữ liệu gần chuẩn.")

# 9. Áp dụng một phương pháp xử lý ngoại lệ
# Chọn clip theo IQR
print("\n9. Xử lý ngoại lệ bằng clip theo IQR")
housing_clean = clip_by_iqr(housing, housing_num)

# 10. Vẽ lại boxplot sau xử lý
print("\n10. Boxplot sau xử lý")
for col in housing_num:
    plt.figure()
    sns.boxplot(x=housing_clean[col])
    plt.title(f"Boxplot sau xử lý - {col}")
    plt.show()


# =========================================================
# BÀI 2: PHÁT HIỆN NGOẠI LỆ TRONG DỮ LIỆU IoT / SENSOR
# =========================================================
print("\n" + "="*60)
print("BÀI 2: IOT / SENSOR")
print("="*60)

# 1. Load dữ liệu, set timestamp làm index, kiểm tra missing values
iot["timestamp"] = pd.to_datetime(iot["timestamp"])
iot = iot.sort_values(["sensor_id", "timestamp"])
iot_indexed = iot.set_index("timestamp")

print("\n1. Missing values")
print(iot_indexed.isnull().sum())

# 2. Vẽ line plot temperature theo thời gian cho từng sensor
print("\n2. Line plot temperature theo từng sensor")
for sid in iot["sensor_id"].unique():
    plt.figure(figsize=(10,4))
    temp_df = iot[iot["sensor_id"] == sid]
    plt.plot(temp_df["timestamp"], temp_df["temperature"])
    plt.title(f"Temperature over time - Sensor {sid}")
    plt.xlabel("Time")
    plt.ylabel("Temperature")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# 3. Phát hiện ngoại lệ bằng rolling mean ± 3*std (window=10)
print("\n3. Ngoại lệ theo rolling mean ± 3*std")
iot_roll = iot.copy()
iot_roll["rolling_mean"] = iot_roll.groupby("sensor_id")["temperature"].transform(
    lambda x: x.rolling(window=10, min_periods=1).mean()
)
iot_roll["rolling_std"] = iot_roll.groupby("sensor_id")["temperature"].transform(
    lambda x: x.rolling(window=10, min_periods=1).std()
)
iot_roll["upper"] = iot_roll["rolling_mean"] + 3 * iot_roll["rolling_std"]
iot_roll["lower"] = iot_roll["rolling_mean"] - 3 * iot_roll["rolling_std"]

iot_roll["outlier_rolling"] = (
    (iot_roll["temperature"] > iot_roll["upper"]) |
    (iot_roll["temperature"] < iot_roll["lower"])
)

print("Số outlier rolling:", iot_roll["outlier_rolling"].sum())

# 4. Tính Z-score cho từng sensor và xác định ngoại lệ (|Z| > 3)
print("\n4. Ngoại lệ theo Z-score từng sensor")
iot_z = iot.copy()
iot_z["z_temperature"] = iot_z.groupby("sensor_id")["temperature"].transform(
    lambda x: zscore(x, nan_policy="omit")
)
iot_z["outlier_z"] = np.abs(iot_z["z_temperature"]) > 3
print("Số outlier Z-score:", iot_z["outlier_z"].sum())

# 5. Vẽ boxplot và scatter plot giữa các biến
print("\n5. Boxplot và scatter plot highlight outlier")

for col in ["temperature", "pressure", "humidity"]:
    plt.figure()
    sns.boxplot(x=iot[col])
    plt.title(f"Boxplot - {col}")
    plt.show()

# Scatter temperature vs pressure
iot_plot = iot_z.merge(
    iot_roll[["timestamp", "sensor_id", "outlier_rolling"]],
    on=["timestamp", "sensor_id"],
    how="left"
)

plt.figure(figsize=(8,6))
sns.scatterplot(
    data=iot_plot,
    x="temperature",
    y="pressure",
    hue="outlier_z",
    palette="Set1"
)
plt.title("Scatter: temperature vs pressure (highlight Z-score outliers)")
plt.show()

# Scatter pressure vs humidity
# Dùng IQR trên pressure và humidity để highlight
_, _, _, p_low, p_up = iqr_bounds(iot["pressure"])
_, _, _, h_low, h_up = iqr_bounds(iot["humidity"])

iot["outlier_scatter"] = (
    (iot["pressure"] < p_low) | (iot["pressure"] > p_up) |
    (iot["humidity"] < h_low) | (iot["humidity"] > h_up)
)

plt.figure(figsize=(8,6))
sns.scatterplot(
    data=iot,
    x="pressure",
    y="humidity",
    hue="outlier_scatter",
    palette="Set1"
)
plt.title("Scatter: pressure vs humidity")
plt.show()

# 6. So sánh số lượng ngoại lệ
print("\n6. So sánh số lượng ngoại lệ")
iot_compare = {
    "Rolling mean ± 3*std": int(iot_roll["outlier_rolling"].sum()),
    "Z-score temperature": int(iot_z["outlier_z"].sum()),
    "Boxplot temperature(IQR)": int(len(get_iqr_outliers(iot, "temperature"))),
    "Scatter pressure-humidity(IQR-based)": int(iot["outlier_scatter"].sum())
}
print(pd.DataFrame.from_dict(iot_compare, orient="index", columns=["So_luong_outlier"]))

# 7. Xử lý ngoại lệ bằng interpolation hoặc clip
print("\n7. Xử lý ngoại lệ bằng interpolation")
iot_clean = iot_roll.copy()
iot_clean["temperature_clean"] = iot_clean["temperature"]

# Gán NaN tại outlier rolling rồi nội suy theo từng sensor
iot_clean.loc[iot_clean["outlier_rolling"], "temperature_clean"] = np.nan
iot_clean["temperature_clean"] = iot_clean.groupby("sensor_id")["temperature_clean"].transform(
    lambda x: x.interpolate(method="linear")
)

# Vẽ lại dữ liệu trước/sau xử lý
for sid in iot_clean["sensor_id"].unique():
    tmp = iot_clean[iot_clean["sensor_id"] == sid]
    plt.figure(figsize=(10,4))
    plt.plot(tmp["timestamp"], tmp["temperature"], label="Original")
    plt.plot(tmp["timestamp"], tmp["temperature_clean"], label="Cleaned")
    plt.title(f"Before vs After cleaning - Sensor {sid}")
    plt.xlabel("Time")
    plt.ylabel("Temperature")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# =========================================================
# BÀI 3: NGOẠI LỆ TRONG GIAO DỊCH E-COMMERCE
# =========================================================
print("\n" + "="*60)
print("BÀI 3: E-COMMERCE")
print("="*60)

# 1. Load dữ liệu, kiểm tra missing values, thống kê mô tả
print("\n1. Missing values")
print(ecom.isnull().sum())

ecom_num = ["price", "quantity", "rating"]
print("\nThống kê mô tả")
print(describe_full(ecom, ecom_num))

# 2. Vẽ boxplot cho price, quantity, rating
print("\n2. Boxplot")
for col in ecom_num:
    plt.figure()
    sns.boxplot(x=ecom[col])
    plt.title(f"Boxplot - {col}")
    plt.show()

# 3. Tính IQR và Z-score cho các biến numeric
print("\n3. IQR và Z-score")

ecom_iqr_summary = {}
for col in ecom_num:
    outliers = get_iqr_outliers(ecom, col)
    ecom_iqr_summary[col] = len(outliers)
    print(f"IQR - {col}: {len(outliers)} outliers")

ecom_z = add_zscore_columns(ecom, ecom_num)
ecom_z_summary = {}
for col in ecom_num:
    outliers = ecom_z[np.abs(ecom_z[f'z_{col}']) > 3]
    ecom_z_summary[col] = len(outliers)
    print(f"Z-score - {col}: {len(outliers)} outliers")

# 4. Scatterplot price vs quantity và đánh dấu ngoại lệ
print("\n4. Scatterplot price vs quantity")
ecom_plot = ecom.copy()

# Đánh dấu outlier nếu price hoặc quantity là outlier theo IQR
_, _, _, price_low, price_up = iqr_bounds(ecom["price"])
_, _, _, qty_low, qty_up = iqr_bounds(ecom["quantity"])

ecom_plot["outlier_scatter"] = (
    (ecom_plot["price"] < price_low) | (ecom_plot["price"] > price_up) |
    (ecom_plot["quantity"] < qty_low) | (ecom_plot["quantity"] > qty_up)
)

plt.figure(figsize=(8,6))
sns.scatterplot(
    data=ecom_plot,
    x="price",
    y="quantity",
    hue="outlier_scatter",
    palette="Set1"
)
plt.title("Scatterplot: price vs quantity")
plt.show()

# 5. Phân tích nguyên nhân
print("\n5. Kiểm tra các trường hợp nghi ngờ")
print("Số dòng price <= 0:", (ecom["price"] <= 0).sum())
print("Số dòng rating > 5:", (ecom["rating"] > 5).sum())
print("Số dòng quantity <= 0:", (ecom["quantity"] <= 0).sum())

print("\nTần suất category:")
print(ecom["category"].value_counts())

# Category hiếm
rare_categories = ecom["category"].value_counts()[ecom["category"].value_counts() < 5]
print("\nCategory hiếm (<5 dòng):")
print(rare_categories)

# 6. Xử lý ngoại lệ
print("\n6. Xử lý ngoại lệ")

ecom_clean = ecom.copy()

# 6.1 Loại bỏ giá trị lỗi nhập liệu
ecom_clean = ecom_clean[ecom_clean["price"] > 0]
ecom_clean = ecom_clean[ecom_clean["quantity"] > 0]
ecom_clean = ecom_clean[(ecom_clean["rating"] >= 0) & (ecom_clean["rating"] <= 5)]

# 6.2 Giữ category hiếm nếu hợp lý, chỉ clip numeric
for col in ecom_num:
    _, _, _, lower, upper = iqr_bounds(ecom_clean[col])
    ecom_clean[col] = ecom_clean[col].clip(lower, upper)

# 6.3 Log-transform price để giảm lệch
ecom_clean["log_price"] = np.log1p(ecom_clean["price"])

# 7. Vẽ lại box plot và scatter plot sau xử lý
print("\n7. Biểu đồ sau xử lý")
for col in ["price", "quantity", "rating", "log_price"]:
    plt.figure()
    sns.boxplot(x=ecom_clean[col])
    plt.title(f"Boxplot sau xử lý - {col}")
    plt.show()

plt.figure(figsize=(8,6))
sns.scatterplot(data=ecom_clean, x="price", y="quantity", hue="category")
plt.title("Scatterplot sau xử lý: price vs quantity")
plt.show()

# Bảng so sánh trước/sau
print("\nSo sánh trước/sau xử lý")
compare_before_after = pd.DataFrame({
    "Before_mean": ecom[ecom_num].mean(),
    "After_mean": ecom_clean[ecom_num].mean(),
    "Before_std": ecom[ecom_num].std(),
    "After_std": ecom_clean[ecom_num].std()
})
print(compare_before_after)


# =========================================================
# BÀI 4: MULTIVARIATE OUTLIER
# =========================================================
print("\n" + "="*60)
print("BÀI 4: MULTIVARIATE OUTLIER")
print("="*60)

# 1. Xác định ngoại lệ multivariate bằng kết hợp 2-3 biến
# Dùng quy tắc: một điểm được xem là multivariate outlier nếu có ít nhất 2 biến là outlier theo Z-score

print("\n1. Housing multivariate outlier")
housing_mv = add_zscore_columns(housing, ["dien_tich", "gia"])
housing_mv["mv_outlier"] = (
    (np.abs(housing_mv["z_dien_tich"]) > 3).astype(int) +
    (np.abs(housing_mv["z_gia"]) > 3).astype(int)
) >= 2
print("Số multivariate outlier Housing:", housing_mv["mv_outlier"].sum())

plt.figure(figsize=(8,6))
sns.scatterplot(
    data=housing_mv,
    x="dien_tich",
    y="gia",
    hue="mv_outlier",
    palette="Set1"
)
plt.title("Housing: dien_tich vs gia")
plt.show()

print("\n2. IoT multivariate outlier")
iot_mv = iot.copy()
iot_mv["z_temperature"] = zscore(iot_mv["temperature"], nan_policy="omit")
iot_mv["z_pressure"] = zscore(iot_mv["pressure"], nan_policy="omit")
iot_mv["mv_outlier"] = (
    (np.abs(iot_mv["z_temperature"]) > 3).astype(int) +
    (np.abs(iot_mv["z_pressure"]) > 3).astype(int)
) >= 2
print("Số multivariate outlier IoT:", iot_mv["mv_outlier"].sum())

plt.figure(figsize=(8,6))
sns.scatterplot(
    data=iot_mv,
    x="temperature",
    y="pressure",
    hue="mv_outlier",
    palette="Set1"
)
plt.title("IoT: temperature vs pressure")
plt.show()

print("\n3. E-commerce multivariate outlier")
ecom_mv = add_zscore_columns(ecom, ["price", "quantity", "rating"])
ecom_mv["mv_outlier"] = (
    (np.abs(ecom_mv["z_price"]) > 3).astype(int) +
    (np.abs(ecom_mv["z_quantity"]) > 3).astype(int) +
    (np.abs(ecom_mv["z_rating"]) > 3).astype(int)
) >= 2
print("Số multivariate outlier E-commerce:", ecom_mv["mv_outlier"].sum())

plt.figure(figsize=(8,6))
sns.scatterplot(
    data=ecom_mv,
    x="price",
    y="quantity",
    hue="mv_outlier",
    palette="Set1"
)
plt.title("E-commerce: price vs quantity")
plt.show()

# Scatter matrix đơn giản
print("\nScatter matrix cho E-commerce")
sns.pairplot(ecom_mv[["price", "quantity", "rating", "mv_outlier"]], hue="mv_outlier")
plt.show()

# 4. So sánh univariate vs multivariate
print("\n4. So sánh univariate vs multivariate")

housing_uni = (
    (np.abs(housing_mv["z_dien_tich"]) > 3) |
    (np.abs(housing_mv["z_gia"]) > 3)
).sum()
housing_multi = housing_mv["mv_outlier"].sum()

iot_uni = (
    (np.abs(iot_mv["z_temperature"]) > 3) |
    (np.abs(iot_mv["z_pressure"]) > 3)
).sum()
iot_multi = iot_mv["mv_outlier"].sum()

ecom_uni = (
    (np.abs(ecom_mv["z_price"]) > 3) |
    (np.abs(ecom_mv["z_quantity"]) > 3) |
    (np.abs(ecom_mv["z_rating"]) > 3)
).sum()
ecom_multi = ecom_mv["mv_outlier"].sum()

compare_uv_mv = pd.DataFrame({
    "Univariate_outlier": [housing_uni, iot_uni, ecom_uni],
    "Multivariate_outlier": [housing_multi, iot_multi, ecom_multi]
}, index=["Housing", "IoT", "E-commerce"])

print(compare_uv_mv)

print("\nNhận xét:")
print("- Univariate: xét từng biến riêng lẻ.")
print("- Multivariate: xét sự kết hợp nhiều biến cùng lúc.")
print("- Có điểm không quá lạ ở từng biến riêng, nhưng khi kết hợp lại thì bất thường.")


# =========================================================
# TỔNG KẾT
# =========================================================
print("\n" + "="*60)
print("TỔNG KẾT")
print("="*60)

print("\nHousing - số outlier theo IQR:")
print(housing_iqr_summary)

print("\nHousing - số outlier theo Z-score:")
print(housing_z_summary)

print("\nIoT - số outlier:")
print(iot_compare)

print("\nE-commerce - số outlier theo IQR:")
print(ecom_iqr_summary)

print("\nE-commerce - số outlier theo Z-score:")
print(ecom_z_summary)

print("\nSo sánh Univariate vs Multivariate:")
print(compare_uv_mv)
