"""
EDA + модель прогнозирования цены + анализ экспортного потенциала
для датасета вторичного авторынка Кореи.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

sns.set_theme(style="whitegrid", palette="deep")
IMG = "/home/claude/car_export_project/images"

df = pd.read_csv("/home/claude/car_export_project/data/cars_korea_export.csv")

print("Размер датасета:", df.shape)
print(df.info())
print(df.describe(include="all").T)

# ---------- 1. Распределение цен ----------
plt.figure(figsize=(8, 5))
sns.histplot(df["price_krw"] / 1_000_000, bins=40, kde=True, color="#1F4E5F")
plt.xlabel("Цена, млн KRW")
plt.ylabel("Количество объявлений")
plt.title("Распределение цен на вторичном рынке")
plt.tight_layout()
plt.savefig(f"{IMG}/01_price_distribution.png", dpi=130)
plt.close()

# ---------- 2. Цена vs пробег ----------
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="mileage_km", y=df["price_krw"] / 1_000_000,
                 hue="accident_history", palette={0: "#2E86AB", 1: "#C0392B"}, alpha=0.5, s=25)
plt.xlabel("Пробег, км")
plt.ylabel("Цена, млн KRW")
plt.title("Цена vs пробег (цвет = история ДТП)")
plt.legend(title="ДТП было", labels=["Нет", "Да"])
plt.tight_layout()
plt.savefig(f"{IMG}/02_price_vs_mileage.png", dpi=130)
plt.close()

# ---------- 3. Средняя цена по маркам/моделям ----------
top = df.groupby(["brand", "model"])["price_krw"].mean().sort_values(ascending=False) / 1_000_000
plt.figure(figsize=(9, 6))
top.plot(kind="barh", color="#3E7CB1")
plt.xlabel("Средняя цена, млн KRW")
plt.title("Средняя цена по моделям")
plt.tight_layout()
plt.savefig(f"{IMG}/03_avg_price_by_model.png", dpi=130)
plt.close()

# ---------- 4. Индекс экспортного спроса по моделям ----------
exp = df.groupby(["brand", "model"])["export_demand_index"].mean().sort_values(ascending=False)
plt.figure(figsize=(9, 6))
exp.plot(kind="barh", color="#1F9E89")
plt.xlabel("Индекс экспортного спроса (эвристика)")
plt.title("Какие модели наиболее востребованы для экспорта")
plt.tight_layout()
plt.savefig(f"{IMG}/04_export_demand_by_model.png", dpi=130)
plt.close()

# ---------- Feature engineering ----------
df["price_per_km"] = df["price_krw"] / df["mileage_km"].replace(0, np.nan)
df["brand_model"] = df["brand"] + " " + df["model"]

features_num = ["age_years", "mileage_km", "owners_count", "engine_cc"]
features_cat = ["brand_model", "fuel_type", "transmission", "region", "accident_history"]

X = df[features_num + features_cat]
y = df["price_krw"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), features_cat),
], remainder="passthrough")

model = Pipeline([
    ("prep", preprocess),
    ("rf", RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)),
])

model.fit(X_train, y_train)
pred = model.predict(X_test)

mae = mean_absolute_error(y_test, pred)
r2 = r2_score(y_test, pred)
print(f"\nMAE: {mae:,.0f} KRW")
print(f"R2: {r2:.3f}")

# ---------- 5. Feature importance ----------
ohe = model.named_steps["prep"].named_transformers_["cat"]
cat_names = ohe.get_feature_names_out(features_cat)
all_names = list(cat_names) + features_num
importances = model.named_steps["rf"].feature_importances_

imp_df = pd.DataFrame({"feature": all_names, "importance": importances})
imp_df = imp_df.sort_values("importance", ascending=False).head(15)

plt.figure(figsize=(9, 6))
sns.barplot(data=imp_df, y="feature", x="importance", color="#6A4C93")
plt.title("Топ-15 важных признаков для прогноза цены (Random Forest)")
plt.tight_layout()
plt.savefig(f"{IMG}/05_feature_importance.png", dpi=130)
plt.close()

# ---------- 6. Факт vs Прогноз ----------
plt.figure(figsize=(7, 7))
plt.scatter(y_test / 1_000_000, pred / 1_000_000, alpha=0.4, color="#2E86AB", s=20)
lims = [0, max(y_test.max(), pred.max()) / 1_000_000]
plt.plot(lims, lims, "--", color="#C0392B")
plt.xlabel("Фактическая цена, млн KRW")
plt.ylabel("Прогноз, млн KRW")
plt.title(f"Факт vs Прогноз (R² = {r2:.2f})")
plt.tight_layout()
plt.savefig(f"{IMG}/06_actual_vs_predicted.png", dpi=130)
plt.close()

# ---------- Бизнес-инсайт: топ моделей по (спрос / цена) для экспортной маржи ----------
biz = df.groupby(["brand", "model"]).agg(
    avg_price_m=("price_krw", lambda s: s.mean() / 1_000_000),
    export_demand=("export_demand_index", "mean"),
).reset_index()
biz["value_score"] = biz["export_demand"] / biz["avg_price_m"]
biz = biz.sort_values("value_score", ascending=False)
biz.to_csv("/home/claude/car_export_project/data/export_value_ranking.csv", index=False)
print("\nТоп моделей по соотношению экспортный спрос / цена:")
print(biz.head(6).to_string(index=False))

with open("/home/claude/car_export_project/data/metrics.txt", "w") as f:
    f.write(f"MAE: {mae:,.0f} KRW\nR2: {r2:.3f}\n")
