"""
Генерация синтетического, но реалистичного датасета вторичного рынка
автомобилей в Южной Корее — специально с акцентом на модели,
популярные для экспорта в Россию/Монголию/СНГ через агентов во Владивостоке.

Распределения цен, пробега и возраста откалиброваны по порядку величины
под реальный корейский рынок (KB Chachacha / Encar, публичные ценовые ориентиры),
чтобы датасет был правдоподобен для учебного портфолио-проекта.
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

N = 2500

# Модели, популярные для экспорта из Кореи (SUV/седаны, высокий спрос в РФ/Монголии)
MODELS = [
    ("Hyundai", "Tucson", 32_000_000, 0.9),
    ("Hyundai", "Santa Fe", 38_000_000, 1.0),
    ("Hyundai", "Sonata", 27_000_000, 0.8),
    ("Hyundai", "Avante", 20_000_000, 0.7),
    ("Kia", "Sportage", 31_000_000, 0.95),
    ("Kia", "Sorento", 37_000_000, 1.0),
    ("Kia", "K5", 26_000_000, 0.8),
    ("Kia", "Carnival", 42_000_000, 1.05),
    ("Genesis", "G80", 55_000_000, 1.1),
    ("Chevrolet", "Trailblazer", 28_000_000, 0.85),
    ("Ssangyong", "Torres", 30_000_000, 0.9),
    ("Renault Korea", "QM6", 26_000_000, 0.85),
]

FUEL = ["Бензин", "Дизель", "Гибрид", "LPG"]
FUEL_P = [0.45, 0.30, 0.20, 0.05]
TRANSMISSION = ["Автомат", "Механика"]
TRANSMISSION_P = [0.93, 0.07]
REGIONS = ["Сеул", "Инчхон", "Кёнгидо (Ансан)", "Пусан", "Тэгу", "Ульсан"]

rows = []
current_year = 2026

for _ in range(N):
    brand, model, base_price, demand_mult = MODELS[rng.integers(0, len(MODELS))]

    age = rng.integers(0, 11)  # 0-10 лет
    year = current_year - age

    # пробег растет с возрастом + случайный разброс (км/год ~ 14-16 тыс)
    avg_km_per_year = rng.normal(15000, 3000)
    mileage = max(500, int(age * avg_km_per_year + rng.normal(0, 4000)))

    fuel = rng.choice(FUEL, p=FUEL_P)
    transmission = rng.choice(TRANSMISSION, p=TRANSMISSION_P)
    region = rng.choice(REGIONS)

    accident = rng.choice([0, 1], p=[0.78, 0.22])  # была ли авария
    owners = rng.integers(1, 4)  # число владельцев

    engine_cc = int(rng.choice([1600, 1998, 2199, 2497, 2999]))

    # Ценовая модель: базовая цена минус амортизация минус пробег минус авария
    depreciation = base_price * (0.11 * age)
    mileage_penalty = mileage * rng.uniform(0.35, 0.55)
    accident_penalty = accident * base_price * rng.uniform(0.05, 0.15)
    owners_penalty = (owners - 1) * base_price * 0.015
    noise = rng.normal(0, base_price * 0.05)

    price = base_price - depreciation - mileage_penalty - accident_penalty - owners_penalty + noise
    price = max(price, base_price * 0.15)
    price = round(price / 100_000) * 100_000  # округление до 100k won

    # Экспортный спрос-индекс (эвристика: SUV/седаны средних лет, без аварий, авто-КПП ценятся выше в РФ/Монголии)
    export_demand = demand_mult
    export_demand *= 1.15 if transmission == "Автомат" else 0.9
    export_demand *= 0.75 if accident == 1 else 1.0
    export_demand *= 1.1 if 2 <= age <= 6 else 0.85
    export_demand *= 1.05 if fuel in ("Дизель", "Бензин") else 0.95
    export_demand = round(export_demand, 3)

    rows.append({
        "brand": brand,
        "model": model,
        "year": year,
        "age_years": age,
        "mileage_km": mileage,
        "fuel_type": fuel,
        "transmission": transmission,
        "region": region,
        "accident_history": accident,
        "owners_count": owners,
        "engine_cc": engine_cc,
        "price_krw": int(price),
        "export_demand_index": export_demand,
    })

df = pd.DataFrame(rows)
df.to_csv("/home/claude/car_export_project/data/cars_korea_export.csv", index=False, encoding="utf-8-sig")
print(df.shape)
print(df.head())
