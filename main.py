import os

import random

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import tensorflow as tf

from ucimlrepo import fetch_ucirepo

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras import Sequential

from tensorflow.keras.layers import Input, LSTM, Dense, Dropout

from tensorflow.keras.callbacks import EarlyStopping

# -------------------- 1. Настройки --------------------

SEED = 42

WINDOW = 24          # Используем предыдущие 24 часа

EPOCHS = 15

BATCH_SIZE = 64

random.seed(SEED)

np.random.seed(SEED)

tf.random.set_seed(SEED)

os.makedirs("results", exist_ok=True)

# -------------------- 2. Загрузка данных --------------------

print("Загрузка набора данных UCI...")

dataset = fetch_ucirepo(id=492)

df = pd.concat(

    [dataset.data.features, dataset.data.targets],

    axis=1

)

df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")

df["traffic_volume"] = pd.to_numeric(

    df["traffic_volume"], errors="coerce"

)

df = df[["date_time", "traffic_volume"]].dropna()

df = df.sort_values("date_time")

df = df.drop_duplicates(subset="date_time", keep="last")

df = df.set_index("date_time")

# Приводим данные к регулярному часовому интервалу

df = df.resample("h").mean()

# Заполняем только небольшие пропуски в последовательности

df["traffic_volume"] = df["traffic_volume"].interpolate(

    method="time", limit=3, limit_area="inside"

)

df = df.dropna().copy()

print("\nПервые строки данных:")

print(df.head())

print("\nКоличество часовых наблюдений:", len(df))

print("Период:", df.index.min(), "—", df.index.max())

# -------------------- 3. Признаки времени --------------------

hour = df.index.hour

day_of_week = df.index.dayofweek

month = df.index.month

df["hour_sin"] = np.sin(2 * np.pi * hour / 24)

df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

df["weekday_sin"] = np.sin(2 * np.pi * day_of_week / 7)

df["weekday_cos"] = np.cos(2 * np.pi * day_of_week / 7)

df["month_sin"] = np.sin(2 * np.pi * month / 12)

df["month_cos"] = np.cos(2 * np.pi * month / 12)

# -------------------- 4. График исходного ряда --------------------

plt.figure(figsize=(14, 5))

plt.plot(df.index, df["traffic_volume"], linewidth=0.7)

plt.title("Интенсивность дорожного движения")

plt.xlabel("Дата")

plt.ylabel("Количество автомобилей за час")

plt.grid(True)

plt.tight_layout()

plt.savefig("results/01_traffic_history.png", dpi=150)

plt.show()

# -------------------- 5. Хронологическое разделение --------------------

n = len(df)

train_end = int(n * 0.70)

val_end = int(n * 0.85)

# Масштабируем целевой показатель только по обучающей части

scaler = MinMaxScaler()

scaler.fit(df[["traffic_volume"]].iloc[:train_end])

traffic_scaled = scaler.transform(

    df[["traffic_volume"]]

).flatten()

calendar_features = df[

    [

        "hour_sin", "hour_cos",

        "weekday_sin", "weekday_cos",

        "month_sin", "month_cos"

    ]

].to_numpy(dtype=np.float32)

# Первый признак — масштабированный трафик

all_features = np.column_stack(

    [traffic_scaled, calendar_features]

).astype(np.float32)

# -------------------- 6. Формирование последовательностей --------------------

X, y, target_positions = [], [], []

for i in range(WINDOW, len(df)):

    X.append(all_features[i - WINDOW:i])

    y.append(traffic_scaled[i])

    target_positions.append(i)

X = np.array(X, dtype=np.float32)

y = np.array(y, dtype=np.float32)

target_positions = np.array(target_positions)

train_mask = target_positions < train_end

val_mask = (

    (target_positions >= train_end)

    & (target_positions < val_end)

)

test_mask = target_positions >= val_end

X_train, y_train = X[train_mask], y[train_mask]

X_val, y_val = X[val_mask], y[val_mask]

X_test, y_test = X[test_mask], y[test_mask]

print("\nРазмеры выборок:")

print("Обучающая:", X_train.shape, y_train.shape)

print("Валидационная:", X_val.shape, y_val.shape)

print("Тестовая:", X_test.shape, y_test.shape)

# -------------------- 7. Создание LSTM --------------------

def build_model(units=32, dropout=0.0):

    model = Sequential([

        Input(shape=(WINDOW, X_train.shape[2])),

        LSTM(units),

        Dropout(dropout),

        Dense(16, activation="relu"),

        Dense(1)

    ])

    model.compile(

        optimizer="adam",

        loss="mse",

        metrics=["mae"]

    )

    return model

# -------------------- 8. Обучение и сравнение --------------------

experiments = [

    {

        "name": "LSTM_32",

        "units": 32,

        "dropout": 0.0

    },

    {

        "name": "LSTM_64_Dropout",

        "units": 64,

        "dropout": 0.2

    }

]

trained_models = {}

histories = {}

validation_scores = {}

for config in experiments:

    print("\n" + "=" * 55)

    print("Обучение модели:", config["name"])

    tf.keras.backend.clear_session()

    tf.random.set_seed(SEED)

    model = build_model(

        units=config["units"],

        dropout=config["dropout"]

    )

    model.summary()

    early_stop = EarlyStopping(

        monitor="val_loss",

        patience=3,

        restore_best_weights=True

    )

    history = model.fit(

        X_train,

        y_train,

        validation_data=(X_val, y_val),

        epochs=EPOCHS,

        batch_size=BATCH_SIZE,

        shuffle=False,

        callbacks=[early_stop],

        verbose=1

    )

    val_prediction_scaled = model.predict(X_val, verbose=0)

    val_prediction = scaler.inverse_transform(

        val_prediction_scaled

    ).flatten()

    val_actual = scaler.inverse_transform(

        y_val.reshape(-1, 1)

    ).flatten()

    val_rmse = np.sqrt(

        mean_squared_error(val_actual, val_prediction)

    )

    validation_scores[config["name"]] = val_rmse

    trained_models[config["name"]] = model

    histories[config["name"]] = history

    print(f"Валидационный RMSE: {val_rmse:.2f}")

# Выбираем лучшую модель по валидации, а не по тесту

best_name = min(validation_scores, key=validation_scores.get)

best_model = trained_models[best_name]

print("\nЛучшая модель по валидационной выборке:", best_name)

# -------------------- 9. Оценка на тестовой выборке --------------------

results = []

for name, model in trained_models.items():

    prediction_scaled = model.predict(X_test, verbose=0)

    prediction = scaler.inverse_transform(

        prediction_scaled

    ).flatten()

    actual = scaler.inverse_transform(

        y_test.reshape(-1, 1)

    ).flatten()

    mae = mean_absolute_error(actual, prediction)

    rmse = np.sqrt(mean_squared_error(actual, prediction))

    r2 = r2_score(actual, prediction)

    results.append({

        "Модель": name,

        "MAE": mae,

        "RMSE": rmse,

        "R2": r2

    })

    print(f"\n{name}")

    print(f"MAE:  {mae:.2f} автомобилей")

    print(f"RMSE: {rmse:.2f} автомобилей")

    print(f"R²:   {r2:.4f}")

results_df = pd.DataFrame(results)

results_df.to_csv(

    "results/model_comparison.csv",

    index=False,

    encoding="utf-8-sig"

)

print("\nСравнение моделей:")

print(results_df.to_string(index=False))

# -------------------- 10. График обучения --------------------

plt.figure(figsize=(10, 5))

for name, history in histories.items():

    plt.plot(history.history["loss"], label=f"{name}: train")

    plt.plot(history.history["val_loss"], linestyle="--",

             label=f"{name}: validation")

plt.title("Изменение функции потерь при обучении")

plt.xlabel("Эпоха")

plt.ylabel("MSE")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig("results/02_training_loss.png", dpi=150)

plt.show()

# -------------------- 11. Прогноз лучшей модели --------------------

best_prediction_scaled = best_model.predict(X_test, verbose=0)

best_prediction = scaler.inverse_transform(

    best_prediction_scaled

).flatten()

actual = scaler.inverse_transform(

    y_test.reshape(-1, 1)

).flatten()

test_dates = df.index[target_positions[test_mask]]

# Показываем первые 7 дней тестового периода

points = min(24 * 7, len(actual))

plt.figure(figsize=(14, 5))

plt.plot(

    test_dates[:points],

    actual[:points],

    label="Фактический трафик"

)

plt.plot(

    test_dates[:points],

    best_prediction[:points],

    label="Прогноз LSTM"

)

plt.title(f"Фактический трафик и прогноз: {best_name}")

plt.xlabel("Дата")

plt.ylabel("Количество автомобилей за час")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig("results/03_forecast.png", dpi = 150)

plt.show()

# -------------------- 12. График сравнения моделей --------------------

plt.figure(figsize=(9, 5))

plt.bar(results_df["Модель"], results_df["RMSE"])

plt.title("Сравнение моделей по RMSE")

plt.xlabel("Модель")

plt.ylabel("RMSE — меньше лучше")

plt.grid(axis="y")

plt.tight_layout()

plt.savefig("results/04_model_comparison.png", dpi=150)

plt.show()

# -------------------- 13. Сохранение лучшей модели --------------------

best_model.save("results/best_traffic_lstm.keras")

# Сохраняем прогнозы и фактические значения

predictions_df = pd.DataFrame({

    "date_time": test_dates,

    "actual_traffic": actual,

    "predicted_traffic": best_prediction

})

predictions_df.to_csv(

    "results/test_predictions.csv",

    index=False,

    encoding="utf-8-sig"

)

print("\nРабота завершена.")

print("Графики, метрики, прогнозы и модель находятся в папке results.")