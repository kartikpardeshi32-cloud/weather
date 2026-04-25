# weather

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier


url = "https://raw.githubusercontent.com/AyushAgnihotri2025/SmartCrop/master/SmartCrop-Dataset.csv"
data = pd.read_csv(url)

data = data.dropna()
data.columns = [col.strip().lower() for col in data.columns]

# Knowledge base
crop_stats = data.groupby("label").mean()

X = data.drop("label", axis=1)
y = data["label"]

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42
)


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("✅ Model trained successfully!")


try:
    print("\nEnter Values:")

    N = float(input("Nitrogen (0-300): "))
    P = float(input("Phosphorus (0-300): "))
    K = float(input("Potassium (0-300): "))
    temp = float(input("Temperature (-10 to 60°C): "))
    humidity = float(input("Humidity (0-100%): "))
    ph = float(input("pH (0-14): "))
    rainfall = float(input("Rainfall (0-5000 mm): "))

    if not (0 <= N <= 300 and 0 <= P <= 300 and 0 <= K <= 300 and
            -10 <= temp <= 60 and 0 <= humidity <= 100 and
            0 <= ph <= 14 and 0 <= rainfall <= 5000):
        print("❌ Entries are wrong")
        exit()

except:
    print("❌ Entries are wrong")
    exit()


input_df = pd.DataFrame([[N, P, K, temp, humidity, ph, rainfall]],
                        columns=X.columns)

input_scaled = scaler.transform(input_df)

probs = model.predict_proba(input_scaled)[0]

top3_idx = np.argsort(probs)[-3:][::-1]
top3_crops = encoder.inverse_transform(top3_idx)
top3_probs = probs[top3_idx]


market_price = {
    "rice": 2000,
    "wheat": 1800,
    "maize": 1700,
    "cotton": 6000,
    "sugarcane": 300
}

yield_data = {
    "rice": 25,
    "wheat": 20,
    "maize": 18,
    "cotton": 10,
    "sugarcane": 300
}


def get_logic(crop, user_input):
    stats = crop_stats.loc[crop]
    reasons = []

    features = ["n", "p", "k", "temperature", "humidity", "ph", "rainfall"]

    for i, feature in enumerate(features):
        user_val = user_input[i]
        ideal_val = stats[feature]

        diff = abs(user_val - ideal_val)

        if diff < 10:
            reasons.append(f"{feature} ✔ (close to ideal)")
        else:
            reasons.append(f"{feature} ⚠ (difference: {diff:.1f})")

    return reasons

# =========================
# FINAL SUMMARY FUNCTION
# =========================
def get_summary(best_crop, profit, user_input):
    stats = crop_stats.loc[best_crop]

    match_score = 0
    features = ["n", "p", "k", "temperature", "humidity", "ph", "rainfall"]

    for i, feature in enumerate(features):
        if abs(user_input[i] - stats[feature]) < 10:
            match_score += 1

    if match_score >= 5:
        suitability = "highly suitable"
    elif match_score >= 3:
        suitability = "moderately suitable"
    else:
        suitability = "less suitable"

    sentence = f"""
👉 Under the given conditions, {best_crop} is {suitability} and is expected to give higher profit (~₹{profit}/acre).
"""
    return sentence

# =========================
# OUTPUT
# =========================
print("\n==============================")
print("🌾 FarmX Intelligent System")
print("==============================")

user_input = [N, P, K, temp, humidity, ph, rainfall]

for i, (crop, prob) in enumerate(zip(top3_crops, top3_probs), 1):
    price = market_price.get(crop, 1500)
    yield_est = yield_data.get(crop, 20)
    profit = price * yield_est

    print(f"\n{i}. {crop} ({prob*100:.2f}% confidence)")
    print(f"💰 Estimated Profit: ₹{profit}/acre")

    logic = get_logic(crop, user_input)

    print("📊 Reasoning:")
    for l in logic:
        print("  -", l)

# Best crop summary
best_crop = top3_crops[0]
best_profit = market_price.get(best_crop, 1500) * yield_data.get(best_crop, 20)

summary = get_summary(best_crop, best_profit, user_input)

print("\n📌 Final Recommendation:")
print(summary)

print("\n==============================")
