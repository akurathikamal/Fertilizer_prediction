import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle


df = pd.read_csv("../data/fertilizer_data.csv")


le_soil = LabelEncoder()
le_crop = LabelEncoder()
le_fert = LabelEncoder()

df["Soil Type"] = le_soil.fit_transform(df["Soil Type"])
df["Crop Type"] = le_crop.fit_transform(df["Crop Type"])
df["Fertilizer Name"] = le_fert.fit_transform(df["Fertilizer Name"])


X = df.drop("Fertilizer Name", axis=1)
y = df["Fertilizer Name"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)


pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, pred))


with open("fertilizer_model.pkl", "wb") as f:
    pickle.dump(model, f)


with open("encoders.pkl", "wb") as f:
    pickle.dump({
        "soil": le_soil,
        "crop": le_crop,
        "fert": le_fert
    }, f)

X_train.to_csv("X_train_for_shap.csv", index=False)

print("Training complete. Model, encoders, and X_train saved successfully!")