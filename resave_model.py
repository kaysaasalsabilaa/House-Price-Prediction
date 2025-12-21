import joblib

# load model lama
obj = joblib.load("best_model.pkl")

# kalau kamu simpan GridSearchCV/RandomizedSearchCV, simpan best_estimator_ aja
if hasattr(obj, "best_estimator_"):
    obj = obj.best_estimator_

# save ulang (lebih aman + lebih kecil)
joblib.dump(obj, "best_model.pkl", compress=3)

print("✅ Model berhasil di-save ulang ke best_model.pkl")
