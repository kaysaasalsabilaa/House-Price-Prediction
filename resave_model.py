import joblib

obj = joblib.load("best_model.pkl")

if hasattr(obj, "best_estimator_"):
    obj = obj.best_estimator_

joblib.dump(obj, "best_model.pkl", compress=3)

print("✅ Model berhasil di-save ulang ke best_model.pkl")
