import pickle

with open("../models/heart_disease_model.pkl", "rb") as f:
    model = pickle.load(f)


with open("heart_disease_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model re-saved successfully")
