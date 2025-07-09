from tensorflow.keras.models import load_model
model = load_model("models/model.h5")
model.save("models/model_compressed.h5", include_optimizer=False)
print("Model compressed and saved as models/model_compressed.h5")
