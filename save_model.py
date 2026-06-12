import os

os.makedirs("model", exist_ok=True)
cnn.save("model/cnn_cat_dog.h5")
print("Model saved to model/cnn_cat_dog.h5")
