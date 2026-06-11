# save_model.py
# Run this ONCE after training your CNN in the notebook to save the model.
# Then place the saved file at:  cat_dog_app/model/cnn_cat_dog.h5

import os

# ── Paste this at the end of your notebook (or run this script separately) ──

# Assumes `cnn` is your trained Keras model object
os.makedirs("model", exist_ok=True)
cnn.save("model/cnn_cat_dog.h5")
print("Model saved to model/cnn_cat_dog.h5")
