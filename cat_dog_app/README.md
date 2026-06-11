# Cat vs Dog Classifier — Flask App

## Folder structure
```
cat_dog_app/
├── app.py               # Flask backend
├── save_model.py        # Helper to save your trained model
├── requirements.txt
├── model/
│   └── cnn_cat_dog.h5   # ← place your saved model here
├── static/
│   └── uploads/         # uploaded images stored here temporarily
└── templates/
    └── index.html       # UI
```

## Setup

```bash
pip install -r requirements.txt
```

## Step 1 — Save your trained model
After training in the notebook, add this at the end:
```python
cnn.save("cat_dog_app/model/cnn_cat_dog.h5")
```
Or run `save_model.py` from inside the notebook environment.

## Step 2 — Run the app
```bash
cd cat_dog_app
python app.py
```
Then open **http://127.0.0.1:5000** in your browser.

## Usage
1. Drag & drop or click to upload a cat/dog image (JPG or PNG, max 5 MB).
2. Click **Classify Image**.
3. The result shows the predicted class and confidence percentage.
