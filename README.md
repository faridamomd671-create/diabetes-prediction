# 🩺 Diabetes Prediction

Machine learning project that predicts whether a patient is likely to have diabetes
from 8 medical features (Pima Indians Diabetes dataset), with an interactive **Streamlit** app.

## Features
- Data exploration (correlation, distributions, outliers)
- 4 models compared: Logistic Regression, SVC, Random Forest, Gradient Boosting
- Preprocessing pipeline: missing-value handling (0 → median) + scaling
- Metrics: Accuracy, Precision, Recall, F1, ROC-AUC
- Streamlit app: live prediction with risk level, model comparison, data explorer

## Project structure
```
├── app.py               # Streamlit app
├── train_model.py       # trains the models and saves the best one
├── requirements.txt
├── data/diabetes.csv    # dataset
├── models/              # saved model + metrics (generated)
└── notebook/            # original Colab notebook
```

## Run locally
```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

## Results
Fill in your final table here after running `train_model.py`.

## Screenshots
Add a screenshot of the app here: `![app](screenshot.png)`

## Disclaimer
Educational project only. Not a medical diagnosis tool.

## Author
Farida — AI and Data Analysis Engineering, Sphinx University
