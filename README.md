# Classical-ML-on-Structured-Data

Startup Profit Prediction (Flask + scikit-learn)
Predict startup profit from R&D, Administration, Marketing spend, and State using a trained scikit-learn pipeline. Includes a simple Flask web UI and a JSON API.

Demo features

Web form to enter spends and state
Uses a serialized model (startup_profit_model.pkl) with preprocessing inside the pipeline
JSON endpoint for programmatic predictions
Reproducible environment and training notes
Table of contents
Project structure
Tech stack
Dataset
Quick start (run locally)
API usage
Train or regenerate the model
Notes on compatibility
Troubleshooting (FAQ)
Limitations and future work
Acknowledgements
Hinglish quick start
Project structure
text

startup_profit_flask/
├── app.py
├── requirements.txt
├── startup_profit_model.pkl              # trained sklearn Pipeline (required)
├── startup_profit_model_meta.json        # metadata (optional but recommended)
└── templates/
    └── index.html
Optional (if you include your notebook or dataset while developing):

notebooks/Startup_Profit_Training.ipynb
data/50_Startups.csv
Tech stack
Python 3.8–3.12
Flask (web framework)
scikit-learn (preprocessing + model)
pandas, numpy, joblib
Bootstrap (simple styling in the HTML)
