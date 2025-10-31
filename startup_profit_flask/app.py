import os
import json
import logging
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, abort

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, 'startup_profit_model.pkl')
META_PATH = os.path.join(APP_DIR, 'startup_profit_model_meta.json')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app and allow templates to be served from the app dir or the
parent_dir = os.path.abspath(os.path.join(APP_DIR, '..'))
app_templates_in_app = os.path.join(APP_DIR, 'templates')
app_templates_in_parent = os.path.join(parent_dir, 'templates')
if os.path.isdir(app_templates_in_app) and os.path.exists(os.path.join(app_templates_in_app, 'index.html')):
    template_folder = app_templates_in_app
elif os.path.isdir(app_templates_in_parent) and os.path.exists(os.path.join(app_templates_in_parent, 'index.html')):
    template_folder = app_templates_in_parent
elif os.path.exists(os.path.join(APP_DIR, 'index.html')):
    template_folder = APP_DIR
elif os.path.exists(os.path.join(parent_dir, 'index.html')):
    template_folder = parent_dir
else:
   
    template_folder = APP_DIR

app = Flask(__name__, template_folder=template_folder)

# Load model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Place startup_profit_model.pkl here.")

model = None
try:
    model = joblib.load(MODEL_PATH)  # Pipeline(preprocess + model)
except Exception as e:
   
    logger.exception("Failed to load model")
    raise RuntimeError(f"Failed to load model at {MODEL_PATH}: {e}") from e

# Try to read states used in training (from the pipeline's OneHotEncoder)
def get_states_from_model():
    try:
        ohe = model.named_steps['prep'].named_transformers_['cat']
        if hasattr(ohe, 'categories_'):
            return list(ohe.categories_[0])
    except Exception:
        logger.debug("Could not introspect model for trained states; using fallback.")
    
    return ['New York', 'California', 'Florida']

TRAINED_STATES = get_states_from_model()


def to_float(x, default=None):
    """Safely convert a value to float. Accepts strings with commas.

    Returns `default` if conversion fails or input is falsy.
    """
    if x is None:
        return default
    try:
        if isinstance(x, str):
            x = x.strip().replace(',', '')
            if x == '':
                return default
        return float(x)
    except Exception:
        return default

def validate_state(state):
    """Return True if state is known from training metadata."""
    if not state:
        return False
    return state in TRAINED_STATES

def build_input_df(rd_spend, admin_spend, mkt_spend, state):
   
    return pd.DataFrame([{
        'R&D Spend': rd_spend,
        'Administration': admin_spend,
        'Marketing Spend': mkt_spend,
        'State': state
    }])

@app.route('/', methods=['GET'])
def home():
    try:
        return render_template('index.html', states=TRAINED_STATES, prediction=None, error=None)
    except Exception as e:
        # Log full exception and return a simple diagnostic message so browser shows something useful.
        logger.exception("Failed to render index.html")
        # Provide brief info to the client (avoid stack traces in production)
        return (f"Internal Server Error while rendering template: {e}"), 500


@app.errorhandler(500)
def handle_500(err):
    # Log the original exception if available
    logger.exception("Unhandled exception: %s", err)
    return ("Internal Server Error"), 500

@app.route('/predict', methods=['POST'])
def predict():
    # Handle JSON and Form submissions
    if request.is_json:
        data = request.get_json() or {}
        rd   = to_float(data.get('R&D Spend') or data.get('rd_spend'))
        admin= to_float(data.get('Administration') or data.get('administration'))
        mkt  = to_float(data.get('Marketing Spend') or data.get('marketing_spend'))
        state= data.get('State') or data.get('state')
        if None in [rd, admin, mkt] or not state:
            return jsonify(error="Missing/invalid inputs"), 400
        if not validate_state(state):
            return jsonify(error=f"Unknown state '{state}'. Valid states: {TRAINED_STATES}"), 400
        X = build_input_df(rd, admin, mkt, state)
        try:
            pred = float(model.predict(X)[0])
        except Exception as e:
            logger.exception("Prediction failed for JSON request")
            return jsonify(error=f"Prediction failed: {e}"), 500
        return jsonify(predicted_profit=pred)
    else:
        rd    = to_float(request.form.get('rd_spend'))
        admin = to_float(request.form.get('admin_spend'))
        mkt   = to_float(request.form.get('marketing_spend'))
        state = request.form.get('state')

        if None in [rd, admin, mkt] or not state:
            return render_template('index.html', states=TRAINED_STATES,
                                   prediction=None, error="Please fill all fields with valid numbers.")

        if not validate_state(state):
            return render_template('index.html', states=TRAINED_STATES,
                                   prediction=None, error=f"Unknown state '{state}'. Please select a valid state.")

        X = build_input_df(rd, admin, mkt, state)
        try:
            pred = float(model.predict(X)[0])
        except Exception as e:
            logger.exception("Prediction failed for form submission")
           
            return render_template('index.html', states=TRAINED_STATES,
                                   prediction=None, error="Prediction failed. Please try different inputs or contact the administrator.")

        
        pred_display = f"{pred:,.2f}"
        return render_template('index.html', states=TRAINED_STATES, prediction=pred_display, error=None)

@app.route('/health', methods=['GET'])
def health():
    return jsonify(status="ok", states=TRAINED_STATES)

if __name__ == '__main__':
    # debug should be enabled explicitly via FLASK_DEBUG=1 in development
    debug_mode = os.environ.get('FLASK_DEBUG', '0') == '1'
    app.run(host='127.0.0.1', port=5000, debug=debug_mode)
