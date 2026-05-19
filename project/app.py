# app.py

#import os
#import threading
#import webbrowser
#import traceback

#import numpy as np
from waitress import serve
from threading import Thread
import webview
#from webbrowser import open as open_browser_url
from traceback import format_exc
from flask import Flask, request, render_template_string
from predict_core import (
    #get_predictor,
    get_model_status,
    row_dict_from_flask_form,
    predict_from_dict
)


app = Flask(__name__)

MODEL_LOADED = False
MODEL_ERROR = None


# =========================
# MODEL LOAD STATUS
# =========================

def try_load_model():
    global MODEL_LOADED, MODEL_ERROR
    MODEL_LOADED, MODEL_ERROR = get_model_status()


try_load_model()


# =========================
# FORM PARSING
# =========================

#def parse_float(value, default=np.nan):
#    if value is None:
#        return default
#
#    value = str(value).strip().replace(",", ".")
#
#    if value == "":
#        return default
#
#    return float(value)
#
#
#def parse_int(value, default=np.nan):
#    if value is None:
#        return default
#
#    value = str(value).strip()
#
#    if value == "":
#        return default
#
#    return int(value)
#
#
#def row_dict_from_form(form):
#    return {
#        "compound": form.get("compound", "").strip(),
#
#        "volume_atom": parse_float(form.get("volume_atom")),
#        "density": parse_float(form.get("density")),
#        "energy_atom": parse_float(form.get("energy_atom")),
#        "enthalpy_formation_atom": parse_float(form.get("enthalpy_formation_atom")),
#
#        "Egap": parse_float(form.get("Egap")),
#        "Egap_type": form.get("Egap_type", "").strip(),
#
 #       "natoms": parse_int(form.get("natoms")),
 #       "nspecies": parse_int(form.get("nspecies")),
 #       "spacegroup_relax": parse_int(form.get("spacegroup_relax")),
 #   }


# =========================
# HTML
# =========================

HTML_TEMPLATE = """
<!doctype html>
<html lang="ru">
<head>
    <meta charset="utf-8">
    <title>AGL Predictor</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1050px;
            margin: 30px auto;
            padding: 0 20px;
            background: #f5f5f5;
            color: #222;
        }

        .card {
            background: white;
            padding: 22px;
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
            margin-bottom: 20px;
        }

        h1 {
            margin-top: 0;
        }

        .status-ok {
            color: #137333;
            font-weight: bold;
        }

        .status-bad {
            color: #b3261e;
            font-weight: bold;
        }

        .grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(260px, 1fr));
            gap: 14px 22px;
        }

        label {
            display: block;
            font-weight: bold;
            margin-bottom: 5px;
        }

        input, select {
            width: 100%;
            padding: 9px;
            border: 1px solid #bbb;
            border-radius: 6px;
            font-size: 14px;
            box-sizing: border-box;
        }

        button {
            margin-top: 20px;
            padding: 11px 18px;
            font-size: 15px;
            border: none;
            border-radius: 7px;
            background: #1a73e8;
            color: white;
            cursor: pointer;
        }

        button:hover {
            background: #1558b0;
        }

        table {
            border-collapse: collapse;
            width: 100%;
            background: white;
        }

        th, td {
            border: 1px solid #ddd;
            padding: 8px 10px;
            text-align: left;
        }

        th {
            background: #eeeeee;
        }

        .error {
            white-space: pre-wrap;
            color: #b3261e;
            background: #fff2f2;
            padding: 12px;
            border-radius: 8px;
            border: 1px solid #f0b8b8;
        }

        .hint {
            color: #666;
            font-size: 13px;
        }
    </style>
</head>
<body>

<div class="card">
    <h1>AGL Predictor</h1>

    {% if model_loaded %}
        <p class="status-ok">Статус модели: загружена</p>
    {% else %}
        <p class="status-bad">Статус модели: не загружена</p>
        {% if model_error %}
            <div class="error">{{ model_error }}</div>
        {% endif %}
    {% endif %}
</div>

<div class="card">
    <h2>Входные данные материала</h2>

    <form method="post">
        <div class="grid">
            <div>
                <label>compound</label>
                <input name="compound" value="{{ form_values.get('compound', 'Ac1H2') }}" required>
                <div class="hint">Например: Te2Zn2, Ac1H2, C1Nb1</div>
            </div>

            <div>
                <label>volume_atom</label>
                <input name="volume_atom" value="{{ form_values.get('volume_atom', '17.0642') }}" required>
            </div>

            <div>
                <label>density</label>
                <input name="density" value="{{ form_values.get('density', '7.42958') }}" required>
            </div>

            <div>
                <label>energy_atom</label>
                <input name="energy_atom" value="{{ form_values.get('energy_atom', '-4.18878') }}" required>
            </div>

            <div>
                <label>enthalpy_formation_atom</label>
                <input name="enthalpy_formation_atom" value="{{ form_values.get('enthalpy_formation_atom', '-0.566575') }}">
                <div class="hint">Можно оставить пустым</div>
            </div>

            <div>
                <label>Egap</label>
                <input name="Egap" value="{{ form_values.get('Egap', '0.0') }}" required>
            </div>

            <div>
                <label>Egap_type</label>
                <select name="Egap_type">
                    {% set egap_value = form_values.get('Egap_type', 'metal') %}
                    <option value="metal" {% if egap_value == 'metal' %}selected{% endif %}>metal</option>
                    <option value="insulator" {% if egap_value == 'insulator' %}selected{% endif %}>insulator</option>
                    <option value="semiconductor" {% if egap_value == 'semiconductor' %}selected{% endif %}>semiconductor</option>
                    <option value="insulator-direct" {% if egap_value == 'insulator-direct' %}selected{% endif %}>insulator-direct</option>
                    <option value="insulator-indirect" {% if egap_value == 'insulator-indirect' %}selected{% endif %}>insulator-indirect</option>
                </select>
            </div>

            <div>
                <label>natoms</label>
                <input name="natoms" value="{{ form_values.get('natoms', '3') }}" required>
            </div>

            <div>
                <label>nspecies</label>
                <input name="nspecies" value="{{ form_values.get('nspecies', '2') }}" required>
            </div>

            <div>
                <label>spacegroup_relax</label>
                <input name="spacegroup_relax" value="{{ form_values.get('spacegroup_relax', '225') }}" required>
            </div>
        </div>

        <button type="submit">Посчитать</button>
    </form>
</div>

{% if error %}
<div class="card">
    <h2>Ошибка</h2>
    <div class="error">{{ error }}</div>
</div>
{% endif %}

{% if result %}
<div class="card">
    <h2>Предсказание</h2>

    <table>
        <tr>
            <th>Параметр</th>
            <th>Значение</th>
        </tr>

        {% for key, value in result.items() %}
        <tr>
            <td>{{ key }}</td>
            <td>
                {% if value is number %}
                    {{ "%.8g"|format(value) }}
                {% else %}
                    {{ value }}
                {% endif %}
            </td>
        </tr>
        {% endfor %}
    </table>
</div>
{% endif %}

</body>
</html>
"""

# =========================
# ROUTES
# =========================

@app.route("/", methods=["GET", "POST"])
def index():
    global MODEL_LOADED, MODEL_ERROR

    result = None
    error = None
    form_values = {}

    if request.method == "POST":
        form_values = request.form.to_dict()

        if not MODEL_LOADED:
            try_load_model()

        if not MODEL_LOADED:
            error = "                   :\n" + str(MODEL_ERROR)
        else:
            try:
                row_dict = row_dict_from_flask_form(request.form)
                result = predict_from_dict(row_dict)

            except Exception:
                error = format_exc()

    return render_template_string(
        HTML_TEMPLATE,
        model_loaded=MODEL_LOADED,
        model_error=MODEL_ERROR,
        result=result,
        error=error,
        form_values=form_values,
    )


# =========================
# RUN
# =========================


HOST = "127.0.0.1"
PORT = 5000
URL = f"http://{HOST}:{PORT}"


def run_server():
    serve(
        app,
        host=HOST,
        port=PORT,
        threads=4
    )


if __name__ == "__main__":
    server_thread = Thread(target=run_server, daemon=True)
    server_thread.start()

    webview.create_window(
        title="AGL Predictor",
        url=URL,
        width=1100,
        height=850,
        resizable=True,
        confirm_close=True
    )

    webview.start()