# app.py

from waitress import serve
from threading import Thread
import webview
from traceback import format_exc
from flask import Flask, request, render_template_string
from predict_core import (
    get_model_status,
    row_dict_from_flask_form,
    predict_from_dict
)


app = Flask(__name__)

MODEL_LOADED = False
MODEL_ERROR = None

RESULT_UNITS = {
    "compound": "",

    "agl_debye": "K",
    "agl_acoustic_debye": "K",
    "agl_gruneisen": "",

    "agl_heat_capacity_Cv_300K": "k<sub>B</sub>/cell",
    "agl_heat_capacity_Cp_300K": "k<sub>B</sub>/cell",

    "agl_thermal_conductivity_300K": "W/(m·K)",
    "agl_thermal_expansion_300K": "K<sup>-1</sup>",

    "agl_bulk_modulus_isothermal_300K": "GPa",
    "agl_bulk_modulus_static_300K": "GPa",

#    # если добавишь пересчёт теплоёмкости:
#    "agl_heat_capacity_Cv_300K_J_kgK": "J/(kg·K)",
#    "agl_heat_capacity_Cp_300K_J_kgK": "J/(kg·K)",
}

# =========================
# MODEL LOAD STATUS
# =========================

def try_load_model():
    global MODEL_LOADED, MODEL_ERROR
    MODEL_LOADED, MODEL_ERROR = get_model_status()


try_load_model()

def format_result_value(key, value):
    if not isinstance(value, (int, float)):
        return str(value)

    if key == "agl_thermal_expansion_300K":
        if value == 0:
            return "0"

        mantissa, exponent = f"{value:.8e}".split("e")
        mantissa = float(mantissa)
        exponent = int(exponent)

        return f"{mantissa:.8g} × 10<sup>{exponent}</sup>"

    return f"{value:.8g}"

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
        
        .input-with-unit {
            position: relative;
            width: 100%;
        }
        
        .input-with-unit input {
            padding-right: 95px;
        }
        
        .unit-suffix {
            position: absolute;
            right: 10px;
            top: 50%;
            transform: translateY(-50%);
            color: #666;
            font-size: 14px;
            pointer-events: none;
            user-select: none;
            white-space: nowrap;
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
                <label>Состав</label>
                <input name="compound" value="{{ form_values.get('compound', 'Ac1H2') }}" required>
                <div class="hint">Например: Te2Zn2, Ac1H2, C1Nb1</div>
            </div>

            <div>
                <label>Атомный Объём</label>
                <div class="input-with-unit">
                    <input name="volume_atom" value="{{ form_values.get('volume_atom', '17.0642') }}" required>
                    <span class="unit-suffix">Å<sup>3</sup>/atom</span>
                </div>
            </div>

            <div>
                <label>Плотность материала</label>
                <div class="input-with-unit">
                    <input name="density" value="{{ form_values.get('density', '7.42958') }}" required>
                    <span class="unit-suffix">g/cm<sup>3</sup></span>
                </div>
            </div>

            <div>
                <label>Энергия на атом</label>
                <div class="input-with-unit">
                    <input name="energy_atom" value="{{ form_values.get('energy_atom', '-4.18878') }}" required>
                    <span class="unit-suffix">eV/atom</span>
                </div>
            </div>

            <div>
                <label>Энтальпия образования на атом</label>
                <div class="input-with-unit">
                    <input name="enthalpy_formation_atom" value="{{ form_values.get('enthalpy_formation_atom', '-0.566575') }}">
                    <span class="unit-suffix">eV/atom</span>
                </div>
                <div class="hint">Можно оставить пустым</div>
            </div>

            <div>
                <label>Ширина запрещённой зоны</label>
                <div class="input-with-unit">
                    <input name="Egap" value="{{ form_values.get('Egap', '0.0') }}" required>
                    <span class="unit-suffix">eV</span>
                </div>
            </div>

            <div>
                <label>Тип запрещённой зоны</label>
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
                <label>Число атомов</label>
                <input name="natoms" value="{{ form_values.get('natoms', '3') }}" required>
            </div>

            <div>
                <label>Число элементов</label>
                <input name="nspecies" value="{{ form_values.get('nspecies', '2') }}" required>
            </div>

            <div>
                <label>Пространственная группа</label>
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
                    {{ format_result_value(key, value)|safe }}
                    {% if units.get(key, "") %}
                        {{ units.get(key, "")|safe }}
                    {% endif %}
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
        units=RESULT_UNITS,
        format_result_value=format_result_value,
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
