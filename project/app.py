# app.py

from datetime import datetime
from argparse import ArgumentParser
from threading import Thread
from traceback import format_exc

from waitress import serve
from flask import Flask, request, render_template_string

from os.path import abspath, dirname, join, isdir, basename, isfile, getsize

import sys
import pandas as pd

from predict_core import (
    get_model_status,
    row_dict_from_flask_form,
    predict_from_dict
)

def get_project_dir():
    if getattr(sys, "frozen", False):
        return dirname(sys.executable)

    source_dir = dirname(abspath(__file__))

    if basename(source_dir) == "src":
        return dirname(source_dir)

    return source_dir


def get_resource_dir():
    project_dir = get_project_dir()
    candidates = []
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        candidates.append(join(sys._MEIPASS, "resources"))
        candidates.append(sys._MEIPASS)
    candidates.extend([
        join(project_dir, "resources"),
        join(project_dir, "_internal", "resources"),
        project_dir,
    ])
    for path in candidates:
        if isdir(path):
            return path
    return project_dir

def get_output_dir():
    if getattr(sys, "frozen", False):
        return dirname(sys.executable)
    source_dir = dirname(abspath(__file__))
    # Linux/server layout:
    # AGLPredictor_web/_internal/src/app.py
    # results.csv должен быть в AGLPredictor_web/
    if basename(source_dir) == "src":
        parent_dir = dirname(source_dir)
        if basename(parent_dir) == "_internal":
            return dirname(parent_dir)
        return parent_dir
    return source_dir

app = Flask(__name__)

MODEL_LOADED = False
MODEL_ERROR = None

AFLOW_CSV_NAME = "aflow_agl.csv"
AFLOW_DF_CACHE = None
RESULTS_CSV_NAME = "results.csv"
BATCH_RESULTS_CSV_NAME = "batch_results.csv"
INPUT_COLUMNS_REQUIRED = [
    "compound",
    "volume_atom",
    "density",
    "energy_atom",
    "Egap",
    "Egap_type",
    "spacegroup_relax",
]
INPUT_COLUMNS_OPTIONAL = [
    "enthalpy_formation_atom",
]
INPUT_COLUMNS_ALL = INPUT_COLUMNS_REQUIRED + INPUT_COLUMNS_OPTIONAL


def load_aflow_df():
    global AFLOW_DF_CACHE
    if AFLOW_DF_CACHE is None:
        csv_path = join(get_resource_dir(), AFLOW_CSV_NAME)
        AFLOW_DF_CACHE = pd.read_csv(csv_path)
        if "compound" not in AFLOW_DF_CACHE.columns:
            raise ValueError(f"В файле {AFLOW_CSV_NAME} нет столбца compound")
        AFLOW_DF_CACHE["compound"] = AFLOW_DF_CACHE["compound"].astype(str).str.strip()
    return AFLOW_DF_CACHE

def get_aflow_check_values(compound, result_keys):
    df = load_aflow_df()
    compound = str(compound).strip()
    rows = df[df["compound"] == compound]
    if rows.empty:
        return None, f"В файле {AFLOW_CSV_NAME} не найден compound: {compound}"
    row = rows.iloc[0]
    check_values = {}
    for key in result_keys:
        if key == "compound":
            check_values[key] = row.get("compound", compound)
            continue
        if key in row.index:
            value = pd.to_numeric(row[key], errors="coerce")
            if pd.isna(value):
                check_values[key] = None
            else:
                check_values[key] = float(value)
        else:
            check_values[key] = None
    return check_values, None

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

RESULT_LABELS = {
    "compound": "Состав",

    "agl_debye": "Температура Дебая",
    "agl_acoustic_debye": "Акустическая температура Дебая",
    "agl_gruneisen": "Параметр Грюнайзена",

    "agl_heat_capacity_Cv_300K": "Теплоёмкость при постоянном объёме при 300 K",
    "agl_heat_capacity_Cp_300K": "Теплоёмкость при постоянном давлении при 300 K",

    "agl_thermal_conductivity_300K": "Теплопроводность при 300 K",
    "agl_thermal_expansion_300K": "Коэффициент теплового расширения при 300 K",

    "agl_bulk_modulus_isothermal_300K": "Изотермический модуль объёмного сжатия при 300 K",
    "agl_bulk_modulus_static_300K": "Статический модуль объёмного сжатия",
}

output_order = [
    "agl_debye",
    "agl_acoustic_debye",
    "agl_gruneisen",
    "agl_heat_capacity_Cp_300K",
    "agl_heat_capacity_Cv_300K",
    "agl_thermal_conductivity_300K",
    "agl_thermal_expansion_300K",
    "agl_bulk_modulus_isothermal_300K",
    "agl_bulk_modulus_static_300K",
]

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

def csv_safe_value(value):
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    try:
        if hasattr(value, "item"):
            return value.item()
    except Exception:
        pass

    return value


def get_results_csv_columns():
    prediction_cols = [
        f"pred_{key}"
        for key in output_order
    ]

    check_cols = [
        f"check_{key}"
        for key in output_order
    ]

    return (
        ["timestamp", "source", "row_number"]
        + INPUT_COLUMNS_ALL
        + prediction_cols
        + check_cols
    )


def make_results_csv_row(row_dict, result, source="manual", row_number=None, check_values=None):
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": source,
        "row_number": "" if row_number is None else row_number,
    }

    for key in INPUT_COLUMNS_ALL:
        row[key] = csv_safe_value(row_dict.get(key, ""))

    for key in output_order:
        row[f"pred_{key}"] = csv_safe_value(result.get(key, ""))

    for key in output_order:
        if check_values:
            row[f"check_{key}"] = csv_safe_value(check_values.get(key, ""))
        else:
            row[f"check_{key}"] = ""

    return row


def append_prediction_to_results_csv(row_dict, result, source="manual", row_number=None, check_values=None):
    output_dir = get_output_dir()
    csv_path = join(output_dir, RESULTS_CSV_NAME)

    columns = get_results_csv_columns()

    new_row = make_results_csv_row(
        row_dict=row_dict,
        result=result,
        source=source,
        row_number=row_number,
        check_values=check_values
    )

    new_df = pd.DataFrame([new_row], columns=columns)

    file_exists = isfile(csv_path) and getsize(csv_path) > 0

    new_df.to_csv(
        csv_path,
        mode="a",
        header=not file_exists,
        index=False,
        encoding="utf-8-sig"
    )

    return csv_path


def save_batch_results_csv(rows):
    output_dir = get_output_dir()
    csv_path = join(output_dir, BATCH_RESULTS_CSV_NAME)

    pd.DataFrame(rows).to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig"
    )

    return csv_path


def predict_batch_from_file(file_storage):
    if file_storage is None or file_storage.filename == "":
        raise ValueError("Файл не выбран")

    df_input = pd.read_csv(
        file_storage,
        sep=None,
        engine="python"
    )

    df_input.columns = [str(c).strip() for c in df_input.columns]

    missing_cols = [
        col for col in INPUT_COLUMNS_REQUIRED
        if col not in df_input.columns
    ]

    if missing_cols:
        raise ValueError(
            "В файле не хватает обязательных столбцов: "
            + ", ".join(missing_cols)
        )

    for col in INPUT_COLUMNS_OPTIONAL:
        if col not in df_input.columns:
            df_input[col] = ""

    batch_rows = []
    batch_errors = []

    for idx, row in df_input.iterrows():
        row_number = idx + 2

        row_dict = {
            col: row.get(col, "")
            for col in INPUT_COLUMNS_ALL
        }

        try:
            result = predict_from_dict(row_dict)

            append_prediction_to_results_csv(
                row_dict=row_dict,
                result=result,
                source=f"batch file: {file_storage.filename}",
                row_number=row_number
            )

            output_row = {}

            for col in INPUT_COLUMNS_ALL:
                output_row[col] = row_dict.get(col, "")

            for key in output_order:
                output_row[key] = result.get(key, "")
                output_row[f"pred_{key}"] = result.get(key, "")

            batch_rows.append(output_row)

        except Exception as e:
            batch_errors.append({
                "row": row_number,
                "compound": row_dict.get("compound", ""),
                "error": str(e),
            })

    batch_csv_path = None

    if batch_rows:
        batch_csv_path = save_batch_results_csv(batch_rows)

    return batch_rows, batch_errors, batch_csv_path








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
        .check-button {
            background: #32a852;
            margin-left: 10px;
        }
        
        .check-button:hover {
            background: #278541;
        }
        .table-scroll {
            width: 100%;
            overflow-x: auto;
            overflow-y: hidden;
        }

        .table-scroll table {
            min-width: 100%;
        }

        .batch-result-card {
            box-sizing: border-box;

            /* Не меньше обычной карточки */
            min-width: 100%;

            /* Расширяется по таблице, но не шире окна минус 40px */
            width: fit-content;
            max-width: calc(100vw - 40px);

            /* Центрирование относительно окна */
            margin-left: 50%;
            transform: translateX(-50%);
        }

        .batch-result-card .table-scroll {
            width: 100%;
            max-width: 100%;
            overflow: auto;
        }

        .batch-table {
            width: max-content;
            min-width: 100%;
            table-layout: auto;
            font-size: 13px;
        }

        .batch-table td {
            white-space: nowrap;
            word-break: normal;
            overflow-wrap: normal;
            vertical-align: top;
        }

        .batch-table th {
            width: 1%;
            max-width: 110px;
            white-space: normal;
            word-break: break-word;
            overflow-wrap: anywhere;
            vertical-align: top;
        }

        .batch-table td:first-child,
        .batch-table th:first-child {
            min-width: 90px;
            position: sticky;
            left: 0;
            background: white;
            z-index: 1;
        }

        .batch-table th:first-child {
            background: #eeeeee;
            z-index: 2;
        }

        .batch-scroll {
            max-height: 650px;
            overflow: auto;
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
                <div class="hint">Например: Te2Zn2, Ac1H2, C1Nb1, требуется ввести в формате с числом атомов</div>
            </div>

            <div>
                <label>Объём на атом</label>
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
                    <option value="metal" {% if egap_value == 'metal' %}selected{% endif %}>Металл</option>
                    <option value="insulator" {% if egap_value == 'insulator' %}selected{% endif %}>Изолятор</option>
                    <option value="semiconductor" {% if egap_value == 'semiconductor' %}selected{% endif %}>Полупроводник</option>
                    <option value="insulator-direct" {% if egap_value == 'insulator-direct' %}selected{% endif %}>Изолятор с прямой запрещённой зоной</option>
                    <option value="insulator-indirect" {% if egap_value == 'insulator-indirect' %}selected{% endif %}>Изолятор с непрямой запрещённой зоной</option>
                </select>
            </div>

            <div>
                <label>Пространственная группа</label>
                <input name="spacegroup_relax" value="{{ form_values.get('spacegroup_relax', '225') }}" required>
            </div>
        </div>

        <button type="submit" name="action" value="predict">Посчитать</button>
        <button type="submit" name="action" value="check" class="check-button">Проверить</button>
    </form>
</div>

<div class="card">
    <h2>Пакетное предсказание из файла</h2>

    <form method="post" enctype="multipart/form-data">
        <input type="hidden" name="action" value="batch">

        <label>CSV-файл с материалами</label>
        <input type="file" name="batch_file" accept=".csv,.txt" required>

        <div class="hint">
            Обязательные столбцы:
            compound, volume_atom, density, energy_atom, Egap, Egap_type, spacegroup_relax.
            Дополнительно можно указать enthalpy_formation_atom.
        </div>

        <button type="submit">Посчитать файл</button>
    </form>
</div>

{% if error %}
<div class="card">
    <h2>Ошибка</h2>
    <div class="error">{{ error }}</div>
</div>
{% endif %}

{% if check_message %}
<div class="card">
    <h2>Проверка</h2>
    <div class="error">{{ check_message }}</div>
</div>
{% endif %}

{% if saved_message %}
<div class="card">
    <h2>Сохранение</h2>
    <p>{{ saved_message }}</p>
</div>
{% endif %}

{% if result %}
<div class="card">
    <h2>Предсказание</h2>

    <table>
        <tr>
            <th>Параметр</th>
            <th>Значение</th>
            {% if check_requested %}
                <th>Проверка</th>
            {% endif %}
        </tr>
    
        {% for key, value in result.items() %}
        <tr>
            <td>{{ labels.get(key, key) }}</td>
    
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
    
            {% if check_requested %}
            <td>
                {% if check_values and key in check_values and check_values.get(key) is not none %}
                    {% set check_value = check_values.get(key) %}
    
                    {% if check_value is number %}
                        {{ format_result_value(key, check_value)|safe }}
                        {% if units.get(key, "") %}
                            {{ units.get(key, "")|safe }}
                        {% endif %}
                    {% else %}
                        {{ check_value }}
                    {% endif %}
                {% else %}
                нет данных
                {% endif %}
            </td>
            {% endif %}
        </tr>
        {% endfor %}
    </table>
</div>
{% endif %}

{% if batch_result %}
<div class="card batch-result-card">
    <h2>Пакетное предсказание</h2>

    <p>Обработано строк: {{ batch_result|length }}</p>

    <div class="table-scroll batch-scroll">
    <table class="batch-table">
        <tr>
            <th>Состав</th>
            {% for key in output_order %}
                <th>{{ labels.get(key, key) }}</th>
            {% endfor %}
        </tr>

        {% for row in batch_result %}
        <tr>
            <td>{{ row.get("compound", "") }}</td>

            {% for key in output_order %}
            <td>
                {% if row.get(key) is number %}
                    {{ format_result_value(key, row.get(key))|safe }}
                    {% if units.get(key, "") %}
                        {{ units.get(key, "")|safe }}
                    {% endif %}
                {% else %}
                    {{ row.get(key, "") }}
                {% endif %}
            </td>
            {% endfor %}
        </tr>
        {% endfor %}
    </table>
    </div>
</div>
{% endif %}

{% if batch_errors %}
<div class="card">
    <h2>Ошибки пакетного предсказания</h2>

    <table>
        <tr>
            <th>Строка</th>
            <th>Состав</th>
            <th>Ошибка</th>
        </tr>

        {% for err in batch_errors %}
        <tr>
            <td>{{ err.get("row") }}</td>
            <td>{{ err.get("compound") }}</td>
            <td>{{ err.get("error") }}</td>
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

    check_requested = False
    check_values = None
    check_message = None

    batch_result = None
    batch_errors = None
    saved_message = None

    if request.method == "POST":
        action = request.form.get("action", "predict")

        if not MODEL_LOADED:
            try_load_model()

        if not MODEL_LOADED:
            error = "Модель не загружена:\n" + str(MODEL_ERROR)
        else:
            try:
                if action == "batch":
                    batch_file = request.files.get("batch_file")

                    batch_result, batch_errors, batch_csv_path = predict_batch_from_file(
                        batch_file
                    )

                    output_dir = get_output_dir()
                    results_csv_path = join(output_dir, RESULTS_CSV_NAME)

                    saved_parts = [
                        f"Результаты дописаны в {results_csv_path}"
                    ]

                    if batch_csv_path:
                        saved_parts.append(
                            f"Пакетный CSV сохранён в {batch_csv_path}"
                        )

                    saved_message = "\n".join(saved_parts)

                else:
                    form_values = request.form.to_dict()
                    check_requested = action == "check"

                    row_dict = row_dict_from_flask_form(request.form)
                    result = predict_from_dict(row_dict)

                    if check_requested:
                        check_values, check_message = get_aflow_check_values(
                            row_dict.get("compound", ""),
                            result.keys()
                        )

                    results_csv_path = append_prediction_to_results_csv(
                        row_dict=row_dict,
                        result=result,
                        source="manual form",
                        check_values=check_values
                    )
                    
                    saved_message = f"Результат сохранён в {results_csv_path}"

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
        labels=RESULT_LABELS,
        check_requested=check_requested,
        check_values=check_values,
        check_message=check_message,
        batch_result=batch_result,
        batch_errors=batch_errors,
        saved_message=saved_message,
        output_order=output_order,
    )

# =========================
# RUN MODES
# =========================

def parse_args():
    parser = ArgumentParser(description="AGL Predictor")

    parser.add_argument(
        "-server",
        "--server",
        action="store_true",
        help="Запустить только веб-сервер без окна pywebview"
    )

    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Адрес сервера. Для доступа из сети: 0.0.0.0"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Порт сервера"
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Количество потоков Waitress"
    )

    return parser.parse_args()


def run_server(host, port, threads):
    serve(
        app,
        host=host,
        port=port,
        threads=threads
    )


def run_desktop(host, port, threads):
    server_thread = Thread(
        target=run_server,
        args=(host, port, threads),
        daemon=True
    )
    server_thread.start()

    # Для окна нельзя нормально открывать 0.0.0.0
    window_host = "127.0.0.1" if host == "0.0.0.0" else host
    url = f"http://{window_host}:{port}"

    import webview

    webview.create_window(
        title="AGL Predictor",
        url=url,
        width=1100,
        height=850,
        resizable=True,
        confirm_close=True
    )

    webview.start()


if __name__ == "__main__":
    args = parse_args()

    if args.server:
        print(f"AGL Predictor server started: http://{args.host}:{args.port}")
        run_server(
            host=args.host,
            port=args.port,
            threads=args.threads
        )
    else:
        run_desktop(
            host=args.host,
            port=args.port,
            threads=args.threads
        )
