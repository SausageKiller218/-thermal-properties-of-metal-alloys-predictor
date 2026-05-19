from os import environ
from os.path import abspath, dirname, join
environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
environ["AUTOGRAPH_VERBOSITY"] = "0"
import sys
from json import load as json_load
from joblib import load as joblib_load
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from re import findall
from warnings import filterwarnings
from mendeleev import element
import xgboost
#from xgboost import XGBRegressor

xgboost.set_config(verbosity=0)

def get_base_dir():
    if getattr(sys, "frozen", False):
        if hasattr(sys, "_MEIPASS"):
            return sys._MEIPASS
        return dirname(sys.executable)
    return dirname(abspath(__file__))
BASE_DIR = get_base_dir()
ARTIFACT_DIR = join(BASE_DIR, "model")
filterwarnings(
    "ignore",
    message=".*has multiple allotropes.*",
    category=UserWarning
)
def input_float(prompt, allow_empty=True):
    value = input(prompt).strip().replace(",", ".")

    if value == "" and allow_empty:
        return np.nan

    try:
        return float(value)
    except ValueError:
        print("      :                   .")
        return input_float(prompt, allow_empty=allow_empty)
def input_int(prompt, allow_empty=False):
    value = input(prompt).strip()

    if value == "" and allow_empty:
        return np.nan

    try:
        return int(value)
    except ValueError:
        print("      :                         .")
        return input_int(prompt, allow_empty=allow_empty)

def input_df(row_dict, print_input=False):
    required_cols = [
        "compound",
        "volume_atom",
        "density",
        "energy_atom",
        "enthalpy_formation_atom",
        "Egap",
        "Egap_type",
        "spacegroup_relax",
    ]
    row = {}
    for col in required_cols:
        row[col] = row_dict.get(col, np.nan)
    df_new = pd.DataFrame([row])

    compound = df_new["compound"].iloc[0]
    comp = parse_formula(compound)
    if len(comp) == 0:
        raise ValueError(
            "Поле 'Состав' должно быть химической формулой, например Ac1H2, Te2Zn2, C1Nb1. "
            f"Сейчас введено: {compound}"
        )
    natoms = sum(comp.values())
    nspecies = len(comp)
    if abs(natoms - round(natoms)) < 1e-9:
        natoms = int(round(natoms))
    df_new["natoms"] = natoms
    df_new["nspecies"] = nspecies
    df_new["compound"] = df_new["compound"].astype(str).str.strip()
    df_new["Egap_type"] = df_new["Egap_type"].astype(str).str.strip()
    numeric_cols = [
        "volume_atom",
        "density",
        "energy_atom",
        "enthalpy_formation_atom",
        "Egap",
        "natoms",
        "nspecies",
        "spacegroup_relax",
    ]
    for col in numeric_cols:
        df_new[col] = pd.to_numeric(df_new[col], errors="coerce")
    if print_input:
        print(df_new.to_string(index=False))
    return df_new

##############################################################
def row_dict_from_flask_form(form):
    def get_str(name, default=""):
        return form.get(name, default).strip()

    def get_float(name, default=np.nan):
        value = form.get(name, "").strip().replace(",", ".")

        if value == "":
            return default

        return float(value)

    def get_int(name, default=np.nan):
        value = form.get(name, "").strip()

        if value == "":
            return default

        return int(value)

    return {
        "compound": get_str("compound"),

        "volume_atom": get_float("volume_atom"),
        "density": get_float("density"),
        "energy_atom": get_float("energy_atom"),
        "enthalpy_formation_atom": get_float("enthalpy_formation_atom"),

        "Egap": get_float("Egap"),
        "Egap_type": get_str("Egap_type"),

        "spacegroup_relax": get_int("spacegroup_relax"),
    }

def load_saved_model():
    BASE_DIR = get_base_dir()
    ARTIFACT_DIR = join(BASE_DIR, "model")
    imputer = joblib_load(join(ARTIFACT_DIR, "imputer.joblib"))
    scaler = joblib_load(join(ARTIFACT_DIR, "scaler.joblib"))
    feature_cols = joblib_load(join(ARTIFACT_DIR, "feature_cols.joblib"))
    target_cols = joblib_load(join(ARTIFACT_DIR, "target_cols.joblib"))
    xgb_models = joblib_load(join(ARTIFACT_DIR, "xgb_models.joblib"))
    blend_weights = joblib_load(join(ARTIFACT_DIR, "blend_weights.joblib"))
    for target in xgb_models:
        try:
            xgb_models[target]["model"].set_params(
                device="cpu",
                verbosity=0
            )
        except Exception:
            pass

    loaded_keras_items = []
    with open(join(ARTIFACT_DIR, "keras_metadata.json"), "r", encoding="utf-8") as f:
        keras_metadata = json_load(f)
    for meta in keras_metadata:
        model = keras.models.load_model(
            join(ARTIFACT_DIR, meta["model_path"]),
            compile=False
        )   
        loaded_keras_items.append({
            "model": model,
            "targets": meta["targets"],
            "use_log_targets": meta["use_log_targets"],
            "y_mean": meta["y_mean"],
            "y_std": meta["y_std"],
            "kind": meta["kind"],
            "seed": meta["seed"],
        })
    print("                ")
    print("Targets:", target_cols)
    print("Feature count:", len(feature_cols))
    return imputer, scaler, feature_cols, target_cols, xgb_models, blend_weights, loaded_keras_items

def parse_formula(formula):
    if pd.isna(formula):
        return {}

    formula = str(formula).strip().replace(" ", "")
    parts = findall(r"([A-Z][a-z]?)([0-9]*\.?[0-9]*)", formula)

    comp = {}

    for el, amount in parts:
        amount = float(amount) if amount != "" else 1.0
        comp[el] = comp.get(el, 0.0) + amount

    return comp

_element_cache = {}

def safe_float(x):
    try:
        if x is None:
            return np.nan
        return float(x)
    except Exception:
        return np.nan

def get_element_props(symbol):
    if symbol in _element_cache:
        return _element_cache[symbol]

    try:
        el = element(symbol)

        props = {
            "Z": safe_float(getattr(el, "atomic_number", np.nan)),
            "atomic_weight": safe_float(getattr(el, "atomic_weight", np.nan)),
            "period": safe_float(getattr(el, "period", np.nan)),
            "en_pauling": safe_float(getattr(el, "en_pauling", np.nan)),
            "atomic_radius": safe_float(getattr(el, "atomic_radius", np.nan)),
            "covalent_radius": safe_float(getattr(el, "covalent_radius_pyykko", np.nan)),
            "melting_point": safe_float(getattr(el, "melting_point", np.nan)),
            "specific_heat": safe_float(getattr(el, "specific_heat", np.nan)),
        }

        try:
            props["ionization_energy"] = safe_float(el.ionenergies.get(1, np.nan))
        except Exception:
            props["ionization_energy"] = np.nan

    except Exception as e:
        print("MENDELEEV ERROR:", symbol, repr(e))

        props = {
            "Z": np.nan,
            "atomic_weight": np.nan,
            "period": np.nan,
            "en_pauling": np.nan,
            "atomic_radius": np.nan,
            "covalent_radius": np.nan,
            "melting_point": np.nan,
            "specific_heat": np.nan,
            "ionization_energy": np.nan,
        }

    _element_cache[symbol] = props
    return props

def weighted_avg_abs_dev(vals, weights, mean):
    return np.sum(weights * np.abs(vals - mean))

def composition_features(formula):
    comp = parse_formula(formula)

    if len(comp) == 0:
        return {}

    elements = list(comp.keys())
    amounts = np.array(list(comp.values()), dtype=float)

    total_atoms = amounts.sum()
    fractions = amounts / (total_atoms + 1e-12)

    feats = {}

    feats["comp_n_elements"] = len(elements)
    feats["comp_total_atoms"] = total_atoms
    feats["comp_max_fraction"] = fractions.max()
    feats["comp_fraction_l2"] = np.sqrt(np.sum(fractions ** 2))
    feats["comp_config_entropy"] = -np.sum(fractions * np.log(fractions + 1e-12))

    prop_names = [
        "Z",
        "atomic_weight",
        "period",
        "en_pauling",
        "atomic_radius",
        "covalent_radius",
        "melting_point",
        "specific_heat",
        "ionization_energy",
    ]

    prop_values = {p: [] for p in prop_names}

    for el_symbol in elements:
        props = get_element_props(el_symbol)

        for p in prop_names:
            prop_values[p].append(props[p])

    for p in prop_names:
        vals = np.array(prop_values[p], dtype=float)
        valid = ~np.isnan(vals)

        if valid.sum() == 0:
            feats[f"{p}_mean_w"] = np.nan
            feats[f"{p}_min"] = np.nan
            feats[f"{p}_max"] = np.nan
            feats[f"{p}_range"] = np.nan
            feats[f"{p}_std_w"] = np.nan
            feats[f"{p}_avg_abs_dev_w"] = np.nan
            continue

        vals_valid = vals[valid]
        fr_valid = fractions[valid]
        fr_valid = fr_valid / (fr_valid.sum() + 1e-12)

        mean_w = np.sum(vals_valid * fr_valid)
        var_w = np.sum(fr_valid * (vals_valid - mean_w) ** 2)

        feats[f"{p}_mean_w"] = mean_w
        feats[f"{p}_min"] = vals_valid.min()
        feats[f"{p}_max"] = vals_valid.max()
        feats[f"{p}_range"] = vals_valid.max() - vals_valid.min()
        feats[f"{p}_std_w"] = np.sqrt(var_w)
        feats[f"{p}_avg_abs_dev_w"] = weighted_avg_abs_dev(vals_valid, fr_valid, mean_w)

    return feats

def make_features_for_prediction(df_raw):
    df_fe = df_raw.copy()

    numeric_base_cols = [
        "volume_atom",
        "volume_cell",
        "density",
        "energy_atom",
        "enthalpy_formation_atom",
        "Egap",
        "natoms",
        "nspecies",
        "spacegroup_relax",
    ]

    for col in numeric_base_cols:
        if col in df_fe.columns:
            df_fe[col] = pd.to_numeric(df_fe[col], errors="coerce")

    comp_feat_df = pd.DataFrame(
        df_fe["compound"].apply(composition_features).tolist(),
        index=df_fe.index
    )

    df_fe = pd.concat([df_fe, comp_feat_df], axis=1)

    all_elements = set()

    for formula in df_fe["compound"]:
        all_elements.update(parse_formula(formula).keys())

    all_elements = sorted(all_elements)

    element_fraction_rows = []

    for formula in df_fe["compound"]:
        comp = parse_formula(formula)
        row = {f"el_{el}": 0.0 for el in all_elements}

        if len(comp) > 0:
            total = sum(comp.values())

            for el, amount in comp.items():
                row[f"el_{el}"] = amount / (total + 1e-12)

        element_fraction_rows.append(row)

    element_fraction_df = pd.DataFrame(element_fraction_rows, index=df_fe.index)
    df_fe = pd.concat([df_fe, element_fraction_df], axis=1)

    eps = 1e-9

    def add_if_cols_exist(new_col, cols, func):
        if set(cols).issubset(df_fe.columns):
            try:
                df_fe[new_col] = func(df_fe)
            except Exception:
                df_fe[new_col] = np.nan

    add_if_cols_exist("inv_volume_atom", ["volume_atom"], lambda d: 1.0 / (d["volume_atom"] + eps))
    add_if_cols_exist("log_volume_atom", ["volume_atom"], lambda d: np.log1p(d["volume_atom"].clip(lower=0)))
    add_if_cols_exist("volume_atom_pow_minus_1_3", ["volume_atom"], lambda d: 1.0 / np.power(d["volume_atom"] + eps, 1.0 / 3.0))
    add_if_cols_exist("volume_atom_pow_minus_2_3", ["volume_atom"], lambda d: 1.0 / np.power(d["volume_atom"] + eps, 2.0 / 3.0))
    add_if_cols_exist("inv_density", ["density"], lambda d: 1.0 / (d["density"] + eps))

    add_if_cols_exist("abs_energy_atom", ["energy_atom"], lambda d: d["energy_atom"].abs())
    add_if_cols_exist("energy_density_proxy", ["energy_atom", "volume_atom"], lambda d: d["energy_atom"].abs() / (d["volume_atom"] + eps))
    add_if_cols_exist("energy_density_signed", ["energy_atom", "volume_atom"], lambda d: d["energy_atom"] / (d["volume_atom"] + eps))

    add_if_cols_exist("density_div_atomic_weight", ["density", "atomic_weight_mean_w"], lambda d: d["density"] / (d["atomic_weight_mean_w"] + eps))
    add_if_cols_exist("Z_div_atomic_weight", ["Z_mean_w", "atomic_weight_mean_w"], lambda d: d["Z_mean_w"] / (d["atomic_weight_mean_w"] + eps))
    add_if_cols_exist("radius_div_volume_atom", ["atomic_radius_mean_w", "volume_atom"], lambda d: d["atomic_radius_mean_w"] / (d["volume_atom"] + eps))
    add_if_cols_exist("volume_div_radius3", ["volume_atom", "atomic_radius_mean_w"], lambda d: d["volume_atom"] / (np.power(d["atomic_radius_mean_w"], 3) + eps))

    add_if_cols_exist("melting_div_atomic_weight", ["melting_point_mean_w", "atomic_weight_mean_w"], lambda d: d["melting_point_mean_w"] / (d["atomic_weight_mean_w"] + eps))

    if "Egap" in df_fe.columns:
        df_fe["Egap"] = pd.to_numeric(df_fe["Egap"], errors="coerce")
        df_fe["log_Egap"] = np.log1p(df_fe["Egap"].clip(lower=0))
        df_fe["Egap_is_zero"] = (df_fe["Egap"].fillna(0) <= 1e-6).astype(float)

    if "Egap_type" in df_fe.columns:
        egap_type = df_fe["Egap_type"].astype(str).str.lower()
        df_fe["Egap_type_is_metal"] = (egap_type == "metal").astype(float)

    if "spacegroup_relax" in df_fe.columns:
        df_fe["spacegroup_relax"] = pd.to_numeric(df_fe["spacegroup_relax"], errors="coerce")

    df_fe["sound_velocity_proxy"] = np.sqrt(
        df_fe["energy_density_proxy"].clip(lower=0) / (df_fe["density"] + eps)
    )

    df_fe["debye_proxy"] = (
        df_fe["sound_velocity_proxy"] *
        df_fe["volume_atom_pow_minus_1_3"]
    )

    df_fe["atomic_weight_rel_range"] = (
        df_fe["atomic_weight_range"] /
        (df_fe["atomic_weight_mean_w"] + eps)
    )

    df_fe["atomic_radius_rel_range"] = (
        df_fe["atomic_radius_range"] /
        (df_fe["atomic_radius_mean_w"] + eps)
    )

    df_fe["en_radius_interaction"] = (
        df_fe["en_pauling_mean_w"] *
        df_fe["atomic_radius_mean_w"]
    )

    df_fe["en_Z_interaction"] = (
        df_fe["en_pauling_mean_w"] *
        df_fe["Z_mean_w"]
    )

    light_elements = ["H", "B", "C", "N", "O"]
    light_cols = [f"el_{el}" for el in light_elements if f"el_{el}" in df_fe.columns]

    if len(light_cols) > 0:
        df_fe["light_element_fraction"] = df_fe[light_cols].sum(axis=1)
    else:
        df_fe["light_element_fraction"] = 0.0

    df_fe = df_fe.replace([np.inf, -np.inf], np.nan)

    return df_fe

_MODEL_CACHE = None

positive_targets = {
    "agl_debye",
    "agl_acoustic_debye",
    "agl_heat_capacity_Cv_300K",
    "agl_heat_capacity_Cp_300K",
    "agl_thermal_conductivity_300K",
    "agl_bulk_modulus_isothermal_300K",
    "agl_bulk_modulus_static_300K",
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


def get_predictor():
    global _MODEL_CACHE

    if _MODEL_CACHE is None:
        _MODEL_CACHE = load_saved_model()

    return _MODEL_CACHE


def get_model_status():
    try:
        get_predictor()
        return True, None
    except Exception as e:
        return False, str(e)


def predict_keras(X_new_scaled, target_cols, loaded_keras_items):
    keras_pred_lists = {
        target: []
        for target in target_cols
    }

    x_tensor = tf.convert_to_tensor(X_new_scaled, dtype=tf.float32)

    for item in loaded_keras_items:
        model = item["model"]
        targets = item["targets"]
        use_log_targets = set(item["use_log_targets"])

        y_mean = pd.Series(item["y_mean"])
        y_std = pd.Series(item["y_std"])

        pred_raw = model(x_tensor, training=False)

        if isinstance(pred_raw, list):
            pred_scaled = np.column_stack([
                p.numpy().flatten()
                for p in pred_raw
            ])
        else:
            pred_scaled = pred_raw.numpy()

        if pred_scaled.ndim == 1:
            pred_scaled = pred_scaled.reshape(-1, 1)

        pred_work = pred_scaled * y_std[targets].values + y_mean[targets].values
        pred_real = pred_work.copy()

        for j, target in enumerate(targets):
            if target in use_log_targets:
                pred_real[:, j] = np.expm1(pred_real[:, j])

            if target in positive_targets:
                pred_real[:, j] = np.clip(pred_real[:, j], 0, None)

            keras_pred_lists[target].append(pred_real[:, j])

    keras_pred_by_target = {}

    for target in target_cols:
        keras_pred_by_target[target] = np.mean(
            keras_pred_lists[target],
            axis=0
        )

    return keras_pred_by_target


def predict_xgb(X_new_imp, target_cols, xgb_models):
    xgb_pred_by_target = {}

    for target in target_cols:
        item = xgb_models[target]

        model = item["model"]
        use_log = item["use_log"]

        try:
            model.set_params(device="cpu", verbosity=0)
        except Exception:
            pass

        pred = model.predict(X_new_imp)

        if use_log:
            pred = np.expm1(pred)

        if target in positive_targets:
            pred = np.clip(pred, 0, None)

        xgb_pred_by_target[target] = pred

    return xgb_pred_by_target


def predict_from_dict(row_dict, return_dataframe=False):
    (
        imputer,
        scaler,
        feature_cols,
        target_cols,
        xgb_models,
        blend_weights,
        loaded_keras_items
    ) = get_predictor()

    df_new = input_df(row_dict)

    df_fe_new = make_features_for_prediction(df_new)

    X_new = df_fe_new.reindex(columns=feature_cols)
    X_new = X_new.select_dtypes(include=[np.number])

    X_new_imp = imputer.transform(X_new)
    X_new_scaled = scaler.transform(X_new_imp)

    keras_pred_by_target = predict_keras(
        X_new_scaled,
        target_cols,
        loaded_keras_items
    )

    xgb_pred_by_target = predict_xgb(
        X_new_imp,
        target_cols,
        xgb_models
    )

    final_pred_by_target = {}

    for target in target_cols:
        w_keras = blend_weights[target]["keras_weight"]
        w_xgb = blend_weights[target]["xgb_weight"]

        final_pred_by_target[target] = (
            w_keras * keras_pred_by_target[target] +
            w_xgb * xgb_pred_by_target[target]
        )

    result = {
        "compound": df_new["compound"].values[0]
    }

    for target in output_order:
        if target in target_cols:
            result[target] = float(final_pred_by_target[target][0])

    if return_dataframe:
        return pd.DataFrame([result])

    return result



# imputer, scaler, feature_cols, target_cols, xgb_models, blend_weights, loaded_keras_items = load_saved_model()
# df_new = input_df(row_dict)
# df_fe_new = make_features_for_prediction(df_new)
# X_new = df_fe_new.reindex(columns=feature_cols)
# X_new = X_new.select_dtypes(include=[np.number])
# X_new_imp = imputer.transform(X_new)
# X_new_scaled = scaler.transform(X_new_imp)


# def pred_list_to_matrix(pred_list):
#     if isinstance(pred_list, list):
#         return np.column_stack([p.flatten() for p in pred_list])
#     pred_arr = np.asarray(pred_list)
#     if pred_arr.ndim == 1:
#         pred_arr = pred_arr.reshape(-1, 1)
#     return pred_arr
# positive_targets = {
#     "agl_debye",
#     "agl_acoustic_debye",
#     "agl_heat_capacity_Cv_300K",
#     "agl_heat_capacity_Cp_300K",
#     "agl_thermal_conductivity_300K",
#     "agl_bulk_modulus_isothermal_300K",
#     "agl_bulk_modulus_static_300K",
# }
# keras_pred_lists = {target: [] for target in target_cols}
# for item in loaded_keras_items:
#     model = item["model"]
#     targets = item["targets"]
#     use_log_targets = set(item["use_log_targets"])

#     y_mean = pd.Series(item["y_mean"])
#     y_std = pd.Series(item["y_std"])

#     x_tensor = tf.convert_to_tensor(X_new_scaled, dtype=tf.float32)

#     pred_raw = model(x_tensor, training=False)

#     if isinstance(pred_raw, list):
#         pred_scaled = np.column_stack([p.numpy().flatten() for p in pred_raw])
#     else:
#         pred_scaled = pred_raw.numpy()

#     if pred_scaled.ndim == 1:
#         pred_scaled = pred_scaled.reshape(-1, 1)

#     pred_work = pred_scaled * y_std[targets].values + y_mean[targets].values
#     pred_real = pred_work.copy()

#     for j, target in enumerate(targets):
#         if target in use_log_targets:
#             pred_real[:, j] = np.expm1(pred_real[:, j])

#         if target in positive_targets:
#             pred_real[:, j] = np.clip(pred_real[:, j], 0, None)

#         keras_pred_lists[target].append(pred_real[:, j])
# keras_pred_by_target = {}
# for target in target_cols:
#     keras_pred_by_target[target] = np.mean(keras_pred_lists[target], axis=0)

# xgb_pred_by_target = {}
# for target in target_cols:
#     item = xgb_models[target]

#     model = item["model"]
#     use_log = item["use_log"]

#     try:
#         model.set_params(device="cpu", verbosity=0)
#     except Exception:
#         pass

#     pred = model.predict(X_new_imp)

#     if use_log:
#         pred = np.expm1(pred)

#     if target in positive_targets:
#         pred = np.clip(pred, 0, None)

#     xgb_pred_by_target[target] = pred

# final_pred_by_target = {}
# for target in target_cols:
#     w_keras = blend_weights[target]["keras_weight"]
#     w_xgb = blend_weights[target]["xgb_weight"]

#     final_pred_by_target[target] = (
#         w_keras * keras_pred_by_target[target] +
#         w_xgb * xgb_pred_by_target[target]
#     )

# prediction_result = pd.DataFrame()
# if "compound" in df_new.columns:
#     prediction_result["compound"] = df_new["compound"].values
# for target in target_cols:
#     prediction_result[f"pred_{target}"] = final_pred_by_target[target]
# print(prediction_result.to_string(index=False))

# prediction_result.to_csv("prediction_result.csv", index=False)
# print("            prediction_result.csv")
















