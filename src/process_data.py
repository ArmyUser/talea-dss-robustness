from pathlib import Path
import json
import numpy as np
import argparse
import shutil
import pandas as pd

WORKING_DIR_PATH = Path.cwd()
RAW_DATA_DIR_PATH = WORKING_DIR_PATH.joinpath("dataset", "raw_data")

def apply_noise_to_output(OUTPUT_DIR_PATH: Path, noise_type: str, noise_source: str, noise_level: str, seed: int = 42):
    match noise_source:
        case "uhei":
            apply_noise_uhei(OUTPUT_DIR_PATH, noise_type, noise_level, seed)
        case "ndvi":
            apply_noise_ndvi(OUTPUT_DIR_PATH, noise_type, noise_level, seed)
        case "land_use":
            apply_noise_land_use(OUTPUT_DIR_PATH, noise_level, seed)
        case "population_density":
            csv_path = OUTPUT_DIR_PATH.joinpath("popolazione_per_area_statistica.csv")
            apply_noise_population(csv_path, noise_type, noise_level, seed)


def apply_noise_land_use(OUTPUT_DIR_PATH, noise_level, seed=42, hect_thresh=3.5):
    np.random.seed(seed)
    frac_map = {"1": 0.1, "2": 0.15, "3": 0.2, "4": 0.3, "5": 0.5}
    frac = frac_map.get(noise_level)
    geojson_path = OUTPUT_DIR_PATH.joinpath("uso_suolo_bologna.geojson")
    with open(geojson_path, "r", encoding="utf-8") as f:
        geojson_data = json.load(f)
    features = geojson_data.get("features", [])
    n = len(features)
    n_to_mislabel = int(n * frac)
    def get_hect(idx):
        return float(features[idx].get("properties", {}).get("HECTARES"))
    def get_descr(idx):
        return features[idx].get("properties", {}).get("DESCR")
    def pair_valid(i, j, hect_threshold):
        hi = get_hect(i)
        hj = get_hect(j)
        if hi > hect_threshold or hj > hect_threshold:
            return False
        di = get_descr(i)
        dj = get_descr(j)
        if di == dj:
            return False
        return True
    fields_to_swap = ["DESCR", "SIGLA", "COD_1", "COD_2", "COD_3", "COD_4", "COD_TOT"]
    swapped = 0
    used = set()
    while swapped < n_to_mislabel:
        i = np.random.randint(1, n)
        j = i - 1
        if i not in used and j not in used and pair_valid(i, j, hect_thresh):
            for field in fields_to_swap:
                features[i]["properties"][field], features[j]["properties"][field] = \
                    features[j]["properties"][field], features[i]["properties"][field]
            used.add(i)
            used.add(j)
            swapped += 2
    with open(geojson_path, "w", encoding="utf-8") as f:
        json.dump(geojson_data, f, indent=2, ensure_ascii=False)


def apply_noise_uhei(OUTPUT_DIR_PATH, noise_type, noise_level, seed=42):
    np.random.seed(seed)
    geojson_path = OUTPUT_DIR_PATH.joinpath("indexes", "urban_heat_exposure_index.geojson")
    with open(geojson_path, "r", encoding="utf-8") as f: 
        geojson_data = json.load(f)
    features = geojson_data.get("features", [])
    values = np.array([float(feat["properties"]["value"]) for feat in features], dtype=float)
    match noise_type:
        case "gauss":
            sigma_map = {"1": 0.1, "2": 0.3, "3": 0.5, "4": 0.6, "5": 0.7}
            sigma = sigma_map.get(noise_level)
            noise = np.random.normal(0, sigma, len(values))
            new_vals = np.clip(values + noise, 0, None)
        case "multiplicative":
            sigma_map = {"1": 0.1, "2": 0.2, "3": 0.3, "4": 0.4, "5": 0.5}
            sigma = sigma_map.get(noise_level)
            noise = np.random.normal(1, sigma, len(values))
            new_vals = np.clip(values * noise, 0, None)
    for feat, nv in zip(features, new_vals):
        feat["properties"]["value"] = str(nv)  
    with open(geojson_path, "w", encoding="utf-8") as f:
        json.dump(geojson_data, f, indent=2, ensure_ascii=False)


def apply_noise_ndvi(OUTPUT_DIR_PATH, noise_type, noise_level, seed=42):
    np.random.seed(seed)
    geojson_path = OUTPUT_DIR_PATH.joinpath("indexes", "ndvi_mean.geojson")
    with open(geojson_path, "r", encoding="utf-8") as f: 
        geojson_data = json.load(f)
    features = geojson_data.get("features", [])
    values = np.array([float(feat["properties"]["value"]) for feat in features], dtype=float)
    match noise_type:
        case "gauss":
            sigma_map = {"1": 0.1, "2": 0.2, "3": 0.3, "4": 0.4, "5": 0.5}
            sigma = sigma_map.get(noise_level)
            noise = np.random.normal(0, sigma, len(values))
            new_vals = values + noise
        case "multiplicative":
            sigma_map = {"1": 0.05, "2": 0.1, "3": 0.2, "4": 0.3, "5": 0.4}
            sigma = sigma_map.get(noise_level)
            noise = np.random.normal(1, sigma, len(values))
            new_vals = values * noise
    new_vals = np.clip(new_vals, -1.0, 1.0)
    for feat, nv in zip(features, new_vals):
        feat["properties"]["value"] = str(nv)  
    with open(geojson_path, "w", encoding="utf-8") as f:
        json.dump(geojson_data, f, indent=2, ensure_ascii=False)


def apply_noise_population(csv_path: Path, noise_type: str, noise_level: str, seed: int = 42):
    np.random.seed(seed)
    df = pd.read_csv(csv_path)

    if "Residenti" not in df.columns:
        raise ValueError("La colonna 'Residenti' non esiste nel CSV.")

    values = pd.to_numeric(df["Residenti"], errors="coerce").to_numpy(dtype=float)

    if noise_type == "gauss":
        sigma_map = {
            "1": 0.05,
            "2": 0.10,
            "3": 0.15,
            "4": 0.20,
            "5": 0.25
        }
        sigma = sigma_map.get(noise_level)
        if sigma is None:
            raise ValueError(f"noise_level non valido: {noise_level}")

        noise = np.random.normal(0, sigma * values)
        new_vals = values + noise
        new_vals = np.maximum(0, np.round(new_vals)).astype(int)
        df["Residenti"] = new_vals

    elif noise_type == "multiplicative":
        sigma_map = {
            "1": 0.02,
            "2": 0.05,
            "3": 0.10,
            "4": 0.15,
            "5": 0.20
        }
        sigma = sigma_map.get(noise_level)
        if sigma is None:
            raise ValueError(f"noise_level non valido: {noise_level}")

        noise = np.random.normal(1, sigma, len(values))
        new_vals = values * noise
        new_vals = np.maximum(0, np.round(new_vals)).astype(int)
        df["Residenti"] = new_vals

    elif noise_type == "mislabelling":
        frac_map = {"1": 0.1, "2": 0.15, "3": 0.2, "4": 0.3, "5": 0.5}
        frac = frac_map.get(noise_level)
        if frac is None:
            raise ValueError(f"noise_level non valido: {noise_level}")

        df = df.copy()

        # mantieni coerenza spaziale: swap SOLO dentro Quartiere + Zona
        grouped = df.groupby(["Quartiere", "Zona"])

        for _, idx in grouped.indices.items():
            idx = np.array(idx)

            n_to_swap = int(len(idx) * frac)
            n_to_swap = (n_to_swap // 2) * 2

            if n_to_swap < 2:
                continue

            selected = np.random.choice(idx, size=n_to_swap, replace=False)
            pairs = selected.reshape(-1, 2)

            for i, j in pairs:
                df.at[i, "Residenti"], df.at[j, "Residenti"] = \
                    df.at[j, "Residenti"], df.at[i, "Residenti"]


    elif noise_type == "missingness":
        frac_map = {
            "1": 0.1,
            "2": 0.2,
            "3": 0.3,
            "4": 0.4,
            "5": 0.5,
        }

        frac = frac_map.get(noise_level)

        if frac is None:
            raise ValueError(
                f"noise_level non valido: {noise_level}"
            )

        df = df.copy()

        n_missing = int(len(df) * frac)

        if n_missing <= 0:
            raise ValueError(
                "Il numero di valori da rimuovere è zero."
            )

        missing_idx = np.random.choice(
            df.index.to_numpy(),
            size=n_missing,
            replace=False,
        )

        # Iniezione della missingness
        df.loc[missing_idx, "Residenti"] = np.nan

        global_median = df["Residenti"].median(
            skipna=True
        )

        if pd.isna(global_median):
            raise ValueError(
                "Impossibile imputare Residenti: "
                "non esistono valori validi dopo la missingness."
            )

        # 1. Imputazione più specifica:
        #    stessa combinazione Quartiere + Zona
        if {
            "Quartiere",
            "Zona",
        }.issubset(df.columns):
            df["Residenti"] = (
                df.groupby(
                    ["Quartiere", "Zona"]
                )["Residenti"]
                .transform(
                    lambda series: series.fillna(
                        series.median()
                    )
                )
            )

        # 2. Fallback a livello di Quartiere
        if "Quartiere" in df.columns:
            df["Residenti"] = (
                df.groupby("Quartiere")["Residenti"]
                .transform(
                    lambda series: series.fillna(
                        series.median()
                    )
                )
            )

        # 3. Fallback globale
        df["Residenti"] = df["Residenti"].fillna(
            global_median
        )

        resident_values = df["Residenti"].to_numpy(
            dtype=float
        )

        if not np.isfinite(resident_values).all():
            invalid_rows = np.flatnonzero(
                ~np.isfinite(resident_values)
            ).tolist()

            raise ValueError(
                "L'imputazione di Residenti non è riuscita. "
                f"Righe non valide: {invalid_rows[:20]}"
            )

        # Residenti rappresenta un conteggio
        df["Residenti"] = (
            df["Residenti"]
            .clip(lower=0)
            .round()
            .astype(int)
        )

        print(
            "Missingness population_density completata: "
            f"{n_missing}/{len(df)} valori rimossi "
            f"e successivamente imputati."
        )
  

    else:
        raise ValueError(f"noise_type non supportato: {noise_type}")

    df.to_csv(csv_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Processing data with noise.")

    parser.add_argument(
        "--noise_type",
        dest="noise_type",
        type=str,
        default="gauss",
        choices=["gauss", "multiplicative", "mislabelling", "missingness"],
        help="Type of noise to apply."
    )

    parser.add_argument(
        "--noise_source",
        dest="noise_source",
        type=str,
        default="uhei",
        choices=["uhei", "ndvi", "land_use", "population_density"],
        help="Where to apply the noise"
    )

    parser.add_argument(
        "--noise_level",
        dest="noise_level",
        type=str,
        default="1",
        choices=["1", "2", "3", "4", "5"],
        help="Noise level."
    )

    args = parser.parse_args()

    folder_name_parts = [
        f"raw_data_{args.noise_type}",
        args.noise_source,
        f"lvl{args.noise_level}"
    ]

    folder_name = "_".join(folder_name_parts)
    OUTPUT_DIR_PATH = WORKING_DIR_PATH.joinpath("dataset", folder_name)

    print(f"Copying content from {RAW_DATA_DIR_PATH} to {OUTPUT_DIR_PATH}\n")

    shutil.copytree(RAW_DATA_DIR_PATH, OUTPUT_DIR_PATH, dirs_exist_ok=True)

    apply_noise_to_output(
        OUTPUT_DIR_PATH,
        args.noise_type,
        args.noise_source,
        args.noise_level
    )
