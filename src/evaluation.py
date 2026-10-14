from __future__ import annotations

from pathlib import Path
import argparse
import re
import warnings

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import GeometryCollection
from shapely.ops import unary_union
from shapely.strtree import STRtree


WORKING_DIR_PATH = Path.cwd()

MODEL_CHOICES = ["ndvi", "diff", "inverse_uhei", "std"]
SIZE_CHOICES = ["center", "full"]
METRIC_CHOICES = [
    "jaccard",
    "iou_buffer",
    "utility_diff",
    "yard_util",
    "street_util",
    "all",
]

NOISE_TYPES_WITH_LEVEL = {
    "gauss",
    "multiplicative",
    "mislabelling",
    "missingness",
    "geometric",
}

KNOWN_NOISE_SOURCES = {
    "ndvi",
    "uhei",
    "land_use",
    "trees",
    "num_areas",
    "full_space",
    "green_area",
    "ext_space",
    "street_space",
    "population_density",
}

AGGREGATION_METHODS = {
    "mean",
    "median",
    "weighted",
}

COLORS = {
    "gauss": "blue",
    "multiplicative": "green",
    "aggregation": "orange",
    "mislabelling": "red",
    "missingness": "purple",
    "geometric": "brown",
}

LINESTYLES = {
    "full_space": (0, (5, 2)),
    "ndvi": (0, (5, 2)),
    "num_areas": (0, (5, 2)),
    "land_use": (0, (5, 2)),
    "uhei": "-.",
    "trees": ":",
    "green_area": "--",
    "ext_space": "-.",
    "street_space": ":",
    "population_density": "-",
}

MARKERS = {
    "full_space": "^",
    "ndvi": "s",
    "num_areas": "o",
    "land_use": "D",
    "uhei": "P",
    "trees": "X",
    "green_area": "v",
    "ext_space": "<",
    "street_space": ">",
    "population_density": "*",
}

RESULT_COLUMNS = [
    "noise_type",
    "noise_source",
    "noise_level",
    "aggregation_method",
]


def build_columns(name: str) -> tuple[str, str, int | None, str | None]:
    """
    Estrae i metadati dal nome di un risultato.

    Formati supportati:
      - <noise_type>_<noise_source>_lvl<1-5>.geojson
      - aggregation_<noise_source>_<method>.geojson

    Il parser gestisce correttamente sorgenti con underscore, per esempio:
      - full_space
      - num_areas
      - green_area
      - ext_space
      - street_space
      - population_density
      - land_use
    """
    stem = Path(name).stem

    if stem == "no_noise":
        raise ValueError("no_noise.geojson è il baseline, non un risultato rumoroso.")

    if stem.startswith("aggregation_"):
        tail = stem.removeprefix("aggregation_")

        for method in sorted(AGGREGATION_METHODS, key=len, reverse=True):
            suffix = f"_{method}"
            if tail.endswith(suffix):
                noise_source = tail[: -len(suffix)]

                if not noise_source:
                    break

                if noise_source not in KNOWN_NOISE_SOURCES:
                    warnings.warn(
                        f"Sorgente di rumore non riconosciuta nel file '{name}': "
                        f"'{noise_source}'. Il file sarà comunque valutato."
                    )

                return "aggregation", noise_source, None, method

        raise ValueError(
            f"Nome aggregation non valido: '{name}'. "
            "Formato atteso: aggregation_<noise_source>_<mean|median|weighted>.geojson"
        )

    noise_types_pattern = "|".join(
        sorted(NOISE_TYPES_WITH_LEVEL, key=len, reverse=True)
    )

    match = re.fullmatch(
        rf"(?P<noise_type>{noise_types_pattern})_"
        rf"(?P<noise_source>.+)_lvl(?P<noise_level>[1-5])",
        stem,
    )

    if match is None:
        raise ValueError(
            f"Nome risultato non riconosciuto: '{name}'. "
            "Formato atteso: <noise_type>_<noise_source>_lvl<1-5>.geojson"
        )

    noise_type = match.group("noise_type")
    noise_source = match.group("noise_source")
    noise_level = int(match.group("noise_level"))

    if noise_source not in KNOWN_NOISE_SOURCES:
        warnings.warn(
            f"Sorgente di rumore non riconosciuta nel file '{name}': "
            f"'{noise_source}'. Il file sarà comunque valutato."
        )

    return noise_type, noise_source, noise_level, None


def iter_result_files(results_dir: Path):
    """
    Restituisce i risultati rumorosi validi e ignora il baseline.

    File con nomi non conformi vengono saltati con un warning, così un file
    ausiliario nella cartella non interrompe l'intera valutazione.
    """
    for file_path in sorted(results_dir.glob("*.geojson")):
        if file_path.name == "no_noise.geojson":
            continue

        try:
            metadata = build_columns(file_path.name)
        except ValueError as exc:
            warnings.warn(f"{exc} File ignorato.")
            continue

        yield file_path, metadata


def validate_paths(results_dir: Path, baseline_path: Path) -> None:
    if not results_dir.exists():
        raise FileNotFoundError(
            f"Cartella dei risultati non trovata: {results_dir}"
        )

    if not baseline_path.exists():
        raise FileNotFoundError(
            f"Baseline non trovato: {baseline_path}. "
            "Esegui prima run_model senza rumore per lo stesso modello e size."
        )


def load_geodata(
    path: Path,
    *,
    target_crs: str | None = None,
    required_columns: tuple[str, ...] = (),
) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)

    missing_columns = [
        column for column in required_columns if column not in gdf.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{path} non contiene le colonne richieste: {missing_columns}"
        )

    if gdf.crs is None:
        raise ValueError(
            f"{path} non ha un CRS. Assegna un CRS valido prima della valutazione."
        )

    if target_crs is not None:
        gdf = gdf.to_crs(target_crs)

    gdf = gdf[
        gdf.geometry.notna() & ~gdf.geometry.is_empty
    ].copy()

    if not gdf.empty:
        invalid = ~gdf.geometry.is_valid
        if invalid.any():
            warnings.warn(
                f"{path}: corrette {int(invalid.sum())} geometrie non valide "
                "tramite buffer(0)."
            )
            gdf.loc[invalid, "geometry"] = (
                gdf.loc[invalid, "geometry"].buffer(0)
            )

    return gdf


def geometry_union(gdf: gpd.GeoDataFrame):
    if gdf.empty:
        return GeometryCollection()

    return unary_union(list(gdf.geometry))


def sort_metric_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()

    if "noise_level" in df.columns:
        df["noise_level"] = pd.to_numeric(
            df["noise_level"],
            errors="coerce",
        ).astype("Int64")

    return df.sort_values(
        by=[
            "noise_type",
            "noise_source",
            "noise_level",
            "aggregation_method",
        ],
        na_position="last",
    ).reset_index(drop=True)


def write_metric_csv(
    rows: list[dict],
    out_csv: Path,
    value_columns: tuple[str, ...],
) -> pd.DataFrame:
    df = pd.DataFrame(
        rows,
        columns=[*RESULT_COLUMNS, *value_columns],
    )
    df = sort_metric_dataframe(df)
    df.to_csv(out_csv, index=False)
    print(f"Metric saved to {out_csv}")
    return df


def metric_jaccard(
    results_dir: Path,
    baseline_path: Path,
    out_dir: Path,
) -> pd.DataFrame:
    out_csv = out_dir / "jaccard.csv"

    baseline_gdf = load_geodata(
        baseline_path,
        target_crs="EPSG:3857",
    )
    baseline_geometry = geometry_union(baseline_gdf)

    rows = []

    for file_path, metadata in iter_result_files(results_dir):
        noise_type, noise_source, noise_level, aggregation_method = metadata

        noise_gdf = load_geodata(
            file_path,
            target_crs="EPSG:3857",
        )
        noise_geometry = geometry_union(noise_gdf)

        if baseline_geometry.is_empty and noise_geometry.is_empty:
            score = 1.0
        elif baseline_geometry.is_empty or noise_geometry.is_empty:
            score = 0.0
        else:
            intersection_area = baseline_geometry.intersection(
                noise_geometry
            ).area
            union_area = baseline_geometry.union(noise_geometry).area
            score = intersection_area / union_area if union_area else 0.0

        rows.append(
            {
                "noise_type": noise_type,
                "noise_source": noise_source,
                "noise_level": noise_level,
                "aggregation_method": aggregation_method,
                "score": score,
            }
        )

    return write_metric_csv(rows, out_csv, ("score",))


def _query_tree_indices(
    tree: STRtree,
    query_geometry,
    geometries: list,
    geometry_id_to_indices: dict[int, list[int]],
) -> list[int]:
    """
    Compatibilità tra STRtree di Shapely 1.x e 2.x:
    - Shapely 2 restituisce indici;
    - alcune versioni precedenti restituiscono geometrie.
    """
    candidates = tree.query(query_geometry)

    result = []

    for candidate in candidates:
        if isinstance(candidate, (int, np.integer)):
            result.append(int(candidate))
        else:
            result.extend(
                geometry_id_to_indices.get(id(candidate), [])
            )

    return result


def maximum_bipartite_matching_size(
    pairs: list[tuple[int, int, float]],
) -> int:
    if not pairs:
        return 0

    try:
        import networkx as nx

        graph = nx.Graph()
        baseline_nodes = {
            f"b_{baseline_index}"
            for baseline_index, _, _ in pairs
        }

        graph.add_nodes_from(
            baseline_nodes,
            bipartite=0,
        )

        noise_nodes = {
            f"n_{noise_index}"
            for _, noise_index, _ in pairs
        }

        graph.add_nodes_from(
            noise_nodes,
            bipartite=1,
        )

        graph.add_edges_from(
            (
                f"b_{baseline_index}",
                f"n_{noise_index}",
            )
            for baseline_index, noise_index, _ in pairs
        )

        matching = nx.algorithms.bipartite.maximum_matching(
            graph,
            top_nodes=baseline_nodes,
        )

        return sum(
            1
            for node in baseline_nodes
            if node in matching
        )

    except Exception as exc:
        warnings.warn(
            "NetworkX non disponibile o matching fallito; "
            f"uso il fallback greedy. Dettaglio: {exc}"
        )

        used_baseline = set()
        used_noise = set()

        for baseline_index, noise_index, _ in sorted(
            pairs,
            key=lambda item: item[2],
        ):
            if (
                baseline_index in used_baseline
                or noise_index in used_noise
            ):
                continue

            used_baseline.add(baseline_index)
            used_noise.add(noise_index)

        return len(used_baseline)


def iou_buffer(
    results_dir: Path,
    baseline_path: Path,
    out_dir: Path,
    distance_m: float = 300.0,
) -> pd.DataFrame:
    if distance_m <= 0:
        raise ValueError("--distance_m deve essere maggiore di zero.")

    out_csv = out_dir / "iou_buffer.csv"

    baseline_gdf = load_geodata(
        baseline_path,
        target_crs="EPSG:3857",
    )

    baseline_centroids = list(
        baseline_gdf.geometry.centroid.reset_index(drop=True)
    )
    baseline_count = len(baseline_centroids)

    rows = []

    for file_path, metadata in iter_result_files(results_dir):
        noise_type, noise_source, noise_level, aggregation_method = metadata

        noise_gdf = load_geodata(
            file_path,
            target_crs="EPSG:3857",
        )

        noise_centroids = list(
            noise_gdf.geometry.centroid.reset_index(drop=True)
        )
        noise_count = len(noise_centroids)

        if baseline_count == 0 and noise_count == 0:
            score = 1.0
        elif baseline_count == 0 or noise_count == 0:
            score = 0.0
        else:
            tree = STRtree(noise_centroids)

            geometry_id_to_indices: dict[int, list[int]] = {}
            for index, geometry in enumerate(noise_centroids):
                geometry_id_to_indices.setdefault(
                    id(geometry),
                    [],
                ).append(index)

            pairs = []

            for baseline_index, baseline_geometry in enumerate(
                baseline_centroids
            ):
                search_area = baseline_geometry.buffer(distance_m)

                candidate_indices = _query_tree_indices(
                    tree,
                    search_area,
                    noise_centroids,
                    geometry_id_to_indices,
                )

                for noise_index in candidate_indices:
                    distance = baseline_geometry.distance(
                        noise_centroids[noise_index]
                    )

                    if distance <= distance_m:
                        pairs.append(
                            (
                                baseline_index,
                                noise_index,
                                distance,
                            )
                        )

            intersection = maximum_bipartite_matching_size(pairs)
            union_count = baseline_count + noise_count - intersection
            score = intersection / union_count if union_count else 1.0

        rows.append(
            {
                "noise_type": noise_type,
                "noise_source": noise_source,
                "noise_level": noise_level,
                "aggregation_method": aggregation_method,
                "score": score,
            }
        )

    return write_metric_csv(rows, out_csv, ("score",))


def safe_mean(gdf: gpd.GeoDataFrame, column: str, path: Path) -> float:
    if column not in gdf.columns:
        raise ValueError(
            f"{path} non contiene la colonna richiesta '{column}'."
        )

    values = pd.to_numeric(
        gdf[column],
        errors="coerce",
    )

    if values.notna().sum() == 0:
        raise ValueError(
            f"{path}: la colonna '{column}' non contiene valori numerici validi."
        )

    return float(values.mean())


def safe_ratio(
    numerator: float,
    denominator: float,
) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        return np.nan

    if np.isclose(denominator, 0.0):
        return np.nan

    return numerator / denominator


def utility_metric(
    results_dir: Path,
    baseline_path: Path,
    out_dir: Path,
    *,
    metric_name: str,
    column: str,
) -> pd.DataFrame:
    out_csv = out_dir / f"{metric_name}.csv"

    baseline_gdf = load_geodata(
        baseline_path,
        required_columns=(column,),
    )
    baseline_mean = safe_mean(
        baseline_gdf,
        column,
        baseline_path,
    )

    rows = []

    for file_path, metadata in iter_result_files(results_dir):
        noise_type, noise_source, noise_level, aggregation_method = metadata

        noise_gdf = load_geodata(
            file_path,
            required_columns=(column,),
        )
        noise_mean = safe_mean(
            noise_gdf,
            column,
            file_path,
        )

        rows.append(
            {
                "noise_type": noise_type,
                "noise_source": noise_source,
                "noise_level": noise_level,
                "aggregation_method": aggregation_method,
                "mean_noise_over_baseline": safe_ratio(
                    noise_mean,
                    baseline_mean,
                ),
                "mean_baseline_over_noise": safe_ratio(
                    baseline_mean,
                    noise_mean,
                ),
                "baseline_mean": baseline_mean,
                "noise_mean": noise_mean,
            }
        )

    return write_metric_csv(
        rows,
        out_csv,
        (
            "mean_noise_over_baseline",
            "mean_baseline_over_noise",
            "baseline_mean",
            "noise_mean",
        ),
    )


def utility_difference(
    results_dir: Path,
    baseline_path: Path,
    out_dir: Path,
) -> pd.DataFrame:
    return utility_metric(
        results_dir,
        baseline_path,
        out_dir,
        metric_name="utility_diff",
        column="utility",
    )


def utility_yard_cells(
    results_dir: Path,
    baseline_path: Path,
    out_dir: Path,
) -> pd.DataFrame:
    return utility_metric(
        results_dir,
        baseline_path,
        out_dir,
        metric_name="yard_util",
        column="yard_cells",
    )


def utility_street_cells(
    results_dir: Path,
    baseline_path: Path,
    out_dir: Path,
) -> pd.DataFrame:
    return utility_metric(
        results_dir,
        baseline_path,
        out_dir,
        metric_name="street_util",
        column="street_cells",
    )


def apply_metric(
    metric: str,
    results_dir: Path,
    baseline_path: Path,
    out_dir: Path,
    distance_m: float,
) -> None:
    functions = {
        "jaccard": lambda: metric_jaccard(
            results_dir,
            baseline_path,
            out_dir,
        ),
        "iou_buffer": lambda: iou_buffer(
            results_dir,
            baseline_path,
            out_dir,
            distance_m=distance_m,
        ),
        "utility_diff": lambda: utility_difference(
            results_dir,
            baseline_path,
            out_dir,
        ),
        "yard_util": lambda: utility_yard_cells(
            results_dir,
            baseline_path,
            out_dir,
        ),
        "street_util": lambda: utility_street_cells(
            results_dir,
            baseline_path,
            out_dir,
        ),
    }

    if metric == "all":
        for metric_name, function in functions.items():
            print(f"Computing metric: {metric_name}")
            function()
        return

    functions[metric]()


def prepare_plot_dataframe(metric_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(metric_csv)

    missing = [
        column for column in RESULT_COLUMNS if column not in df.columns
    ]
    if missing:
        raise ValueError(
            f"{metric_csv} non contiene le colonne richieste: {missing}"
        )

    df["noise_level"] = pd.to_numeric(
        df["noise_level"],
        errors="coerce",
    )

    return sort_metric_dataframe(df)


def plot_line_groups(
    axis,
    dataframe: pd.DataFrame,
    value_column: str,
) -> None:
    for (noise_type, noise_source), subset in dataframe.groupby(
        ["noise_type", "noise_source"],
        dropna=False,
    ):
        subset = subset.sort_values("noise_level")

        axis.plot(
            subset["noise_level"],
            subset[value_column],
            label=f"{noise_type}-{noise_source}",
            color=COLORS.get(noise_type, "black"),
            linestyle=LINESTYLES.get(noise_source, "-"),
            marker=MARKERS.get(noise_source, "o"),
        )



AGGREGATION_METHOD_ORDER = ("mean", "median", "weighted")


def _ordered_aggregation_methods(dataframe: pd.DataFrame) -> list[str]:
    """Restituisce i metodi aggregation in un ordine stabile."""
    available = [
        str(value)
        for value in dataframe["aggregation_method"].dropna().unique()
    ]

    ordered = [
        method
        for method in AGGREGATION_METHOD_ORDER
        if method in available
    ]
    extras = sorted(
        method
        for method in available
        if method not in AGGREGATION_METHOD_ORDER
    )
    return ordered + extras


def _annotate_aggregation_bars(
    axis,
    bars,
    raw_values: list[float],
    delta_values: list[float],
) -> None:
    """Mostra sopra ogni barra il valore originale della metrica."""
    for bar, raw_value, delta_value in zip(
        bars,
        raw_values,
        delta_values,
    ):
        if not np.isfinite(raw_value) or not np.isfinite(delta_value):
            continue

        bar_center = bar.get_x() + bar.get_width() / 2

        if np.isclose(delta_value, 0.0, atol=1e-12):
            axis.scatter(
                [bar_center],
                [0.0],
                marker="s",
                s=45,
                color=bar.get_facecolor(),
                zorder=3,
            )
            axis.annotate(
                f"{raw_value:.6f}",
                xy=(bar_center, 0.0),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )
        else:
            axis.annotate(
                f"{raw_value:.6f}",
                xy=(bar_center, delta_value / 2.0),
                ha="center",
                va="center",
                fontsize=8,
                rotation=90,
            )


def _configure_aggregation_delta_axis(
    axis,
    delta_values: list[float],
    *,
    baseline_value: float,
) -> None:
    """
    Configura l'asse aggregation come variazione percentuale dal baseline.

    Il valore originale resta visibile come etichetta sulle barre. In questo
    modo un risultato pari a 1.0 viene rappresentato esplicitamente come 0%
    di variazione, invece di produrre una linea piatta poco interpretabile.
    """
    finite_deltas = np.asarray(
        [value for value in delta_values if np.isfinite(value)],
        dtype=float,
    )

    axis.axhline(
        0.0,
        color="gray",
        linestyle="--",
        linewidth=1.5,
    )
    axis.set_ylabel("Change from baseline (%)")
    axis.grid(True, axis="y", alpha=0.35)

    if finite_deltas.size == 0:
        return

    max_abs_delta = float(np.max(np.abs(finite_deltas)))
    limit = max(0.15, max_abs_delta * 1.35)
    axis.set_ylim(-limit, limit)

    if np.allclose(finite_deltas, 0.0, atol=1e-10, rtol=0.0):
        axis.text(
            0.5,
            0.93,
            (
                "No change from baseline\n"
                f"all values = {baseline_value:.6f}"
            ),
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=10,
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "white",
                "edgecolor": "gray",
                "alpha": 0.9,
            },
        )


def plot_aggregation_groups(
    axis,
    dataframe: pd.DataFrame,
    value_column: str,
    baseline_value: float = 1.0,
) -> None:
    """
    Disegna l'aggregation come barre della variazione percentuale dal baseline.

    In un grafico combinato, ogni noise_source costituisce un gruppo di barre.
    Le etichette sulle barre riportano il valore originale della metrica.
    """
    clean = dataframe.dropna(
        subset=["aggregation_method", "noise_source", value_column]
    ).copy()

    if clean.empty:
        return

    methods = _ordered_aggregation_methods(clean)
    sources = [
        str(value)
        for value in clean["noise_source"].dropna().unique()
    ]

    x_positions = np.arange(len(methods), dtype=float)
    source_count = max(len(sources), 1)
    bar_width = min(0.75 / source_count, 0.32)

    all_deltas: list[float] = []

    for source_index, noise_source in enumerate(sources):
        source_rows = (
            clean[clean["noise_source"].astype(str) == noise_source]
            .drop_duplicates(
                subset=["aggregation_method"],
                keep="last",
            )
            .set_index("aggregation_method")
            .reindex(methods)
        )

        raw_values = pd.to_numeric(
            source_rows[value_column],
            errors="coerce",
        ).to_numpy(dtype=float)
        delta_values = (raw_values - baseline_value) * 100.0

        offset = (
            source_index - (source_count - 1) / 2
        ) * bar_width

        bars = axis.bar(
            x_positions + offset,
            delta_values,
            width=bar_width,
            label=f"aggregation-{noise_source}",
        )

        _annotate_aggregation_bars(
            axis,
            bars,
            raw_values.tolist(),
            delta_values.tolist(),
        )
        all_deltas.extend(delta_values.tolist())

    axis.set_xticks(x_positions)
    axis.set_xticklabels(methods)
    axis.set_xlabel("Aggregation method")

    _configure_aggregation_delta_axis(
        axis,
        all_deltas,
        baseline_value=baseline_value,
    )

    if len(sources) > 1:
        axis.legend(
            fontsize="small",
            title="Aggregation - source",
            title_fontsize="medium",
        )




def plot_jaccard_and_iou_buffer(
    metric: str,
    metric_csv: Path,
    plot_dir: Path,
    baseline_score: float = 1.0,
) -> None:
    if metric == "jaccard":
        title = "Jaccard Index"
        out_plot = plot_dir / "jaccard.png"
    else:
        title = "IoU buffer index"
        out_plot = plot_dir / "IoU_buffer.png"

    df = prepare_plot_dataframe(metric_csv)

    if "score" not in df.columns:
        raise ValueError(
            f"{metric_csv} non contiene la colonna 'score'."
        )

    df_aggregation = df[df["noise_type"] == "aggregation"]
    df_main = df[df["noise_type"] != "aggregation"]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, 8),
        sharey=False,
        gridspec_kw={"width_ratios": [2.5, 1.25]},
    )

    left_axis, right_axis = axes

    fig.suptitle(
        title,
        fontsize=20,
        fontweight="bold",
    )

    if not df_main.empty:
        plot_line_groups(
            left_axis,
            df_main,
            "score",
        )

    left_axis.axhline(
        baseline_score,
        color="gray",
        linestyle="--",
    )
    left_axis.set_xlabel("Noise Level")
    left_axis.set_ylabel("Score")
    left_axis.grid(True)

    if not df_main.empty:
        left_axis.legend(
            fontsize="small",
            title="Noise type - source",
            title_fontsize="medium",
        )

    if not df_aggregation.empty:
        right_axis.set_title(
            "Aggregation — change from baseline"
        )
        plot_aggregation_groups(
            right_axis,
            df_aggregation,
            "score",
            baseline_value=baseline_score,
        )
    else:
        right_axis.set_visible(False)

    fig.tight_layout()
    fig.savefig(
        out_plot,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"Plot saved in {out_plot}")




def plot_utility_yard_street(
    metric: str,
    metric_csv: Path,
    plot_dir: Path,
) -> None:
    settings = {
        "utility_diff": (
            "Utility score",
            plot_dir / "utility.png",
        ),
        "yard_util": (
            "Yard cells utility score",
            plot_dir / "yard_cells.png",
        ),
        "street_util": (
            "Street cells utility score",
            plot_dir / "street_cells.png",
        ),
    }

    title, out_plot = settings[metric]

    df = prepare_plot_dataframe(metric_csv)

    required = {
        "mean_noise_over_baseline",
        "mean_baseline_over_noise",
    }

    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            f"{metric_csv} non contiene le colonne richieste: {missing}"
        )

    df_aggregation = df[df["noise_type"] == "aggregation"]
    df_main = df[df["noise_type"] != "aggregation"]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(16, 16),
        sharey=False,
        gridspec_kw={"width_ratios": [2.5, 1.25]},
    )

    top_left = axes[0, 0]
    top_right = axes[0, 1]
    bottom_left = axes[1, 0]
    bottom_right = axes[1, 1]

    fig.suptitle(
        title,
        fontsize=20,
        fontweight="bold",
    )

    top_left.set_title("Noise / Baseline")
    bottom_left.set_title("Baseline / Noise")

    if not df_main.empty:
        plot_line_groups(
            top_left,
            df_main,
            "mean_noise_over_baseline",
        )
        plot_line_groups(
            bottom_left,
            df_main,
            "mean_baseline_over_noise",
        )

    for axis in (top_left, bottom_left):
        axis.axhline(
            1.0,
            color="gray",
            linestyle="--",
        )
        axis.set_xlabel("Noise Level")
        axis.set_ylabel("Score")
        axis.grid(True)

        if not df_main.empty:
            axis.legend(
                fontsize="small",
                title="Noise type - source",
                title_fontsize="medium",
            )

    if not df_aggregation.empty:
        top_right.set_title(
            "Aggregation: Noise / Baseline — change"
        )
        bottom_right.set_title(
            "Aggregation: Baseline / Noise — change"
        )

        plot_aggregation_groups(
            top_right,
            df_aggregation,
            "mean_noise_over_baseline",
            baseline_value=1.0,
        )
        plot_aggregation_groups(
            bottom_right,
            df_aggregation,
            "mean_baseline_over_noise",
            baseline_value=1.0,
        )
    else:
        top_right.set_visible(False)
        bottom_right.set_visible(False)

    fig.tight_layout()
    fig.savefig(
        out_plot,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"Plot saved in {out_plot}")



def _safe_plot_filename(value: str) -> str:
    """Converte una sorgente in un nome file sicuro e leggibile."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return cleaned or "unknown_source"


def _score_metric_title(metric: str) -> str:
    return "Jaccard Index" if metric == "jaccard" else "IoU buffer index"


def _utility_metric_title(metric: str) -> str:
    return {
        "utility_diff": "Utility score",
        "yard_util": "Yard cells utility score",
        "street_util": "Street cells utility score",
    }[metric]


def _plot_source_lines(
    axis,
    dataframe: pd.DataFrame,
    value_column: str,
) -> None:
    """
    Disegna, per una singola noise_source, una curva per ogni noise_type.
    """
    noise_type_markers = {
        "gauss": "o",
        "multiplicative": "s",
        "mislabelling": "D",
        "missingness": "X",
        "geometric": "^",
    }

    for noise_type, subset in dataframe.groupby(
        "noise_type",
        dropna=False,
    ):
        subset = subset.dropna(
            subset=["noise_level", value_column]
        ).sort_values("noise_level")

        if subset.empty:
            continue

        axis.plot(
            subset["noise_level"],
            subset[value_column],
            label=str(noise_type),
            color=COLORS.get(noise_type, "black"),
            marker=noise_type_markers.get(noise_type, "o"),
            linewidth=2,
        )



def plot_score_sources(
    metric: str,
    metric_csv: Path,
    plot_dir: Path,
    baseline_score: float = 1.0,
) -> None:
    """
    Crea un grafico separato per ogni noise_source per Jaccard e IoU.

    Per aggregation usa barre della variazione percentuale dal baseline e
    riporta il valore originale sopra ogni barra.
    """
    df = prepare_plot_dataframe(metric_csv)

    if "score" not in df.columns:
        raise ValueError(
            f"{metric_csv} non contiene la colonna 'score'."
        )

    df_main = df[df["noise_type"] != "aggregation"].copy()
    df_aggregation = df[df["noise_type"] == "aggregation"].copy()

    if df_main.empty and df_aggregation.empty:
        return

    source_dir = plot_dir / "by_source" / metric
    source_dir.mkdir(parents=True, exist_ok=True)

    title = _score_metric_title(metric)

    for noise_source, subset in df_main.groupby(
        "noise_source",
        dropna=False,
    ):
        fig, axis = plt.subplots(figsize=(9, 6))

        _plot_source_lines(
            axis,
            subset,
            "score",
        )

        axis.axhline(
            baseline_score,
            color="gray",
            linestyle="--",
            linewidth=1.5,
            label="baseline",
        )
        axis.set_title(
            f"{title} — noise source: {noise_source}",
            fontsize=14,
            fontweight="bold",
        )
        axis.set_xlabel("Noise Level")
        axis.set_ylabel("Score")
        axis.set_xticks([1, 2, 3, 4, 5])
        axis.grid(True, alpha=0.35)
        axis.legend(
            title="Noise type",
            fontsize="small",
            title_fontsize="medium",
        )

        fig.tight_layout()

        out_plot = source_dir / (
            f"{_safe_plot_filename(noise_source)}.png"
        )
        fig.savefig(
            out_plot,
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)

        print(f"Source plot saved in {out_plot}")

    for noise_source, subset in df_aggregation.groupby(
        "noise_source",
        dropna=False,
    ):
        subset = subset.dropna(
            subset=["aggregation_method", "score"]
        ).copy()
        if subset.empty:
            continue

        methods = _ordered_aggregation_methods(subset)
        ordered = (
            subset.drop_duplicates(
                subset=["aggregation_method"],
                keep="last",
            )
            .set_index("aggregation_method")
            .reindex(methods)
        )

        raw_values = pd.to_numeric(
            ordered["score"],
            errors="coerce",
        ).to_numpy(dtype=float)
        delta_values = (raw_values - baseline_score) * 100.0

        fig, axis = plt.subplots(figsize=(8, 5.5))
        bars = axis.bar(
            methods,
            delta_values,
        )

        _annotate_aggregation_bars(
            axis,
            bars,
            raw_values.tolist(),
            delta_values.tolist(),
        )
        _configure_aggregation_delta_axis(
            axis,
            delta_values.tolist(),
            baseline_value=baseline_score,
        )

        axis.set_title(
            (
                f"{title} — aggregation source: {noise_source}\n"
                "change from baseline"
            ),
            fontsize=14,
            fontweight="bold",
        )
        axis.set_xlabel("Aggregation method")

        fig.tight_layout()
        out_plot = source_dir / (
            f"aggregation__{_safe_plot_filename(noise_source)}.png"
        )
        fig.savefig(out_plot, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Aggregation source plot saved in {out_plot}")




def plot_utility_sources(
    metric: str,
    metric_csv: Path,
    plot_dir: Path,
) -> None:
    """
    Crea un grafico separato per ogni noise_source per le metriche utility.

    Per aggregation mostra la variazione percentuale rispetto al valore neutro
    1.0 e annota il rapporto originale sopra ogni barra.
    """
    df = prepare_plot_dataframe(metric_csv)

    required = {
        "mean_noise_over_baseline",
        "mean_baseline_over_noise",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            f"{metric_csv} non contiene le colonne richieste: {missing}"
        )

    df_main = df[df["noise_type"] != "aggregation"].copy()
    df_aggregation = df[df["noise_type"] == "aggregation"].copy()

    if df_main.empty and df_aggregation.empty:
        return

    source_dir = plot_dir / "by_source" / metric
    source_dir.mkdir(parents=True, exist_ok=True)

    title = _utility_metric_title(metric)

    for noise_source, subset in df_main.groupby(
        "noise_source",
        dropna=False,
    ):
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(13, 5.5),
            sharex=True,
        )

        left_axis, right_axis = axes

        _plot_source_lines(
            left_axis,
            subset,
            "mean_noise_over_baseline",
        )
        _plot_source_lines(
            right_axis,
            subset,
            "mean_baseline_over_noise",
        )

        left_axis.set_title("Noise / Baseline")
        right_axis.set_title("Baseline / Noise")

        for axis in axes:
            axis.axhline(
                1.0,
                color="gray",
                linestyle="--",
                linewidth=1.5,
                label="baseline",
            )
            axis.set_xlabel("Noise Level")
            axis.set_ylabel("Score")
            axis.set_xticks([1, 2, 3, 4, 5])
            axis.grid(True, alpha=0.35)
            axis.legend(
                title="Noise type",
                fontsize="small",
                title_fontsize="medium",
            )

        fig.suptitle(
            f"{title} — noise source: {noise_source}",
            fontsize=14,
            fontweight="bold",
        )
        fig.tight_layout()

        out_plot = source_dir / (
            f"{_safe_plot_filename(noise_source)}.png"
        )
        fig.savefig(
            out_plot,
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)

        print(f"Source plot saved in {out_plot}")

    for noise_source, subset in df_aggregation.groupby(
        "noise_source",
        dropna=False,
    ):
        subset = subset.dropna(
            subset=[
                "aggregation_method",
                "mean_noise_over_baseline",
                "mean_baseline_over_noise",
            ]
        ).copy()
        if subset.empty:
            continue

        methods = _ordered_aggregation_methods(subset)
        ordered = (
            subset.drop_duplicates(
                subset=["aggregation_method"],
                keep="last",
            )
            .set_index("aggregation_method")
            .reindex(methods)
        )

        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        values = (
            (
                "mean_noise_over_baseline",
                "Noise / Baseline — change",
            ),
            (
                "mean_baseline_over_noise",
                "Baseline / Noise — change",
            ),
        )

        for axis, (value_column, subtitle) in zip(axes, values):
            raw_values = pd.to_numeric(
                ordered[value_column],
                errors="coerce",
            ).to_numpy(dtype=float)
            delta_values = (raw_values - 1.0) * 100.0

            bars = axis.bar(
                methods,
                delta_values,
            )
            _annotate_aggregation_bars(
                axis,
                bars,
                raw_values.tolist(),
                delta_values.tolist(),
            )
            _configure_aggregation_delta_axis(
                axis,
                delta_values.tolist(),
                baseline_value=1.0,
            )
            axis.set_title(subtitle)
            axis.set_xlabel("Aggregation method")

        fig.suptitle(
            f"{title} — aggregation source: {noise_source}",
            fontsize=14,
            fontweight="bold",
        )
        fig.tight_layout()

        out_plot = source_dir / (
            f"aggregation__{_safe_plot_filename(noise_source)}.png"
        )
        fig.savefig(out_plot, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Aggregation source plot saved in {out_plot}")



def create_plot(
    metric: str,
    metric_csv: Path,
    plot_dir: Path,
) -> None:
    # Mantiene invariato il grafico combinato originale.
    if metric in {"jaccard", "iou_buffer"}:
        plot_jaccard_and_iou_buffer(
            metric,
            metric_csv,
            plot_dir,
        )

        # Aggiunge un grafico separato per ogni noise_source.
        plot_score_sources(
            metric,
            metric_csv,
            plot_dir,
        )
    else:
        plot_utility_yard_street(
            metric,
            metric_csv,
            plot_dir,
        )

        # Aggiunge un grafico separato per ogni noise_source.
        plot_utility_sources(
            metric,
            metric_csv,
            plot_dir,
        )


def create_requested_plots(
    metric: str,
    evaluation_dir: Path,
    plot_dir: Path,
) -> None:
    metrics = (
        [
            "jaccard",
            "iou_buffer",
            "utility_diff",
            "yard_util",
            "street_util",
        ]
        if metric == "all"
        else [metric]
    )

    for metric_name in metrics:
        metric_csv = evaluation_dir / f"{metric_name}.csv"

        if not metric_csv.exists():
            warnings.warn(
                f"CSV non trovato per '{metric_name}': {metric_csv}. "
                "Calcola prima la metrica con --plot False."
            )
            continue

        create_plot(
            metric_name,
            metric_csv,
            plot_dir,
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate TALEA result GeoJSON files."
    )

    parser.add_argument(
        "--size",
        type=str,
        default="full",
        choices=SIZE_CHOICES,
        help="Map size used by run_model: center or full.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="ndvi",
        choices=MODEL_CHOICES,
        help="Model evaluated: ndvi, diff, inverse_uhei or std.",
    )

    parser.add_argument(
        "--metric",
        type=str,
        default="jaccard",
        choices=METRIC_CHOICES,
        help=(
            "Metric to compute or plot. Use 'all' to process all metrics."
        ),
    )

    parser.add_argument(
        "--plot",
        type=str,
        default="False",
        choices=["True", "False"],
        help=(
            "False computes CSV metrics; True creates plots from existing CSV files."
        ),
    )

    parser.add_argument(
        "--distance_m",
        type=float,
        default=300.0,
        help="Maximum matching distance, in metres, for iou_buffer.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    results_dir = (
        WORKING_DIR_PATH
        / "results"
        / args.size
        / args.model
        / "std"
    )

    baseline_path = results_dir / "no_noise.geojson"

    # Include size in the output path to prevent center/full collisions.
    evaluation_dir = (
        WORKING_DIR_PATH
        / "evaluation"
        / args.size
        / args.model
    )

    plot_dir = (
        WORKING_DIR_PATH
        / "plot"
        / args.size
        / args.model
    )

    if args.plot == "False":
        validate_paths(
            results_dir,
            baseline_path,
        )
        evaluation_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        apply_metric(
            metric=args.metric,
            results_dir=results_dir,
            baseline_path=baseline_path,
            out_dir=evaluation_dir,
            distance_m=args.distance_m,
        )
    else:
        plot_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        create_requested_plots(
            metric=args.metric,
            evaluation_dir=evaluation_dir,
            plot_dir=plot_dir,
        )


if __name__ == "__main__":
    main()
