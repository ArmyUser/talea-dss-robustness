import os
import geopandas as gpd
from pathlib import Path
import pandas as pd
import argparse
from qgis.core import *
from qgis.analysis import QgsNativeAlgorithms
import processing
from processing.core.Processing import Processing
import numpy as np


WORKING_DIR_PATH = Path.cwd()

QgsApplication.setPrefixPath("/usr", True)
qgs = QgsApplication([], False)
qgs.initQgis()

Processing.initialize()

if qgs.processingRegistry().providerById("native") is None:
    qgs.processingRegistry().addProvider(
        QgsNativeAlgorithms()
    )


def save_layer(output_layer, layer_name, path, remove_attrs="geo_point_2d"):
    """
    Save a QGIS vector layer to a GeoJSON file with filtered attributes.

    Input:
    - output_layer: QgsVectorLayer, the layer to save
    - layer_name: str, the name of the layer to be saved
    - path: str, the directory path where the layer will be saved
    - remove_attrs: list or str, attributes to remove from the layer

    """

    output_file_path = os.path.join(path, f'{layer_name}.geojson')

    original_fields = output_layer.fields()
    new_fields = QgsFields()
    field_indices_to_keep = []

    # Prepare field filtering
    if remove_attrs:
        if isinstance(remove_attrs, str):
            remove_attrs = [remove_attrs]

        for idx, field in enumerate(original_fields):
            if field.name() not in remove_attrs:
                new_fields.append(field)
                field_indices_to_keep.append(idx)
    else:
        # No fields to remove
        new_fields = original_fields
        field_indices_to_keep = list(range(len(original_fields)))

    # Set up writer
    geometry_type = output_layer.wkbType()
    crs = output_layer.crs()
    transform_context = QgsProject.instance().transformContext()

    options = QgsVectorFileWriter.SaveVectorOptions()
    options.driverName = "GeoJSON"
    options.fileEncoding = "UTF-8"
    options.actionOnExistingFile = QgsVectorFileWriter.CreateOrOverwriteFile

    writer = QgsVectorFileWriter.create(
        output_file_path,
        new_fields,
        geometry_type,
        crs,
        transform_context,
        options,
        QgsFeatureSink.SinkFlags()
    )

    # Write features with filtered attributes
    if writer.hasError() == QgsVectorFileWriter.NoError:
        for original_feature in output_layer.getFeatures():
            new_feature = QgsFeature()
            new_feature.setGeometry(original_feature.geometry())

            new_attributes = []
            for idx in field_indices_to_keep:
                val = original_feature.attributes()[idx]
                if val == "" or val is None:
                    val = "Unknown"  # Or any other safe default
                new_attributes.append(val)

            new_feature.setAttributes(new_attributes)
            writer.addFeature(new_feature)

        del writer
        print(f"Layer saved to: {output_file_path}")
    else:
        print(f"Failed to save layer: {writer.errorMessage()}")


def compute_weighted_uhei(grid, uhei_layer):
    """
    Compute the weighted UHEI for each grid cell based on the intersection area with UHEI data.

    Input:
    - grid: QgsVectorLayer or pathlib.Path, the grid layer or path to the grid GeoJSON file
    - uhei_layer: QgsVectorLayer or pathlib.Path, the UHEI layer or path to the GeoJSON file
    Output:
    - result: GeoDataFrame, containing the grid cells with their corresponding weighted UHEI values

    """

    grid_uhei = processing.run("native:fieldcalculator", {'INPUT':grid,'FIELD_NAME':'area_grid','FIELD_TYPE':0,'FIELD_LENGTH':0,'FIELD_PRECISION':0,'FORMULA':'$area','OUTPUT':'TEMPORARY_OUTPUT'})['OUTPUT']
    save_layer(grid_uhei, 'grid_uhei', PROCESSED_DATA_DIR_PATH)

    # Weighted UHEI on the basis of the area computation
    intersection_layer = processing.run("native:intersection", {'INPUT':grid_uhei,'OVERLAY':uhei_layer,'INPUT_FIELDS':[],'OVERLAY_FIELDS':['uhei'],'OVERLAY_FIELDS_PREFIX':'','OUTPUT':'TEMPORARY_OUTPUT','GRID_SIZE':None})['OUTPUT']
    intersection_layer = processing.run("native:fieldcalculator", {'INPUT':intersection_layer,'FIELD_NAME':'area_intersection','FIELD_TYPE':0,'FIELD_LENGTH':0,'FIELD_PRECISION':0,'FORMULA':'$area','OUTPUT':'TEMPORARY_OUTPUT'})['OUTPUT']
    intersection_layer = processing.run("native:fieldcalculator", {'INPUT':intersection_layer,'FIELD_NAME':'weighted_uhei','FIELD_TYPE':0,'FIELD_LENGTH':0,'FIELD_PRECISION':0,'FORMULA':'area_intersection/area_grid*uhei','OUTPUT':'TEMPORARY_OUTPUT'})['OUTPUT']
    save_layer(intersection_layer, 'intersection_layer_uhei', PROCESSED_DATA_DIR_PATH)

    intersection = gpd.read_file(PROCESSED_DATA_DIR_PATH.joinpath("intersection_layer_uhei.geojson"))
    grid_uhei = gpd.read_file(PROCESSED_DATA_DIR_PATH.joinpath("grid_uhei.geojson"))

    id_col = 'id'
    sum_col = 'weighted_uhei'

    intersection = intersection.drop(['area_grid', 'area_intersection', 'uhei'], axis=1)
    agg = {c: 'first' for c in intersection.columns if c not in [id_col, sum_col]}
    agg[sum_col] = 'sum'

    result = intersection.groupby(id_col, as_index=False).agg(agg)
    geom_map = grid_uhei.set_index("id")["geometry"]
    result["geometry"] = result["id"].map(geom_map)
    result = gpd.GeoDataFrame(result, geometry='geometry', crs=intersection.crs)

    return result


def compute_mean_uhei(grid, uhei_layer):

    grid_uhei = processing.run("native:fieldcalculator", {'INPUT':grid,'FIELD_NAME':'area_grid','FIELD_TYPE':0,'FIELD_LENGTH':0,'FIELD_PRECISION':0,'FORMULA':'$area','OUTPUT':'TEMPORARY_OUTPUT'})['OUTPUT']
    save_layer(grid_uhei, 'grid_uhei', PROCESSED_DATA_DIR_PATH)

    intersection_layer = processing.run("native:intersection", {'INPUT':grid_uhei,'OVERLAY':uhei_layer,'INPUT_FIELDS':[],'OVERLAY_FIELDS':['uhei'],'OVERLAY_FIELDS_PREFIX':'','OUTPUT':'TEMPORARY_OUTPUT','GRID_SIZE':None})['OUTPUT']
    save_layer(intersection_layer, 'intersection_layer_uhei', PROCESSED_DATA_DIR_PATH)

    intersection = gpd.read_file(PROCESSED_DATA_DIR_PATH.joinpath("intersection_layer_uhei.geojson"))
    grid_uhei = gpd.read_file(PROCESSED_DATA_DIR_PATH.joinpath("grid_uhei.geojson"))

    id_col = 'id'
    mean_col = 'weighted_uhei'

    intersection = intersection.rename(columns={'uhei': mean_col})
    agg = {c: 'first' for c in intersection.columns if c not in [id_col, mean_col, 'geometry']}
    agg[mean_col] = 'mean'

    result = intersection.groupby(id_col, as_index=False).agg(agg)
    geom_map = grid_uhei.set_index("id")["geometry"]
    result["geometry"] = result["id"].map(geom_map)
    result = gpd.GeoDataFrame(result, geometry='geometry', crs=intersection.crs)

    return result


def compute_median_uhei(grid, uhei_layer):

    grid_uhei = processing.run("native:fieldcalculator", {'INPUT':grid,'FIELD_NAME':'area_grid','FIELD_TYPE':0,'FIELD_LENGTH':0,'FIELD_PRECISION':0,'FORMULA':'$area','OUTPUT':'TEMPORARY_OUTPUT'})['OUTPUT']
    save_layer(grid_uhei, 'grid_uhei', PROCESSED_DATA_DIR_PATH)

    intersection_layer = processing.run("native:intersection", {'INPUT':grid_uhei,'OVERLAY':uhei_layer,'INPUT_FIELDS':[],'OVERLAY_FIELDS':['uhei'],'OVERLAY_FIELDS_PREFIX':'','OUTPUT':'TEMPORARY_OUTPUT','GRID_SIZE':None})['OUTPUT']
    save_layer(intersection_layer, 'intersection_layer_uhei', PROCESSED_DATA_DIR_PATH)

    intersection = gpd.read_file(PROCESSED_DATA_DIR_PATH.joinpath("intersection_layer_uhei.geojson"))
    grid_uhei = gpd.read_file(PROCESSED_DATA_DIR_PATH.joinpath("grid_uhei.geojson"))

    id_col = 'id'
    mean_col = 'weighted_uhei'

    intersection = intersection.rename(columns={'uhei': mean_col})
    agg = {c: 'first' for c in intersection.columns if c not in [id_col, mean_col, 'geometry']}
    agg[mean_col] = 'median'

    result = intersection.groupby(id_col, as_index=False).agg(agg)
    geom_map = grid_uhei.set_index("id")["geometry"]
    result["geometry"] = result["id"].map(geom_map)
    result = gpd.GeoDataFrame(result, geometry='geometry', crs=intersection.crs)

    return result


def compute_weighted_ndvi(grid, ndvi_layer):
    """
    Compute the weighted NDVI for each grid cell based on the intersection area with NDVI data.

    Input:
    - grid: QgsVectorLayer or pathlib.Path, the grid layer or path to the grid GeoJSON file
    - ndvi_layer: QgsVectorLayer or pathlib.Path, the NDVI index layer or path to the GeoJSON file
    Output:
    - result: GeoDataFrame, containing the grid cells with their corresponding weighted NDVI values

    """

    grid_ndvi = processing.run("native:fieldcalculator", {'INPUT':grid,'FIELD_NAME':'area_grid','FIELD_TYPE':0,'FIELD_LENGTH':0,'FIELD_PRECISION':0,'FORMULA':'$area','OUTPUT':'TEMPORARY_OUTPUT'})['OUTPUT']
    save_layer(grid_ndvi, 'grid_ndvi', PROCESSED_DATA_DIR_PATH)

    # Weighted NDVI on the basis of the area computation
    intersection_layer = processing.run("native:intersection", {'INPUT':grid_ndvi,'OVERLAY':ndvi_layer,'INPUT_FIELDS':[],'OVERLAY_FIELDS':['ndvi'],'OVERLAY_FIELDS_PREFIX':'','OUTPUT':'TEMPORARY_OUTPUT','GRID_SIZE':None})['OUTPUT']
    intersection_layer = processing.run("native:fieldcalculator", {'INPUT':intersection_layer,'FIELD_NAME':'area_intersection','FIELD_TYPE':0,'FIELD_LENGTH':0,'FIELD_PRECISION':0,'FORMULA':'$area','OUTPUT':'TEMPORARY_OUTPUT'})['OUTPUT']
    intersection_layer = processing.run("native:fieldcalculator", {'INPUT':intersection_layer,'FIELD_NAME':'weighted_ndvi','FIELD_TYPE':0,'FIELD_LENGTH':0,'FIELD_PRECISION':0,'FORMULA':'area_intersection/area_grid*ndvi','OUTPUT':'TEMPORARY_OUTPUT'})['OUTPUT']
    save_layer(intersection_layer, 'intersection_layer_ndvi', PROCESSED_DATA_DIR_PATH)

    intersection = gpd.read_file(PROCESSED_DATA_DIR_PATH.joinpath("intersection_layer_ndvi.geojson"))
    grid_ndvi = gpd.read_file(PROCESSED_DATA_DIR_PATH.joinpath("grid_ndvi.geojson"))

    id_col = 'id'
    sum_col = 'weighted_ndvi'

    intersection = intersection.drop(['area_grid', 'area_intersection', 'ndvi'], axis=1)
    agg = {c: 'first' for c in intersection.columns if c not in [id_col, sum_col]}
    agg[sum_col] = 'sum'

    result = intersection.groupby(id_col, as_index=False).agg(agg)
    geom_map = grid_ndvi.set_index("id")["geometry"]
    result["geometry"] = result["id"].map(geom_map)
    result = gpd.GeoDataFrame(result, geometry='geometry', crs=intersection.crs)

    return result

def compute_mean_ndvi(grid, ndvi_layer):
    grid_ndvi = processing.run("native:fieldcalculator", {'INPUT':grid,'FIELD_NAME':'area_grid','FIELD_TYPE':0,'FIELD_LENGTH':0,'FIELD_PRECISION':0,'FORMULA':'$area','OUTPUT':'TEMPORARY_OUTPUT'})['OUTPUT']
    save_layer(grid_ndvi, 'grid_ndvi', PROCESSED_DATA_DIR_PATH)

    intersection_layer = processing.run("native:intersection", {'INPUT':grid_ndvi,'OVERLAY':ndvi_layer,'INPUT_FIELDS':[],'OVERLAY_FIELDS':['ndvi'],'OVERLAY_FIELDS_PREFIX':'','OUTPUT':'TEMPORARY_OUTPUT','GRID_SIZE':None})['OUTPUT']
    save_layer(intersection_layer, 'intersection_layer_ndvi', PROCESSED_DATA_DIR_PATH)

    intersection = gpd.read_file(PROCESSED_DATA_DIR_PATH.joinpath("intersection_layer_ndvi.geojson"))
    grid_ndvi = gpd.read_file(PROCESSED_DATA_DIR_PATH.joinpath("grid_ndvi.geojson"))

    id_col = 'id'
    mean_col = 'weighted_ndvi'

    intersection = intersection.rename(columns={'ndvi': mean_col})
    agg = {c: 'first' for c in intersection.columns if c not in [id_col, mean_col, 'geometry']}
    agg[mean_col] = 'mean'

    result = intersection.groupby(id_col, as_index=False).agg(agg)
    geom_map = grid_ndvi.set_index("id")["geometry"]
    result["geometry"] = result["id"].map(geom_map)
    result = gpd.GeoDataFrame(result, geometry='geometry', crs=intersection.crs)

    return result

def compute_median_ndvi(grid, ndvi_layer):
    grid_ndvi = processing.run("native:fieldcalculator", {'INPUT':grid,'FIELD_NAME':'area_grid','FIELD_TYPE':0,'FIELD_LENGTH':0,'FIELD_PRECISION':0,'FORMULA':'$area','OUTPUT':'TEMPORARY_OUTPUT'})['OUTPUT']
    save_layer(grid_ndvi, 'grid_ndvi', PROCESSED_DATA_DIR_PATH)

    intersection_layer = processing.run("native:intersection", {'INPUT':grid_ndvi,'OVERLAY':ndvi_layer,'INPUT_FIELDS':[],'OVERLAY_FIELDS':['ndvi'],'OVERLAY_FIELDS_PREFIX':'','OUTPUT':'TEMPORARY_OUTPUT','GRID_SIZE':None})['OUTPUT']
    save_layer(intersection_layer, 'intersection_layer_ndvi', PROCESSED_DATA_DIR_PATH)

    intersection = gpd.read_file(PROCESSED_DATA_DIR_PATH.joinpath("intersection_layer_ndvi.geojson"))
    grid_ndvi = gpd.read_file(PROCESSED_DATA_DIR_PATH.joinpath("grid_ndvi.geojson"))

    id_col = 'id'
    mean_col = 'weighted_ndvi'

    intersection = intersection.rename(columns={'ndvi': mean_col})
    agg = {c: 'first' for c in intersection.columns if c not in [id_col, mean_col, 'geometry']}
    agg[mean_col] = 'median'

    result = intersection.groupby(id_col, as_index=False).agg(agg)
    geom_map = grid_ndvi.set_index("id")["geometry"]
    result["geometry"] = result["id"].map(geom_map)
    result = gpd.GeoDataFrame(result, geometry='geometry', crs=intersection.crs)

    return result


def streets_processing():
    """
    Deletion of the squares and the highway from the street data.

    Output:
    - streets: QgsVectorLayer, vector layer containing the cleaned street data
    
    """

    # Squares elimination
    gdf_streets = gpd.read_file(RAW_DATA_DIR_PATH.joinpath("aree-stradali.geojson"))
    gdf_streets_1 = gdf_streets[gdf_streets['descrizion'] != 'Tronco di intersezione tra strade a raso']
    gdf_streets_2 = gdf_streets[gdf_streets['descrizion'] == 'Tronco di intersezione tra strade a raso']
    gdf_streets_2 = gdf_streets_2[gdf_streets_2['area_ogg'] <= 7500]
    gdf_streets = pd.concat([gdf_streets_1, gdf_streets_2], axis=0)
    gdf_streets.to_file(PROCESSED_DATA_DIR_PATH.joinpath("aree-stradali-modified.geojson"), driver='GeoJSON')

    # Highway elimination
    highway = gpd.read_file(str(RAW_DATA_DIR_PATH.joinpath('uso_suolo_bologna.geojson')))
    highway = highway[highway['DESCR'] == 'Autostrade e superstrade']
    highway.to_file(str(PROCESSED_DATA_DIR_PATH.joinpath('autostrada.geojson')), driver='GeoJSON')

    streets = processing.run("native:difference", {'INPUT':str(PROCESSED_DATA_DIR_PATH.joinpath("aree-stradali-modified.geojson")),'OVERLAY':str(PROCESSED_DATA_DIR_PATH.joinpath('autostrada.geojson')),'OUTPUT':'TEMPORARY_OUTPUT','GRID_SIZE':None})['OUTPUT']
    save_layer(streets, 'aree-stradali-modified', PROCESSED_DATA_DIR_PATH)
    return streets


def apply_noise_trees(noise_level, GRID_DIR_PATH, seed=42, window_size=5):
    np.random.seed(seed)
    frac_map = {"1": 0.1, "2": 0.2, "3": 0.3, "4": 0.4, "5": 0.5}
    frac = frac_map.get(noise_level)
    geojson_path = GRID_DIR_PATH .joinpath("trees_outside_green.geojson")
    gdf = gpd.read_file(str(geojson_path))
    tree_field = "tree_number"
    total_cells = len(gdf)
    n_mask = int(np.floor(frac * total_cells))
    mask_indices = np.random.choice(gdf.index, size=n_mask, replace=False)
    coord_to_idx = {}
    for idx, row in gdf.iterrows():
        r = int(row["row_index"])
        c = int(row["col_index"])
        coord_to_idx[(r, c)] = idx
    values = gdf[tree_field].copy()
    values.loc[mask_indices] = np.nan
    half = window_size // 2
    def neighbor_values(r_center, c_center):
        vals = []
        for dr in range(-half, half + 1):
            for dc in range(-half, half + 1):
                rr = r_center + dr
                cc = c_center + dc
                if rr == r_center and cc == c_center:
                    continue
                key = (rr, cc)
                idx = coord_to_idx.get(key)
                if idx is None:
                    continue
                v = values.loc[idx]
                if not np.isnan(v):
                    vals.append(v)
        return vals
    computed_values = {}
    for idx in mask_indices:
        row = gdf.loc[idx]
        r = int(row["row_index"])
        c = int(row["col_index"])
        n_val = neighbor_values(r, c)
        med = float(np.nanmedian(n_val))
        computed_values[idx] = med
    for idx, val in computed_values.items():
        gdf.at[idx, tree_field] = val
    gdf.to_file(str(geojson_path), driver="GeoJSON")


def apply_num_areas_gauss(noise_level, grid, seed=42):
    np.random.seed(seed)
    sigma_map = {"1": 0.5, "2": 1, "3": 1.5, "4": 2.5}
    sigma = sigma_map.get(noise_level)
    fld_idx = grid.fields().indexOf('free_space_number')
    feats = list(grid.getFeatures())
    values = np.array([0.0 if f['free_space_number'] is None else float(f['free_space_number']) for f in feats], dtype=float)
    noise = np.random.normal(0, sigma, len(values))
    new_vals = np.clip(values + noise, 0, None).astype(int)
    dp = grid.dataProvider()
    grid.startEditing()
    changes = {}
    for f, nv in zip(feats, new_vals):
        fid = f.id()
        changes[fid] = {fld_idx: int(nv)}
    dp.changeAttributeValues(changes)
    grid.commitChanges()
    return grid


def apply_num_areas_missingness(noise_level, grid, seed=42, distance_m=300):
    np.random.seed(seed)
    frac_map = {"1": 0.1, "2": 0.2, "3": 0.3, "4": 0.4, "5": 0.5}
    frac = frac_map.get(noise_level)
    fld_idx = grid.fields().indexOf('free_space_number')
    feats = list(grid.getFeatures())
    fids = [f.id() for f in feats]
    centroids = [f.geometry().centroid() for f in feats]
    vals = {f.id(): (0.0 if f['free_space_number'] is None else float(f['free_space_number'])) for f in feats}
    index = QgsSpatialIndex()
    for f in feats:
        index.addFeature(f)
    n_modify = int(len(grid) * frac)
    chosen_fids = list(np.random.choice(fids, size=n_modify, replace=False))
    changes = {}
    for cfid in chosen_fids:
        # trova posizione dell'elemento nella lista feats (per centr)
        try:
            pos = fids.index(cfid)
        except ValueError:
            continue
        cgeom = centroids[pos]
        if cgeom is None or cgeom.isEmpty():
            continue

        # crea buffer attorno al centroide (distance_m in unità CRS)
        try:
            buf = cgeom.buffer(distance_m, 8)  # 8 segmenti, regola se vuoi più precisione
        except Exception:
            # fallback: se centroide non valido -> skip
            continue

        # usa bounding box per interrogare index (più veloce)
        candidate_ids = index.intersects(buf.boundingBox())

        # rimuovi self e calcola distanza esatta per filtrare i veri vicini
        neighbor_vals = []
        for nid in candidate_ids:
            if nid == cfid:
                continue
            # ottieni posizione di nid nella lista feats (potrebbe non esserci)
            try:
                npos = fids.index(nid)
            except ValueError:
                continue
            ncent = centroids[npos]
            if ncent is None or ncent.isEmpty():
                continue
            # distanza geometrica (unità CRS)
            try:
                d = cgeom.distance(ncent)
            except Exception:
                # se distance fallisce, salta
                continue
            if d <= distance_m + 1e-9:
                # accetta come vicino
                v = vals.get(nid, None)
                if v is not None:
                    neighbor_vals.append(v)

        # se abbiamo vicini, calcola la mediana e programma la sostituzione
        if neighbor_vals:
            median_val = int(np.median(neighbor_vals))
            changes[cfid] = {fld_idx: int(median_val)}

    if changes:
        dp = grid.dataProvider()
        grid.startEditing()
        dp.changeAttributeValues(changes)
        grid.commitChanges()

    return grid
    

def create_verde(raw_data_path, processed_data_path):
    """
    Creation of the vector containig all the green elements unified from different datasets

    Input:
    - raw_data_path: pathlib.Path, path to the raw data directory
    - processed_data_path: pathlib.Path, path to the processed data directory
    Output:
    - verde: QgsVectorLayer, vector layer containing the union of all green elements

    """
    
    un_gest = processing.run("native:fixgeometries", {'INPUT':str(raw_data_path.joinpath('un_gest.fgb')), 'OUTPUT':'TEMPORARY_OUTPUT'})['OUTPUT']
    null_values = processing.run("native:extractbyattribute", {'INPUT':un_gest,'FIELD':'area_prato','OPERATOR':8,'VALUE':'','OUTPUT':'TEMPORARY_OUTPUT'})['OUTPUT']
    to_delete = processing.run("native:extractbyattribute", {'INPUT':null_values,'FIELD':'nome','OPERATOR':0,'VALUE':'AIUOLE ALBERTO MANZI','OUTPUT':'TEMPORARY_OUTPUT'})['OUTPUT']
    un_gest = processing.run("native:difference", {'INPUT':un_gest,'OVERLAY':to_delete,'OUTPUT':'TEMPORARY_OUTPUT','GRID_SIZE':None})['OUTPUT']
    verde_privato_urbanizzato = processing.run("native:fixgeometries", {'INPUT':str(raw_data_path.joinpath('verde_privato_urbanizzato.fgb')), 'OUTPUT':'TEMPORARY_OUTPUT'})['OUTPUT']

    aree_boschive_1 = QgsVectorLayer(str(raw_data_path.joinpath('aree-boschive.gpkg')) + "|layername=V_AAI_GPG", 'aree_boschive_1', 'ogr')
    aree_boschive_1 = processing.run("native:fixgeometries", {'INPUT':aree_boschive_1, 'OUTPUT':'TEMPORARY_OUTPUT'})['OUTPUT']
    aree_boschive_2 = QgsVectorLayer(str(raw_data_path.joinpath('aree-boschive.gpkg')) + "|layername=V_PSR_GPG", 'aree_boschive_2', 'ogr')
    aree_boschive_2 = processing.run("native:fixgeometries", {'INPUT':aree_boschive_2, 'OUTPUT':'TEMPORARY_OUTPUT'})['OUTPUT']
    aree_boschive_3 = QgsVectorLayer(str(raw_data_path.joinpath('aree-boschive.gpkg')) + "|layername=V_BSC_GPG", 'aree_boschive_3', 'ogr')
    aree_boschive_3 = processing.run("native:fixgeometries", {'INPUT':aree_boschive_3, 'OUTPUT':'TEMPORARY_OUTPUT'})['OUTPUT']
    aree_fluviali = QgsVectorLayer(str(raw_data_path.joinpath('aree-fluviali.gpkg')) + "|layername=V_ABA_GPG", 'aree_fluviali', 'ogr')
    aree_fluviali = processing.run("native:fixgeometries", {'INPUT':aree_fluviali, 'OUTPUT':'TEMPORARY_OUTPUT'})['OUTPUT']

    gdf = gpd.read_file(raw_data_path.joinpath('aree-statistiche.geojson'))
    gdf = gdf[gdf["area_statistica"].isin(["GIARDINI MARGHERITA"])]
    gdf.to_file(processed_data_path.joinpath("giardini_margherita.geojson"), driver="GeoJSON")

    verde = processing.run("native:multiunion", {'INPUT':un_gest,'OVERLAYS':[str(processed_data_path.joinpath('giardini_margherita.geojson')), verde_privato_urbanizzato, aree_boschive_1, aree_boschive_2, aree_boschive_3, aree_fluviali],'OVERLAY_FIELDS_PREFIX':'','OUTPUT':'TEMPORARY_OUTPUT'})['OUTPUT']
    verde = processing.run("native:fixgeometries", {'INPUT':verde, 'OUTPUT':str(PROCESSED_DATA_DIR_PATH.joinpath('verde.gpkg'))})['OUTPUT']
    print(f"Layer saved to: {str(PROCESSED_DATA_DIR_PATH.joinpath('verde.gpkg'))}")
    return verde


############################################################################################
# Media di tutte le aree: 1165.74 m² -> Se vogliamo simulare un erroe realistico il massimo che si può considerare è un 100 di metri sbagliati
# media in gradi 1.318766499529894e-07
# per cui i sigma possono andare da 1, 3, 5, 10, 20


from shapely.affinity import scale as shapely_scale
from shapely.geometry import Polygon, MultiPolygon

try:
    from shapely import make_valid as shapely_make_valid
except ImportError:
    try:
        from shapely.validation import make_valid as shapely_make_valid
    except ImportError:
        shapely_make_valid = None

# =============================================================================
# RUMORE GEOMETRICO POLIGONALE
# =============================================================================

_GEOMETRIC_NOISE_LEVELS = {
    # Variazione massima delle coordinate rispetto al centroide.
    #
    # L'area varia approssimativamente con il quadrato del fattore:
    # livello 1 -> circa ±0.5% di area
    # livello 2 -> circa ±1.0% di area
    # livello 3 -> circa ±2.0% di area
    # livello 4 -> circa ±3.0% di area
    # livello 5 -> circa ±5.0% di area
    "1": 0.0025,
    "2": 0.0050,
    "3": 0.0100,
    "4": 0.0150,
    "5": 0.0250,
}


def _polygon_parts(geometry):
    """
    Estrae tutte le componenti Polygon da una geometria Shapely.

    Sono supportati:
    - Polygon
    - MultiPolygon
    - GeometryCollection

    Point e LineString vengono ignorati.
    """
    if geometry is None or geometry.is_empty:
        return []

    polygon_parts = []
    geometries_to_process = [geometry]

    while geometries_to_process:
        current = geometries_to_process.pop()

        if current is None or current.is_empty:
            continue

        geometry_type = current.geom_type

        if geometry_type == "Polygon":
            polygon_parts.append(current)

        elif geometry_type == "MultiPolygon":
            polygon_parts.extend(
                polygon
                for polygon in current.geoms
                if polygon is not None and not polygon.is_empty
            )

        elif geometry_type == "GeometryCollection":
            geometries_to_process.extend(
                list(current.geoms)
            )

    return polygon_parts


def _repair_as_multipolygon(geometry):
    """
    Restituisce una geometria MultiPolygon valida.

    Restituisce None se la geometria:
    - è vuota;
    - non contiene componenti poligonali;
    - non può essere riparata.
    """
    if geometry is None or geometry.is_empty:
        return None

    candidate = geometry

    if not candidate.is_valid:
        repaired = None

        if shapely_make_valid is not None:
            try:
                repaired = shapely_make_valid(candidate)
            except Exception:
                repaired = None

        if repaired is None or repaired.is_empty:
            try:
                repaired = candidate.buffer(0)
            except Exception:
                repaired = None

        if repaired is None or repaired.is_empty:
            return None

        candidate = repaired

    polygon_parts = _polygon_parts(candidate)

    if not polygon_parts:
        return None

    try:
        candidate = MultiPolygon(polygon_parts)
    except Exception:
        return None

    if candidate.is_empty or candidate.area <= 0:
        return None

    if not candidate.is_valid:
        repaired = None

        if shapely_make_valid is not None:
            try:
                repaired = shapely_make_valid(candidate)
            except Exception:
                repaired = None

        if repaired is None or repaired.is_empty:
            try:
                repaired = candidate.buffer(0)
            except Exception:
                repaired = None

        polygon_parts = _polygon_parts(repaired)

        if not polygon_parts:
            return None

        try:
            candidate = MultiPolygon(polygon_parts)
        except Exception:
            return None

    if (
        candidate.is_empty
        or candidate.area <= 0
        or not candidate.is_valid
    ):
        return None

    return candidate


def _sample_geometric_scale(rng, maximum_delta):
    """
    Genera un fattore di scala leggermente inferiore o superiore a 1.

    Ogni feature viene quindi:
    - rimpicciolita; oppure
    - ingrandita.

    La magnitudine è compresa tra il 50% e il 100% del livello
    selezionato, così da evitare fattori praticamente uguali a 1.
    """
    direction = -1.0 if rng.random() < 0.5 else 1.0

    magnitude = rng.uniform(
        maximum_delta * 0.5,
        maximum_delta
    )

    return 1.0 + direction * magnitude


def inject_polygon_geometric_noise(
    input_path,
    noise_level,
    output_path,
    layer_name,
    seed=42
):
    """
    Applica un lieve rumore geometrico a un layer poligonale.

    La trasformazione avviene in EPSG:25832 e consiste in uno scaling
    uniforme delle coordinate rispetto al centroide della feature.

    Il metodo:
    - mantiene gli attributi;
    - produce sempre MultiPolygon;
    - conserva la geometria originale se la trasformazione fallisce;
    - elimina soltanto geometrie vuote o non poligonali;
    - restituisce un QgsVectorLayer pronto per le elaborazioni QGIS.
    """
    noise_level = str(noise_level)

    maximum_delta = _GEOMETRIC_NOISE_LEVELS.get(
        noise_level
    )

    if maximum_delta is None:
        raise ValueError(
            f"Livello di rumore geometrico non valido: "
            f"{noise_level}"
        )

    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Layer di input non trovato: {input_path}"
        )

    source_gdf = gpd.read_file(
        str(input_path)
    ).reset_index(drop=True)

    if source_gdf.crs is None:
        raise ValueError(
            f"Il layer {layer_name} non ha un CRS definito."
        )

    if source_gdf.empty:
        raise ValueError(
            f"Il layer {layer_name} è vuoto."
        )

    original_crs = source_gdf.crs

    # Tutte le variazioni vengono calcolate in metri.
    metric_gdf = source_gdf.to_crs(
        "EPSG:25832"
    )

    rng = np.random.default_rng(seed)

    kept_positions = []
    clean_metric_geometries = []
    noisy_metric_geometries = []

    enlarged_count = 0
    reduced_count = 0
    fallback_count = 0
    dropped_before_count = 0

    for position, geometry in enumerate(
        metric_gdf.geometry
    ):
        clean_geometry = _repair_as_multipolygon(
            geometry
        )

        if clean_geometry is None:
            dropped_before_count += 1

            print(
                f"WARNING: feature {position} di {layer_name} "
                "vuota o non poligonale; feature eliminata."
            )
            continue

        scale_factor = _sample_geometric_scale(
            rng,
            maximum_delta
        )

        try:
            noisy_geometry = shapely_scale(
                clean_geometry,
                xfact=scale_factor,
                yfact=scale_factor,
                origin="centroid"
            )
        except Exception:
            noisy_geometry = None

        noisy_geometry = _repair_as_multipolygon(
            noisy_geometry
        )

        if noisy_geometry is None:
            noisy_geometry = clean_geometry
            fallback_count += 1
        elif scale_factor > 1:
            enlarged_count += 1
        else:
            reduced_count += 1

        kept_positions.append(position)
        clean_metric_geometries.append(
            clean_geometry
        )
        noisy_metric_geometries.append(
            noisy_geometry
        )

    if not kept_positions:
        raise RuntimeError(
            f"Nessuna geometria poligonale utilizzabile "
            f"nel layer {layer_name}."
        )

    # Layer metrico con rumore.
    noisy_metric_gdf = metric_gdf.iloc[
        kept_positions
    ].copy().reset_index(drop=True)

    noisy_metric_gdf.geometry = gpd.GeoSeries(
        noisy_metric_geometries,
        crs=metric_gdf.crs
    )

    # Layer metrico pulito, usato come fallback.
    clean_metric_gdf = metric_gdf.iloc[
        kept_positions
    ].copy().reset_index(drop=True)

    clean_metric_gdf.geometry = gpd.GeoSeries(
        clean_metric_geometries,
        crs=metric_gdf.crs
    )

    # Ritorno al CRS originale.
    noisy_original_crs = noisy_metric_gdf.to_crs(
        original_crs
    )

    clean_original_crs = clean_metric_gdf.to_crs(
        original_crs
    )

    final_positions = []
    final_geometries = []
    dropped_after_count = 0

    for position in range(
        len(noisy_original_crs)
    ):
        final_geometry = _repair_as_multipolygon(
            noisy_original_crs.geometry.iloc[position]
        )

        # Se la riproiezione della geometria rumorosa non è valida,
        # conserva la geometria pulita.
        if final_geometry is None:
            final_geometry = _repair_as_multipolygon(
                clean_original_crs.geometry.iloc[position]
            )

            fallback_count += 1

        if final_geometry is None:
            dropped_after_count += 1

            print(
                f"WARNING: feature {position} di {layer_name} "
                "non utilizzabile dopo la riproiezione; "
                "feature eliminata."
            )
            continue

        final_positions.append(position)
        final_geometries.append(
            final_geometry
        )

    final_gdf = noisy_original_crs.iloc[
        final_positions
    ].copy().reset_index(drop=True)

    final_gdf.geometry = gpd.GeoSeries(
        final_geometries,
        crs=original_crs
    )

    if final_gdf.empty:
        raise RuntimeError(
            f"Il layer geometrico finale {layer_name} è vuoto."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if output_path.exists():
        output_path.unlink()

    if output_path.suffix.lower() == ".gpkg":
        final_gdf.to_file(
            str(output_path),
            layer=output_path.stem,
            driver="GPKG"
        )
    else:
        final_gdf.to_file(
            str(output_path),
            driver="GeoJSON"
        )

    qgis_layer = QgsVectorLayer(
        str(output_path),
        layer_name,
        "ogr"
    )

    if not qgis_layer.isValid():
        raise RuntimeError(
            f"Layer QGIS rumoroso non valido: "
            f"{output_path}"
        )

    print(
        f"{layer_name}: "
        f"{reduced_count} feature rimpicciolite; "
        f"{enlarged_count} feature ingrandite; "
        f"{fallback_count} fallback alla geometria originale; "
        f"{dropped_before_count} feature eliminate prima; "
        f"{dropped_after_count} feature eliminate dopo."
    )

    return qgis_layer





############################################################################################

def verde_gauss_noise(grid, noise_level, seed=42):
    np.random.seed(seed)

    sigma_map = {"1": 0.5, "2": 1.0, "3": 1.5, "4": 2.5, "5": 4.0}
    sigma = sigma_map.get(str(noise_level))
    if sigma is None:
        raise ValueError(f"noise_level non valido: {noise_level}")

    fld_area_idx = grid.fields().indexOf("verde_area")
    fld_pc_idx = grid.fields().indexOf("verde_pc")

    if fld_area_idx == -1 or fld_pc_idx == -1:
        raise ValueError("Campi verde_area e/o verde_pc non trovati nel layer.")

    feats = list(grid.getFeatures())
    area_values = np.array(
        [0.0 if f["verde_area"] is None else float(f["verde_area"]) for f in feats],
        dtype=float
    )

    noise = np.random.normal(0, sigma, len(area_values))
    new_area_vals = np.clip(area_values + noise, 0, None)

    geom_areas = np.array(
        [0.0 if f.geometry() is None else float(f.geometry().area()) for f in feats],
        dtype=float
    )
    new_pc_vals = np.where(geom_areas > 0, 100.0 * new_area_vals / geom_areas, 0.0)

    dp = grid.dataProvider()
    grid.startEditing()

    changes = {}
    for f, new_area, new_pc in zip(feats, new_area_vals, new_pc_vals):
        fid = f.id()
        changes[fid] = {
            fld_area_idx: float(new_area),
            fld_pc_idx: float(new_pc),
        }

    dp.changeAttributeValues(changes)
    grid.commitChanges()
    return grid

def verde_multiplicative_noise(grid, noise_level, seed=42):
    rng = np.random.default_rng(seed)

    sigma_map = {"1": 0.05, "2": 0.10, "3": 0.20, "4": 0.30, "5": 0.40}
    sigma = sigma_map.get(str(noise_level))
    if sigma is None:
        raise ValueError(f"noise_level non valido: {noise_level}")

    fld_area_idx = grid.fields().indexOf("verde_area")
    fld_pc_idx = grid.fields().indexOf("verde_pc")

    if fld_area_idx == -1 or fld_pc_idx == -1:
        raise ValueError("Campi verde_area e/o verde_pc non trovati nel layer.")

    feats = list(grid.getFeatures())
    area_values = np.array(
        [0.0 if f["verde_area"] is None else float(f["verde_area"]) for f in feats],
        dtype=float
    )

    factors = rng.normal(1.0, sigma, len(area_values))
    new_area_vals = np.clip(area_values * factors, 0, None)

    geom_areas = np.array(
        [0.0 if f.geometry() is None else float(f.geometry().area()) for f in feats],
        dtype=float
    )
    new_pc_vals = np.where(geom_areas > 0, 100.0 * new_area_vals / geom_areas, 0.0)

    dp = grid.dataProvider()
    grid.startEditing()

    changes = {}
    for f, new_area, new_pc in zip(feats, new_area_vals, new_pc_vals):
        fid = f.id()
        changes[fid] = {
            fld_area_idx: float(new_area),
            fld_pc_idx: float(new_pc),
        }

    dp.changeAttributeValues(changes)
    grid.commitChanges()
    return grid

########################################################################################################
def _apply_area_noise(grid, area_field, pc_field, noise_level, mode="gauss", seed=42):
    """
    Applica rumore ai campi di area e percentuale di copertura di un layer QGIS.

    Parameters
    ----------
    grid : QgsVectorLayer
        Layer su cui modificare gli attributi.
    area_field : str
        Nome del campo area da perturbare.
    pc_field : str
        Nome del campo percentuale da ricalcolare.
    noise_level : str
        Livello di rumore ("1", "2", "3", "4", "5").
    mode : str
        "gauss" oppure "multiplicative".
    seed : int
        Seed per la riproducibilità.

    Returns
    -------
    QgsVectorLayer
        Layer aggiornato.
    """
    fld_area_idx = grid.fields().indexOf(area_field)
    fld_pc_idx = grid.fields().indexOf(pc_field)

    if fld_area_idx == -1 or fld_pc_idx == -1:
        raise ValueError(
            f"Campi non trovati nel layer: {area_field} / {pc_field}"
        )

    feats = list(grid.getFeatures())
    area_values = np.array(
        [0.0 if f[area_field] is None else float(f[area_field]) for f in feats],
        dtype=float
    )

    if mode == "gauss":
        np.random.seed(seed)
        sigma_map = {"1": 0.5, "2": 1.0, "3": 1.5, "4": 2.5, "5": 4.0}
        sigma = sigma_map.get(str(noise_level))
        if sigma is None:
            raise ValueError(f"noise_level non valido: {noise_level}")
        new_area_vals = np.clip(area_values + np.random.normal(0, sigma, len(area_values)), 0, None)

    elif mode == "multiplicative":
        rng = np.random.default_rng(seed)
        sigma_map = {"1": 0.05, "2": 0.10, "3": 0.20, "4": 0.30, "5": 0.40}
        sigma = sigma_map.get(str(noise_level))
        if sigma is None:
            raise ValueError(f"noise_level non valido: {noise_level}")
        factors = rng.normal(1.0, sigma, len(area_values))
        new_area_vals = np.clip(area_values * factors, 0, None)

    else:
        raise ValueError(f"mode non valido: {mode}")

    geom_areas = np.array(
        [0.0 if f.geometry() is None else float(f.geometry().area()) for f in feats],
        dtype=float
    )
    new_pc_vals = np.where(geom_areas > 0, 100.0 * new_area_vals / geom_areas, 0.0)

    dp = grid.dataProvider()
    grid.startEditing()

    changes = {}
    for f, new_area, new_pc in zip(feats, new_area_vals, new_pc_vals):
        changes[f.id()] = {
            fld_area_idx: float(new_area),
            fld_pc_idx: float(new_pc),
        }

    dp.changeAttributeValues(changes)
    grid.commitChanges()
    return grid


def ext_space_gauss_noise(grid, noise_level, seed=42):
    return _apply_area_noise(
        grid,
        area_field="Single parts_area",
        pc_field="Single parts_pc",
        noise_level=noise_level,
        mode="gauss",
        seed=seed,
    )


def ext_space_multiplicative_noise(grid, noise_level, seed=42):
    return _apply_area_noise(
        grid,
        area_field="Single parts_area",
        pc_field="Single parts_pc",
        noise_level=noise_level,
        mode="multiplicative",
        seed=seed,
    )


def street_space_gauss_noise(grid, noise_level, seed=42):
    return _apply_area_noise(
        grid,
        area_field="aree-stradali-modified_area",
        pc_field="aree-stradali-modified_pc",
        noise_level=noise_level,
        mode="gauss",
        seed=seed,
    )


def street_space_multiplicative_noise(grid, noise_level, seed=42):
    return _apply_area_noise(
        grid,
        area_field="aree-stradali-modified_area",
        pc_field="aree-stradali-modified_pc",
        noise_level=noise_level,
        mode="multiplicative",
        seed=seed,
    )

########################################################################################################

def create_ferrovia(raw_data_path, processed_data_path, verde_layer):
    railway = gpd.read_file(
        str(raw_data_path.joinpath('uso_suolo_bologna.geojson'))
    )

    railway = railway[railway['DESCR'] == 'Reti ferroviare']

    railway.to_file(
        str(processed_data_path.joinpath('ferrovia.geojson')),
        driver='GeoJSON'
    )

    railway = processing.run(
        "native:multidifference",
        {
            'INPUT': str(processed_data_path.joinpath('ferrovia.geojson')),
            'OVERLAYS': [
                str(raw_data_path.joinpath('rifter_edif_pl.geojson')),
                verde_layer
            ],
            'OUTPUT': 'TEMPORARY_OUTPUT'
        }
    )['OUTPUT']

    return railway


def main(bologna_size, noise_type, noise_source, aggregation_method, noise_level):

    if bologna_size == 'full':
        loc_abitativa = gpd.read_file(RAW_DATA_DIR_PATH.joinpath('localita_abitative.gpkg'), layer="V_LAB_GPG")
        loc_abitativa = loc_abitativa[loc_abitativa["NM_LAB"] == "BOLOGNA"]
        loc_abitativa.to_file(PROCESSED_DATA_DIR_PATH.joinpath('localita_abitative.geojson'), driver="GeoJSON")
        
        area_abitata = processing.run("native:clip", {'INPUT':str(RAW_DATA_DIR_PATH.joinpath('aree-statistiche.geojson')),'OVERLAY':str(PROCESSED_DATA_DIR_PATH.joinpath('localita_abitative.geojson')),'OUTPUT':'TEMPORARY_OUTPUT'})['OUTPUT']
        save_layer(area_abitata, 'area_abitata_full', PROCESSED_DATA_DIR_PATH)
        area_abitata = gpd.read_file(PROCESSED_DATA_DIR_PATH.joinpath('area_abitata_full.geojson'))
        area_abitata = area_abitata[~ area_abitata["area_statistica"].isin(["RIGOSA", "LAVINO DI MEZZO", "AEROPORTO", "BARGELLINO", "VIA DEL VIVAIO", "LA BIRRA", "LA NOCE", "TIRO A SEGNO", "LAGHETTI DEL ROSARIO", "SAVENA ABBANDONATO", "MULINO DEL GOMITO", "CADRIANO-CALAMOSCO", "FIERA", "STRADELLI GUELFI", "LUNGO SAVENA", "OSPEDALE BELLARIA", "MONTE DONATO", "PONTE SAVENA-LA BASTIA", "PADERNO", "RAVONE", "VIA DEL GENIO", "SAN LUCA", "LUNGO RENO", "CAAB"])]
        area_abitata.to_file(PROCESSED_DATA_DIR_PATH.joinpath('area_abitata_full.geojson'), driver="GeoJSON")

        # Create the grid for the entire city
        grid_bologna = processing.run("native:creategrid", {'TYPE':2,'EXTENT':str(PROCESSED_DATA_DIR_PATH.joinpath('area_abitata_full.geojson')),'HSPACING':100,'VSPACING':100,'HOVERLAY':0,'VOVERLAY':0,'CRS':QgsCoordinateReferenceSystem('EPSG:3857'),'OUTPUT':'TEMPORARY_OUTPUT'})['OUTPUT']
        grid_bologna = processing.run("native:clip", {'INPUT':grid_bologna,'OVERLAY':str(PROCESSED_DATA_DIR_PATH.joinpath('area_abitata_full.geojson')),'OUTPUT':'TEMPORARY_OUTPUT'})['OUTPUT']

        aree_statistiche = processing.run("native:intersection", {'INPUT':grid_bologna,'OVERLAY':str(PROCESSED_DATA_DIR_PATH.joinpath('area_abitata_full.geojson')),'INPUT_FIELDS':[],'OVERLAY_FIELDS':[],'OVERLAY_FIELDS_PREFIX':'','OUTPUT':'TEMPORARY_OUTPUT','GRID_SIZE':None})['OUTPUT']
    
    else:
        loc_abitativa = gpd.read_file(RAW_DATA_DIR_PATH.joinpath('aree-statistiche.geojson'))
        loc_abitativa = loc_abitativa[loc_abitativa["area_statistica"].isin(["IRNERIO-1", "IRNERIO-2", "MARCONI-1", "MARCONI-2", "MALPIGHI-1", "MALPIGHI-2", "GALVANI-1", "GALVANI-2"])]
        loc_abitativa.to_file(PROCESSED_DATA_DIR_PATH.joinpath('localita_abitative.geojson'), driver="GeoJSON")

        area_abitata = processing.run("native:clip", {'INPUT':str(RAW_DATA_DIR_PATH.joinpath('aree-statistiche.geojson')),'OVERLAY':str(PROCESSED_DATA_DIR_PATH.joinpath('localita_abitative.geojson')),'OUTPUT':'TEMPORARY_OUTPUT'})['OUTPUT']

        # Create the grid for the city center
        grid_bologna = processing.run("native:creategrid", {'TYPE':2,'EXTENT':area_abitata,'HSPACING':100,'VSPACING':100,'HOVERLAY':0,'VOVERLAY':0,'CRS':QgsCoordinateReferenceSystem('EPSG:3857'),'OUTPUT':'TEMPORARY_OUTPUT'})['OUTPUT']
        grid_bologna = processing.run("native:clip", {'INPUT':grid_bologna,'OVERLAY':area_abitata,'OUTPUT':'TEMPORARY_OUTPUT'})['OUTPUT']

        aree_statistiche = processing.run("native:intersection", {'INPUT':grid_bologna,'OVERLAY':area_abitata,'INPUT_FIELDS':[],'OVERLAY_FIELDS':[],'OVERLAY_FIELDS_PREFIX':'','OUTPUT':'TEMPORARY_OUTPUT','GRID_SIZE':None})['OUTPUT']

    # Compute the area of each area_statistica for each grid in bologna 
    aree_statistiche = processing.run("native:fieldcalculator", {'INPUT':aree_statistiche,'FIELD_NAME':'intersect_area_statistica','FIELD_TYPE':0,'FIELD_LENGTH':0,'FIELD_PRECISION':0,'FORMULA':'area($geometry)','OUTPUT':'TEMPORARY_OUTPUT'})['OUTPUT']
    save_layer(aree_statistiche, 'aree_statistiche_grid', GRID_DIR_PATH)

    # Create the green vector layer
    verde = create_verde(
        RAW_DATA_DIR_PATH,
        PROCESSED_DATA_DIR_PATH
    )

    if (
        noise_type == "geometric"
        and noise_source == "green_area"
    ):
        verde = inject_polygon_geometric_noise(
            input_path=PROCESSED_DATA_DIR_PATH.joinpath(
                "verde.gpkg"
            ),
            noise_level=noise_level,
            output_path=PROCESSED_DATA_DIR_PATH.joinpath(
                "verde_noisy.geojson"
            ),
            layer_name="verde",
            seed=42
        )

    # Create the railway vector layer
    railway = create_ferrovia(
        RAW_DATA_DIR_PATH,
        PROCESSED_DATA_DIR_PATH,
        verde
    )
    
    save_layer(railway, 'ferrovia', PROCESSED_DATA_DIR_PATH)

    # Clean the street data
    strade_clean = processing.run("native:multidifference", {'INPUT':str(PROCESSED_DATA_DIR_PATH.joinpath("aree-stradali-modified.geojson")),'OVERLAYS':[verde, railway],'OUTPUT':'TEMPORARY_OUTPUT'})['OUTPUT']
    save_layer(strade_clean, 'aree-stradali-modified', PROCESSED_DATA_DIR_PATH)

    # Compute the free space in which is possible to create new green cells
    available_space = gpd.read_file(str(RAW_DATA_DIR_PATH.joinpath('uso_suolo_bologna.geojson')))
    available_space = available_space[available_space['DESCR'].isin(['Tessuto residenziale compatto e denso', 'Tessuto residenziale urbano', 'Tessuto residenziale  rado'])]
    available_space.to_file(str(PROCESSED_DATA_DIR_PATH.joinpath('free_space.geojson')), driver='GeoJSON')

    free_space = processing.run("native:multidifference", {'INPUT':str(PROCESSED_DATA_DIR_PATH.joinpath('free_space.geojson')),'OVERLAYS':[verde, str(RAW_DATA_DIR_PATH.joinpath('aree-stradali.geojson')), str(RAW_DATA_DIR_PATH.joinpath('rifter_edif_pl.geojson')), railway],'OUTPUT':'TEMPORARY_OUTPUT'})['OUTPUT']
    free_space = processing.run("native:multiparttosingleparts", {'INPUT':free_space,'OUTPUT':'TEMPORARY_OUTPUT'})['OUTPUT']
    save_layer(free_space, 'free_space', PROCESSED_DATA_DIR_PATH)

    # -------------------------------------------------------------------------
    # Seleziona i layer da usare: puliti oppure con rumore geometrico
    # -------------------------------------------------------------------------

    strade_clean_path = (
        PROCESSED_DATA_DIR_PATH
        / "aree-stradali-modified.geojson"
    )

    free_space_path = (
        PROCESSED_DATA_DIR_PATH
        / "free_space.geojson"
    )

    # STREET SPACE
    if (
        noise_type == "geometric"
        and noise_source == "street_space"
    ):
        strade_noisy_path = (
            PROCESSED_DATA_DIR_PATH
            / "aree-stradali-modified-noisy.geojson"
        )

        inject_polygon_geometric_noise(
            input_path=strade_clean_path,
            noise_level=noise_level,
            output_path=strade_noisy_path,
            layer_name="aree-stradali-modified",
            seed=43
        )

        strade_clean_path = strade_noisy_path

    # EXTERNAL / FREE SPACE
    if (
        noise_type == "geometric"
        and noise_source == "ext_space"
    ):
        free_space_noisy_path = (
            PROCESSED_DATA_DIR_PATH
            / "free_space-noisy.geojson"
        )

        inject_polygon_geometric_noise(
            input_path=free_space_path,
            noise_level=noise_level,
            output_path=free_space_noisy_path,
            layer_name="Single parts",
            seed=44
        )

        free_space_path = free_space_noisy_path

    # Caricamento definitivo dei layer selezionati
    strade_clean_layer = QgsVectorLayer(
        str(strade_clean_path),
        "aree-stradali-modified",
        "ogr"
    )

    if not strade_clean_layer.isValid():
        raise ValueError(
            f"Layer delle strade non valido: {strade_clean_path}"
        )

    free_space_layer = QgsVectorLayer(
        str(free_space_path),
        "Single parts",
        "ogr"
    )

    if not free_space_layer.isValid():
        raise ValueError(
            f"Layer free space non valido: {free_space_path}"
        )



    # -------------------------------------------------------------------------
    # Carica i layer selezionati
    #
    # I nomi sono impostati esplicitamente per mantenere i nomi dei campi:
    # - aree-stradali-modified_area
    # - aree-stradali-modified_pc
    # - Single parts_area
    # - Single parts_pc
    # -------------------------------------------------------------------------

    strade_clean_layer = QgsVectorLayer(
        str(strade_clean_path),
        "aree-stradali-modified",
        "ogr"
    )

    if not strade_clean_layer.isValid():
        raise ValueError(
            f"Layer delle strade non valido: {strade_clean_path}"
        )

    free_space_layer = QgsVectorLayer(
        str(free_space_path),
        "Single parts",
        "ogr"
    )

    if not free_space_layer.isValid():
        raise ValueError(
            f"Layer free space non valido: {free_space_path}"
        )


    # -------------------------------------------------------------------------
    # Calcola gli overlap usando i layer puliti o rumorosi selezionati sopra
    # -------------------------------------------------------------------------

    grid_with_areas = processing.run(
        "native:calculatevectoroverlaps",
        {
            'INPUT': grid_bologna,
            'LAYERS': [
                verde,
                str(
                    RAW_DATA_DIR_PATH.joinpath(
                        'rifter_edif_pl.geojson'
                    )
                ),
                strade_clean_layer,
                free_space_layer,
                railway
            ],
            'OUTPUT': 'TEMPORARY_OUTPUT',
            'GRID_SIZE': None
        }
    )['OUTPUT']


    # -------------------------------------------------------------------------
    # Salva il free space usato realmente in GeoPackage
    #
    # Se noise_source == "ext_space" e noise_type == "geometric",
    # viene indicizzato il free space rumoroso.
    # Altrimenti viene indicizzato quello pulito.
    # -------------------------------------------------------------------------

    free_space_indexed_path = (
        PROCESSED_DATA_DIR_PATH /
        "free_space_indexed.gpkg"
    )

    if free_space_indexed_path.exists():
        free_space_indexed_path.unlink()

    processing.run(
        "native:savefeatures",
        {
            "INPUT": free_space_layer,
            "OUTPUT": str(free_space_indexed_path)
        }
    )

    free_space_indexed = QgsVectorLayer(
        str(free_space_indexed_path),
        "free_space_indexed",
        "ogr"
    )

    if not free_space_indexed.isValid():
        raise ValueError(
            f"Layer free space indicizzato non valido: "
            f"{free_space_indexed_path}"
        )


    # -------------------------------------------------------------------------
    # Conteggio ottimizzato
    # -------------------------------------------------------------------------

    grid_with_areas = processing.run(
        "native:countpointsinpolygon",
        {
            'POLYGONS': grid_with_areas,
            'POINTS': free_space_indexed,
            'WEIGHT': '',
            'CLASSFIELD': '',
            'FIELD': 'free_space_number',
            'OUTPUT': 'TEMPORARY_OUTPUT'
        }
    )['OUTPUT']

    if noise_source == "green_area":
        if noise_type == "gauss":
            grid_with_areas = verde_gauss_noise(grid_with_areas, noise_level)
        elif noise_type == "multiplicative":
            grid_with_areas = verde_multiplicative_noise(grid_with_areas, noise_level)
    
    # free_space
    if noise_source == "street_space":
        if noise_type == "gauss":
            grid_with_areas = street_space_gauss_noise(grid_with_areas, noise_level)
        elif noise_type == "multiplicative":
            grid_with_areas = street_space_multiplicative_noise(grid_with_areas, noise_level)
    
    # aree-stardali_modified
    if noise_source == "ext_space":
        if noise_type == "gauss":
            grid_with_areas = ext_space_gauss_noise(grid_with_areas, noise_level)
        elif noise_type == "multiplicative":
            grid_with_areas = ext_space_multiplicative_noise(grid_with_areas, noise_level)
    
    if noise_source == "num_areas": 
        if noise_type == "gauss":
            grid_with_areas = apply_num_areas_gauss(noise_level, grid_with_areas)
        if noise_type == "missingness":
            grid_with_areas = apply_num_areas_missingness(noise_level, grid_with_areas)

    # UHEI and NDVI computation
    uhei_layer = processing.run("native:fieldcalculator", {'INPUT':str(RAW_DATA_DIR_PATH.joinpath('indexes', 'urban_heat_exposure_index.geojson')),'FIELD_NAME':'uhei','FIELD_TYPE':0,'FIELD_LENGTH':0,'FIELD_PRECISION':0,'FORMULA':'to_real(value)','OUTPUT':'TEMPORARY_OUTPUT'})['OUTPUT']
    ndvi_layer = processing.run("native:fieldcalculator", {'INPUT':str(RAW_DATA_DIR_PATH.joinpath('indexes', 'ndvi_mean.geojson')),'FIELD_NAME':'ndvi','FIELD_TYPE':0,'FIELD_LENGTH':0,'FIELD_PRECISION':0,'FORMULA':'to_real(value)','OUTPUT':'TEMPORARY_OUTPUT'})['OUTPUT']

    if noise_type == "aggregation":
        if noise_source == "uhei":
            if aggregation_method == "mean":
                grid_with_areas = compute_mean_uhei(grid_with_areas, uhei_layer)
                grid_with_areas.to_file(str(GRID_DIR_PATH.joinpath("grid_with_areas.geojson")), driver="GeoJSON")
                grid_with_areas = compute_weighted_ndvi(str(GRID_DIR_PATH.joinpath("grid_with_areas.geojson")), ndvi_layer)
                grid_with_areas.to_file(str(GRID_DIR_PATH.joinpath("grid_with_areas.geojson")), driver="GeoJSON")
            else: 
                grid_with_areas = compute_median_uhei(grid_with_areas, uhei_layer)
                grid_with_areas.to_file(str(GRID_DIR_PATH.joinpath("grid_with_areas.geojson")), driver="GeoJSON")
                grid_with_areas = compute_weighted_ndvi(str(GRID_DIR_PATH.joinpath("grid_with_areas.geojson")), ndvi_layer)
                grid_with_areas.to_file(str(GRID_DIR_PATH.joinpath("grid_with_areas.geojson")), driver="GeoJSON")
        else:
            if aggregation_method == "mean":
                grid_with_areas = compute_weighted_uhei(grid_with_areas, uhei_layer)
                grid_with_areas.to_file(str(GRID_DIR_PATH.joinpath("grid_with_areas.geojson")), driver="GeoJSON")
                grid_with_areas = compute_mean_ndvi(str(GRID_DIR_PATH.joinpath("grid_with_areas.geojson")), ndvi_layer)
                grid_with_areas.to_file(str(GRID_DIR_PATH.joinpath("grid_with_areas.geojson")), driver="GeoJSON")
            else: 
                grid_with_areas = compute_weighted_uhei(grid_with_areas, uhei_layer)
                grid_with_areas.to_file(str(GRID_DIR_PATH.joinpath("grid_with_areas.geojson")), driver="GeoJSON")
                grid_with_areas = compute_median_ndvi(str(GRID_DIR_PATH.joinpath("grid_with_areas.geojson")), ndvi_layer)
                grid_with_areas.to_file(str(GRID_DIR_PATH.joinpath("grid_with_areas.geojson")), driver="GeoJSON")
    else:
        grid_with_areas = compute_weighted_uhei(grid_with_areas, uhei_layer)
        grid_with_areas.to_file(str(GRID_DIR_PATH.joinpath("grid_with_areas.geojson")), driver="GeoJSON")
        grid_with_areas = compute_weighted_ndvi(str(GRID_DIR_PATH.joinpath("grid_with_areas.geojson")), ndvi_layer)
        grid_with_areas.to_file(str(GRID_DIR_PATH.joinpath("grid_with_areas.geojson")), driver="GeoJSON")

    # Compute the number of trees outside the green areas

    trees_path = str(RAW_DATA_DIR_PATH.joinpath('alberi-manutenzioni.fgb'))
    trees_outside_path = str(PROCESSED_DATA_DIR_PATH.joinpath("trees_outside_green_points.gpkg"))

    trees_outside_green_points = processing.run(
        "native:difference",
        {
            'INPUT': trees_path,
            'OVERLAY': verde,
            'OUTPUT': trees_outside_path,
            'GRID_SIZE': None
        }
    )['OUTPUT']


    trees_outside_green_points_layer = QgsVectorLayer(
        trees_outside_path,
        "trees_outside_green_points",
        "ogr"
    )

    trees_outside_green = processing.run(
        "native:countpointsinpolygon",
        {
            'POLYGONS': grid_bologna,
            'POINTS': trees_outside_green_points_layer,
            'FIELD': 'tree_number',
            'CLASSFIELD': None,
            'WEIGHT': None,
            'OUTPUT': 'TEMPORARY_OUTPUT'
        }
    )['OUTPUT']

    save_layer(trees_outside_green, 'trees_outside_green', GRID_DIR_PATH)

    if noise_source == "trees":
        apply_noise_trees(noise_level, GRID_DIR_PATH)

    # Integrate all the computed data in a single grid
    grid = gpd.read_file(GRID_DIR_PATH.joinpath('grid_with_areas.geojson'))
    trees = gpd.read_file(GRID_DIR_PATH.joinpath('trees_outside_green.geojson'))[['id', 'tree_number']]
    trees = trees[~trees['id'].isin(list(set(trees['id']) - set(grid['id'])))]

    final_grid = grid.merge(trees, how='outer', on='id').fillna(0)
    final_grid = final_grid.rename(columns={
        'Single parts_area': 'free_space_area',
        'Single parts_pc': 'free_space_pc',
        'Difference_area': 'railways_area',
        'Difference_pc': 'railways_pc',
        'verde_area': 'green_area',
        'verde_pc': 'green_pc',
        'rifter_edif_pl_area': 'buildings_area',
        'rifter_edif_pl_pc': 'buildings_pc',
        'aree-stradali-modified_area': 'road_area',
        'aree-stradali-modified_pc': 'road_pc',
        'weighted_uhei_sum': 'weighted_uhei',
        'weighted_ndvi_sum': 'weighted_ndvi',
    })
    uhei_max = final_grid['weighted_uhei'].max()
    final_grid['inverse_uhei'] = uhei_max - final_grid['weighted_uhei']
    final_grid.to_file(GRID_DIR_PATH.joinpath("final_grid.geojson"), driver="GeoJSON")
    

    # Compute the green area in each statistical area
    aree_statistiche = processing.run("native:fieldcalculator", {'INPUT':str(RAW_DATA_DIR_PATH.joinpath('aree-statistiche.geojson')),'FIELD_NAME':'area','FIELD_TYPE':0,'FIELD_LENGTH':0,'FIELD_PRECISION':0,'FORMULA':'$area','OUTPUT':'TEMPORARY_OUTPUT'})['OUTPUT']
    aree_statistiche = processing.run("native:calculatevectoroverlaps", {'INPUT':aree_statistiche,'LAYERS':[verde],'OUTPUT':'TEMPORARY_OUTPUT','GRID_SIZE':None})['OUTPUT']
    
    aree_statistiche = processing.run("native:joinbylocationsummary", {'INPUT':aree_statistiche,'PREDICATE':[0],'JOIN':uhei_layer,'JOIN_FIELDS':['uhei'],'SUMMARIES':[6],'DISCARD_NONMATCHING':False,'OUTPUT':'TEMPORARY_OUTPUT'})['OUTPUT']
    aree_statistiche = processing.run("native:joinbylocationsummary", {'INPUT':aree_statistiche,'PREDICATE':[0],'JOIN':ndvi_layer,'JOIN_FIELDS':['ndvi'],'SUMMARIES':[6],'DISCARD_NONMATCHING':False,'OUTPUT':'TEMPORARY_OUTPUT'})['OUTPUT']
    save_layer(aree_statistiche, 'aree_statistiche_stat', PROCESSED_DATA_DIR_PATH)

    aree_statistiche = gpd.read_file(PROCESSED_DATA_DIR_PATH.joinpath('aree_statistiche_stat.geojson'))
    aree_statistiche = aree_statistiche.rename(columns={
        'uhei_mean': 'uhei',
        'ndvi_mean': 'ndvi',
    })
    aree_statistiche.to_file(PROCESSED_DATA_DIR_PATH.joinpath("aree_statistiche_stat.geojson"), driver="GeoJSON")


if __name__ == "__main__":
    global RAW_DATA_DIR_PATH, PROCESSED_DATA_DIR_PATH, GRID_DIR_PATH
    RAW_DATA_DIR_PATH = WORKING_DIR_PATH.joinpath("dataset", "raw_data")
    PROCESSED_DATA_DIR_PATH = WORKING_DIR_PATH.joinpath("dataset", "processed_data")

    parser = argparse.ArgumentParser(description="Extraction of information from data through GIS-based processing algorithms.")
    parser.add_argument("--size", 
                        type=str, 
                        default="full", 
                        choices=["center", "full"], 
                        help="Whether to run the model on the city center or on the entire cityscape.")
    parser.add_argument("--noise_type", dest="noise_type", type=str, default="",
                        choices=["gauss", "multiplicative", "aggregation", "mislabelling", "missingness", "geometric"],
                        help="Type of noise to apply.")
    
    parser.add_argument(
        "--noise_source",
        dest="noise_source",
        type=str,
        default="",
        choices=["uhei", "ndvi", "land_use", "trees", "num_areas", "green_area", "street_space", "ext_space", "population_density"],
        help="Where to add noise."
    )
    
    parser.add_argument("--noise_level", dest="noise_level", type=str, default="", choices=["1", "2", "3", "4", "5"],
                        help="Noise level.")
    parser.add_argument("--aggregation_method", dest="aggregation_method", type=str, default="", choices=["mean", "median"],
                        help="Aggregation method to use in case the noise to apply is aggregation")
    args = parser.parse_args()
    
    if args.noise_type != "":
        # Forza la presenza di noise_level se è stato specificato un tipo di rumore che non sia 'aggregation'
        if args.noise_type != "aggregation" and args.noise_level == "":
            parser.error(f"--noise_level è obbligatorio quando --noise_type è impostato su {args.noise_type}")

        if args.noise_type == "aggregation":
            processed_filename = "processed_data_aggregation_" + args.noise_source + "_" + args.aggregation_method
            PROCESSED_DATA_DIR_PATH = WORKING_DIR_PATH.joinpath("dataset", processed_filename)
        
        # Escludiamo 'population_density' dalla missingness qui, perché i suoi file raw vengono creati da process_data.py
        # Includiamo 'ext_space' e 'street_space' assieme a 'green_area' perché il loro rumore è on-the-fly.
        elif (args.noise_type in ["missingness", "geometric"] and args.noise_source != "population_density") or \
             (args.noise_type == "gauss" and args.noise_source == "num_areas") or \
             (args.noise_source in ["green_area", "ext_space", "street_space"]):
             
            processed_filename = "processed_data_" + args.noise_type + "_" + args.noise_source + "_" + "lvl" + args.noise_level
            PROCESSED_DATA_DIR_PATH = WORKING_DIR_PATH.joinpath("dataset", processed_filename)
            
        else:
            # Questo blocco gestisce correttamente i pre-processati: uhei, ndvi, land_use e population_density
            raw_filename = "raw_data_" + args.noise_type + "_" + args.noise_source + "_" + "lvl" + args.noise_level
            processed_filename = "processed_data_" + args.noise_type + "_" + args.noise_source + "_" + "lvl" + args.noise_level
            RAW_DATA_DIR_PATH = WORKING_DIR_PATH.joinpath("dataset", raw_filename)
            PROCESSED_DATA_DIR_PATH = WORKING_DIR_PATH.joinpath("dataset", processed_filename)
    
    if args.size == "center":
        GRID_DIR_PATH = PROCESSED_DATA_DIR_PATH.joinpath("center")
    else:
        GRID_DIR_PATH = PROCESSED_DATA_DIR_PATH.joinpath("full")
    
    PROCESSED_DATA_DIR_PATH.mkdir(parents=True, exist_ok=True)
    GRID_DIR_PATH.mkdir(parents=True, exist_ok=True)
    
    streets_processing()
    main(
        args.size,
        args.noise_type,
        args.noise_source,
        args.aggregation_method,
        args.noise_level
    )
