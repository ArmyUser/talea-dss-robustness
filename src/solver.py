from minizinc import Instance, Model, Solver
import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
import argparse


WORKING_DIR_PATH = Path.cwd()
MINIZINC_DIR_PATH = WORKING_DIR_PATH.joinpath("src", "Minizinc")
INSTANCES_DIR_PATH = MINIZINC_DIR_PATH.joinpath("instances")
MODELS_DIR_PATH = MINIZINC_DIR_PATH.joinpath("models")


def solve(size, model_name, res_name):
    """
    Solves the linear model for green cells placement optimization for the specified size.

    Input:
    - size: str, specifying whether to run the model on the city center or on the entire cityscape
    - model_name: str, choice of the model to run
    - res_name: str, name of the result file to be saved

    """

    model = Model(MODELS_DIR_PATH.joinpath(f"{model_name}_model.mzn"))
    solver = Solver.lookup('highs')
    instance = Instance(solver, model)
    instance.add_file(INSTANCES_DIR_PATH.joinpath(f"{model_name}_instance.dzn"))

    result = instance.solve()
    street_cells = np.array(result.solution.street_cells)
    yard_cells = np.array(result.solution.yard_cells)
    utility = np.array(result.solution.utility)

    df_result = pd.DataFrame(columns=['id', 'street_cells', 'yard_cells', 'utility'])
    for i in range(len(street_cells)):
        df_result.loc[len(df_result)] = {
            'id': None,
            'street_cells': street_cells[i],
            'yard_cells': yard_cells[i],
            'utility': utility[i]
        }

    if size == "center":
        GRID_DIR_PATH = PROCESSED_DATA_DIR_PATH.joinpath("center")
    elif size == "full":
        GRID_DIR_PATH = PROCESSED_DATA_DIR_PATH.joinpath("full")
    
    gdf_data = gpd.read_file(GRID_DIR_PATH.joinpath("final_grid.geojson"))
    gdf_result = gpd.GeoDataFrame(df_result, geometry=gdf_data.geometry, crs='EPSG:3857')
    gdf_result['id'] = gdf_data['id']
    gdf_result['num_areas'] = gdf_data['free_space_number']
    gdf_result_1 = gdf_result[gdf_result['street_cells'] != 0]
    gdf_result_2 = gdf_result[gdf_result['yard_cells'] != 0]
    gdf_result = pd.merge(gdf_result_1, gdf_result_2, on=gdf_result.columns.tolist(), how='outer')

    RESULTS_DIR_PATH = WORKING_DIR_PATH.joinpath("results", size, model_name, "std")
    if not RESULTS_DIR_PATH.exists():
        RESULTS_DIR_PATH.mkdir(parents=True)
    gdf_result.to_file(RESULTS_DIR_PATH.joinpath(f"{res_name}.geojson"), driver='GeoJSON')

if __name__ == "__main__":
    global PROCESSED_DATA_DIR_PATH
    PROCESSED_DATA_DIR_PATH = WORKING_DIR_PATH.joinpath("dataset", "processed_data")

    parser = argparse.ArgumentParser(description="Solve the linear model for green cells placement optimization (specify again the size).")
    parser.add_argument("--size",
                        type=str,
                        default="full",
                        choices=["center", "full"],
                        help="Whether to run the model on the city center or on the entire cityscape.")
    parser.add_argument("--model",
                        type=str,
                        default="std",
                        choices=["std", "diff", "inverse_uhei", "ndvi"],
                        help="Choice of the model to run.")
    parser.add_argument("--noise_type", dest="noise_type", type=str, default="",
                        choices=["gauss", "multiplicative", "aggregation", "mislabelling", "missingness", "geometric"],
                        help="Type of noise to apply.")
    parser.add_argument("--noise_source", dest="noise_source", type=str, default="", 
                        choices=["uhei", "ndvi", "land_use", "trees", "full_space", "num_areas", "green_area", "ext_space", "street_space", "population_density"],
                        help="Where to add noise.")
    parser.add_argument("--noise_level", dest="noise_level", type=str, default="", choices=["1", "2", "3", "4", "5"],
                        help="Noise level.")
    parser.add_argument("--aggregation_method", dest="aggregation_method", type=str, default="", choices=["mean", "median"],
                        help="Aggregation method to use in case the noise to apply is aggregation")
    args = parser.parse_args()

    if args.noise_type != "":
        if args.noise_type == "aggregation":
            processed_filename = "processed_data_aggregation_" + args.noise_source + "_" + args.aggregation_method
            PROCESSED_DATA_DIR_PATH = WORKING_DIR_PATH.joinpath("dataset", processed_filename)
            res_name = "aggregation_" + args.noise_source + "_" + args.aggregation_method
        elif args.noise_type in ["gauss", "multiplicative"] and args.noise_source == "full_space":
            res_name = args.noise_type + "_" + args.noise_source + "_" + "lvl" + args.noise_level
        else:
            processed_filename = "processed_data_" + args.noise_type + "_" + args.noise_source + "_" + "lvl" + args.noise_level
            PROCESSED_DATA_DIR_PATH = WORKING_DIR_PATH.joinpath("dataset", processed_filename)
            res_name = args.noise_type + "_" + args.noise_source + "_" + "lvl" + args.noise_level
    else:
        res_name = "no_noise"
    solve(args.size, args.model, res_name)
