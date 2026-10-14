import pandas as pd
import geopandas as gpd
from pathlib import Path
import argparse


WORKING_DIR_PATH = Path.cwd()

def density_estimation():
    """
    Enlarges the dataset of the population per statistical area with the relative density.

    """

    gdf_area = gpd.read_file(PROCESSED_DATA_DIR_PATH.joinpath("aree_statistiche_stat.geojson"))
    df_pop = pd.read_csv(RAW_DATA_DIR_PATH.joinpath("popolazione_per_area_statistica.csv"))
    df_pop = df_pop[df_pop['Codice Area Statistica'] != 99]
    densities = []
    for _, row in df_pop.iterrows():
        code = row['Codice Area Statistica']
        temp_gdf = gdf_area[gdf_area['codice_area_statistica'] == code]
        area_km2 = temp_gdf['area'].values[0] / 1000000
        densities.append(row['Residenti'] / area_km2)
    df_pop['Densità'] = densities
    df_pop.to_csv(PROCESSED_DATA_DIR_PATH.joinpath("densità_per_area_statistica.csv"), sep=';', index=False)

def macro_factors_computation(density_param=0.33, green_param=0.33, uhei_param=0.34):
    """
    Estimates the factor to be applied to the macro utility function for statitistical areas.

    Input:
    - density_param: float, weight for the population density in the macro utility function (default 0.33)
    - green_param: float, weight for the green space in the macro utility function (default 0.33)
    - uhei_param: float, weight for the UHEI in the macro utility function (default 0.34)

    """

    if density_param + green_param + uhei_param != 1.0:
        raise ValueError("The sum of density_param, green_param and uhei_param must be equal to 1.0.")
    else:
        gdf_area = gpd.read_file(PROCESSED_DATA_DIR_PATH.joinpath("aree_statistiche_stat.geojson"))
        df_dens = pd.read_csv(PROCESSED_DATA_DIR_PATH.joinpath("densità_per_area_statistica.csv"), sep=';')
        max_density = df_dens['Densità'].max()
        max_uhei = gdf_area['uhei'].max()
        factors = []
        for row in gdf_area.itertuples():
            green_factor = 100.0 - row.verde_pc
            green_factor = green_factor / 100.0
            density = df_dens[df_dens['Codice Area Statistica'] == row.codice_area_statistica]['Densità'].values[0]
            density_factor = density / max_density
            uhei_factor = row.uhei / max_uhei
            factors.append((density_param * density_factor) + (green_param * green_factor) + (uhei_param * uhei_factor))
        gdf_area['macro_utility_factor'] = factors
        gdf_area.to_file(PROCESSED_DATA_DIR_PATH.joinpath("aree_statistiche_macro.geojson"), driver='GeoJSON')

if __name__ == "__main__":
    global RAW_DATA_DIR_PATH, PROCESSED_DATA_DIR_PATH
    RAW_DATA_DIR_PATH = WORKING_DIR_PATH.joinpath("dataset", "raw_data")
    PROCESSED_DATA_DIR_PATH = WORKING_DIR_PATH.joinpath("dataset", "processed_data")

    parser = argparse.ArgumentParser(description="Preprocess the data in order to adapt them to the model.")
    parser.add_argument("--density_param", 
                        type=float, 
                        default=0.33, 
                        help="Weight for the population density in the macro utility function.")
    parser.add_argument("--green_param", 
                        type=float, 
                        default=0.33, 
                        help="Weight for the green space in the macro utility function.")
    parser.add_argument("--uhei_param", 
                        type=float, 
                        default=0.34, 
                        help="Weight for the UHEI in the macro utility function.")
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
            
        elif args.noise_type in ["gauss", "multiplicative"] and args.noise_source == "full_space":
            pass
            
        elif (
            args.noise_type == "missingness"
            and args.noise_source in ["trees", "num_areas"]
        ) or (
            args.noise_type == "gauss"
            and args.noise_source == "num_areas"
        ) or (
            args.noise_source in [
                "green_area",
                "ext_space",
                "street_space",
            ]
        ) or (
            args.noise_type == "geometric"
        ):
            processed_filename = (
                f"processed_data_{args.noise_type}_"
                f"{args.noise_source}_lvl{args.noise_level}"
            )

            PROCESSED_DATA_DIR_PATH = (
                WORKING_DIR_PATH
                .joinpath("dataset", processed_filename)
            )
            
        else:
            raw_filename = "raw_data_" + args.noise_type + "_" + args.noise_source + "_" + "lvl" + args.noise_level
            processed_filename = "processed_data_" + args.noise_type + "_" + args.noise_source + "_" + "lvl" + args.noise_level
            RAW_DATA_DIR_PATH = WORKING_DIR_PATH.joinpath("dataset", raw_filename)
            PROCESSED_DATA_DIR_PATH = WORKING_DIR_PATH.joinpath("dataset", processed_filename)
    
    density_estimation()
    macro_factors_computation(args.density_param, args.green_param, args.uhei_param)
