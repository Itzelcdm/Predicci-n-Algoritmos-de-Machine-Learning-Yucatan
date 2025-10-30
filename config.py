
from dataclasses import dataclass

@dataclass
class Config:
    csv_path: str = "eggs_data.csv"
    merida_shp: str = "Merida/2024_31050_L22102025_2228.shp"
    target_crs: str = "EPSG:32616"  # UTM 16N para Yucatán