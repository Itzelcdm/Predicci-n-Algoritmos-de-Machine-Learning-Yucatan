from config import Config
from dengue import DengueVisualizer

def main():
    print("INICIANDO VISUALIZACIÓN ESPACIAL DE DATOS DE DENGUE")
    print("=" * 50)
    
    config = Config(
        csv_path="eggs_data.csv",
        merida_shp="Merida/2024_31050_L22102025_2228.shp"
    )
    
   
    visualizer = DengueVisualizer(config)
    visualizer.create_basic_maps()
    
    print("=" * 50)
    print("ANÁLISIS COMPLETADO!")
    print("Mapas generados:")
    print("   - mapa_puntos_simple.png")
    print("   - mapa_calor.png") 
    print("   - mapa_categorias.png")

if __name__ == "__main__":
    main()