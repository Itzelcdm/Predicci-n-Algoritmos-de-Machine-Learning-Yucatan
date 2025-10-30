import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import os

def crear_mapa_idw():
    """Función simple para crear mapa IDW en PNG"""
    
    print("Iniciando interpolación IDW...")
    
    # Configuración
    csv_path = "eggs_data.csv"
    shp_path = "Merida/2024_31050_L22102025_2228.shp"
    output_path = "mapa_interpolacion_idw.png"
    
    # Verificar archivos
    if not os.path.exists(csv_path):
        print(f" No se encuentra: {csv_path}")
        return False
        
    if not os.path.exists(shp_path):
        print(f" No se encuentra: {shp_path}")
        return False
    
    try:
        # 1. CARGAR DATOS
        print("Cargando datos...")
        df = pd.read_csv(csv_path)
        merida = gpd.read_file(shp_path)
        
        print(f" Datos CSV: {len(df)} filas, columnas: {list(df.columns)}")
        
        # 2. PREPARAR DATOS ESPACIALES
        print(" Preparando datos espaciales...")
        gdf = gpd.GeoDataFrame(
            df, 
            geometry=gpd.points_from_xy(df.x, df.y),
            crs="EPSG:4326"
        )
        
        # Convertir a UTM
        gdf = gdf.to_crs("EPSG:32616")
        merida = merida.to_crs("EPSG:32616")
        
        # Filtrar puntos dentro de Mérida
        gdf = gdf[gdf.within(merida.unary_union)]
        
        if len(gdf) == 0:
            print(" No hay puntos dentro de Mérida")
            return False
            
        print(f" {len(gdf)} puntos dentro de Mérida")
        
        # 3. EXTRAER COORDENADAS
        x_known = gdf.geometry.x.values
        y_known = gdf.geometry.y.values
        z_known = gdf['eggs'].values
        
        print(f" X: {x_known.min():.0f}-{x_known.max():.0f}")
        print(f" Y: {y_known.min():.0f}-{y_known.max():.0f}")
        print(f" Huevos: {z_known.min()}-{z_known.max()}")
        
        # 4. CREAR GRID
        print(" Creando grid...")
        bounds = merida.total_bounds
        x_min, y_min, x_max, y_max = bounds
        
        resolution = 80  # Más rápido
        x_grid = np.linspace(x_min, x_max, resolution)
        y_grid = np.linspace(y_min, y_max, resolution)
        X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
        
        # 5. INTERPOLACIÓN IDW SIMPLE
        print(" Interpolando...")
        Z_grid = np.zeros(X_grid.shape)
        
        for i in range(X_grid.shape[0]):
            for j in range(X_grid.shape[1]):
                x_target = X_grid[i, j]
                y_target = Y_grid[i, j]
                
                # Calcular distancias
                distances = np.sqrt((x_known - x_target)**2 + (y_known - y_target)**2)
                
                # Usar solo puntos cercanos (10km)
                mask = distances <= 10000
                if np.any(mask):
                    valid_dist = distances[mask]
                    valid_vals = z_known[mask]
                    
                    # Peso inverso a la distancia
                    weights = 1.0 / (valid_dist**2 + 1.0)
                    Z_grid[i, j] = np.sum(weights * valid_vals) / np.sum(weights)
                else:
                    Z_grid[i, j] = np.nan
        
        # 6. CREAR MAPA
        print(" Generando mapa PNG...")
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Fondo
        merida.plot(ax=ax, color='lightgray', alpha=0.3, edgecolor='black')
        
        # Interpolación
        contour = ax.contourf(X_grid, Y_grid, Z_grid, levels=50, cmap='YlOrRd', alpha=0.8)
        
        # Puntos originales
        ax.scatter(x_known, y_known, c='blue', s=20, alpha=0.7, 
                  label='Puntos Muestreados')
        
        # Colorbar
        plt.colorbar(contour, ax=ax, label='Huevos Predichos')
        
        # Configuración
        ax.set_title('Interpolación IDW - Predicción de Huevos', fontsize=14, fontweight='bold')
        ax.set_xlabel('Coordenada X (UTM 16N)')
        ax.set_ylabel('Coordenada Y (UTM 16N)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Guardar
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f" MAPA GENERADO: {output_path}")
        
        # Estadísticas
        valid_values = Z_grid[~np.isnan(Z_grid)]
        if len(valid_values) > 0:
            print(f" Estadísticas:")
            print(f"   - Puntos interpolados: {len(valid_values)}")
            print(f"   - Rango: {valid_values.min():.1f} - {valid_values.max():.1f} huevos")
            print(f"   - Promedio: {valid_values.mean():.1f} huevos")
        
        return True
        
    except Exception as e:
        print(f" Error: {e}")
        import traceback
        traceback.print_exc()
        return False

# EJECUTAR
if __name__ == "__main__":
    print("=" * 50)
    print("GENERADOR DE MAPA IDW")
    print("=" * 50)
    
    success = crear_mapa_idw()
    
    if success:
        print("\n ¡Mapa IDW generado exitosamente!")
    else:
        print("\n No se pudo generar el mapa")