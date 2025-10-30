import os
import math
import json
import warnings
import gc
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

# Configuración inicial
warnings.filterwarnings('ignore')

def idw_interpolation(x_known, y_known, z_known, x_target, y_target, power=2, radius=5000):
    """
    Interpolación IDW manual
    """
    n_target = len(x_target)
    z_predicted = np.full(n_target, np.nan, dtype=np.float64)
    
    for i in range(n_target):
        x_t, y_t = x_target[i], y_target[i]
        
        # Calcular distancias a todos los puntos conocidos
        distances = np.sqrt((x_known - x_t)**2 + (y_known - y_t)**2)
        
        # Filtrar puntos dentro del radio
        valid_mask = distances <= radius
        valid_distances = distances[valid_mask]
        valid_values = z_known[valid_mask]
        
        if len(valid_values) >= 2:  # Mínimo 2 puntos para interpolación
            valid_distances = np.maximum(valid_distances, 1.0)
            
            # Calcular pesos IDW
            weights = 1.0 / (valid_distances ** power)
            
            # Interpolación
            z_predicted[i] = np.sum(weights * valid_values) / np.sum(weights)
    
    return z_predicted

def crear_mapa_semaforizacion_merida():
    """Crea el mapa IDW para Mérida"""
    
    print("INICIANDO PREDICCIÓN ESPACIAL PARA MÉRIDA...")
    
    try:
        # 1. CARGAR Y PREPARAR DATOS
        print(" Cargando datos...")
        df = pd.read_csv("eggs_data.csv")
        merida = gpd.read_file("Merida/2024_31050_L22102025_2228.shp")
        
        # Convertir a UTM 16N
        gdf = gpd.GeoDataFrame(
            df, 
            geometry=gpd.points_from_xy(df.x, df.y),
            crs="EPSG:4326"
        ).to_crs("EPSG:32616")
        merida = merida.to_crs("EPSG:32616")
        
        # Filtrar puntos dentro de Mérida
        gdf = gdf[gdf.within(merida.unary_union)]
        print(f" {len(gdf)} puntos para calibración IDW")
        
        # 1. EXTRAER DATOS PARA INTERPOLACIÓN
        x_known = gdf.geometry.x.values
        y_known = gdf.geometry.y.values
        z_known = gdf['eggs'].values
        
        print(f"Rango de huevos en datos: {z_known.min()} - {z_known.max()}")
        
        # 2. CREAR GRID SOBRE EL SHAPEFILE
        bounds = merida.total_bounds
        x_min, y_min, x_max, y_max = bounds
        
        # Resolución optimizada
        resolution = 250  # metros
        x_coords = np.arange(x_min, x_max, resolution)
        y_coords = np.arange(y_min, y_max, resolution)
        X, Y = np.meshgrid(x_coords, y_coords)
        
        print(f" Grid de interpolación: {X.shape[1]}x{X.shape[0]} celdas")
        
        # 4. INTERPOLACIÓN IDW EN EL TERRITORIO
        print(" Realizando interpolación IDW...")
        
        # Crear máscara del shapefile
        merida_union = merida.unary_union
        points_inside = []
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                point = gpd.points_from_xy([X[i,j]], [Y[i,j]])[0]
                if merida_union.contains(point):
                    points_inside.append((X[i,j], Y[i,j]))
        
        points_inside = np.array(points_inside)
        print(f" {len(points_inside)} puntos dentro del área de Mérida")
        
        # Interpolación IDW
        if len(points_inside) > 0:
            z_predicted = idw_interpolation(x_known, y_known, z_known, 
                                          points_inside[:, 0], points_inside[:, 1], 
                                          power=2, radius=4000)
        else:
            raise ValueError("No hay puntos dentro del shapefile para interpolar")
        
        # Reconstruir matriz Z
        Z = np.full(X.shape, np.nan, dtype=np.float64)
        point_index = 0
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                point = gpd.points_from_xy([X[i,j]], [Y[i,j]])[0]
                if merida_union.contains(point):
                    Z[i,j] = z_predicted[point_index]
                    point_index += 1
        
        # DEFINIR CATEGORÍAS
        # Estadísticas para categorías
        valid_values = Z[~np.isnan(Z)]
        
        if len(valid_values) == 0:
            raise ValueError("No se generaron valores de interpolación válidos")
        
        # Calcular percentiles para categorías balanceadas
        p25 = np.percentile(valid_values, 25)
        p50 = np.percentile(valid_values, 50)
        p75 = np.percentile(valid_values, 75)
        p90 = np.percentile(valid_values, 90)
        
        categorias = {
            'MUY BAJO': (0, p25),
            'BAJO': (p25, p50),
            'MEDIO': (p50, p75),
            'ALTO': (p75, p90),
            'MUY ALTO': (p90, np.max(valid_values) * 1.1)
        }
        
        print(f" CATEGORÍAS DE RIESGO (percentiles):")
        for cat, (min_val, max_val) in categorias.items():
            print(f"   - {cat}: {min_val:.1f} - {max_val:.1f} huevos")
        
        # CREAR MAPA 
        print("Generando mapa de predicción...")
        
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # FONDO BLANCO CON CUADRÍCULA
        ax.set_facecolor('white')  # Fondo blanco
        ax.grid(True, color='lightgray', linestyle='-', linewidth=0.5, alpha=0.7)  # Cuadrícula visible
        
        # Colores semáforo 
        colors_semaforo = ['#00cc00', '#99ff33', '#ffff00', '#ff9900', '#ff0000']
        cmap_semaforo = ListedColormap(colors_semaforo)
        
        # Crear matriz de categorías
        Z_categorias = np.full(Z.shape, -1, dtype=int)
        
        for i, (cat, (min_val, max_val)) in enumerate(categorias.items()):
            mask = (Z >= min_val) & (Z < max_val)
            Z_categorias[mask] = i
        
        # Rellenar áreas sin datos
        Z_categorias[np.isnan(Z)] = -1
        
        # Visualización con imshow para mejor calidad 
        im = ax.imshow(Z_categorias, extent=[x_min, x_max, y_min, y_max], 
                      cmap=cmap_semaforo, alpha=0.85, origin='lower',
                      aspect='auto')
        
        # Borde del municipio
        merida.boundary.plot(ax=ax, color='black', linewidth=2, alpha=0.9)
        
        # Títulos
        ax.set_title('Prediccion espacial de las ovitrampas de Aedes Aegypti en Merida mediante IDW', 
                    fontsize=18, fontweight='bold', pad=20)
        
        ax.set_xlabel('Coordenada Este (UTM 16N)', fontsize=12)
        ax.set_ylabel('Coordenada Norte (UTM 16N)', fontsize=12)
        
        # Leyenda
        legend_elements = [
            mpatches.Patch(color='#00cc00', alpha=0.85, label='MUY BAJO'),
            mpatches.Patch(color='#99ff33', alpha=0.85, label='BAJO'),
            mpatches.Patch(color='#ffff00', alpha=0.85, label='MEDIO'),
            mpatches.Patch(color='#ff9900', alpha=0.85, label='ALTO'),
            mpatches.Patch(color='#ff0000', alpha=0.85, label='MUY ALTO')
        ]
        
        ax.legend(handles=legend_elements, loc='upper right', 
                 framealpha=0.95, fontsize=11, title='NIVELES DE RIESGO',
                 title_fontsize=12)
        
        # Ajustar límites
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        
        plt.tight_layout()
        plt.savefig('prediccion_ovitrampas_merida_idw.png', dpi=300, 
                   bbox_inches='tight', facecolor='white')
        plt.close()
        
        #GUARDAR DATOS GEOESPACIALES
        print(" Guardando resultados geoespaciales...")
        
        # Guardar datos de predicción
        resultados_df = pd.DataFrame({
            'x': X.flatten(),
            'y': Y.flatten(),
            'huevos_predichos': Z.flatten(),
            'categoria': pd.cut(Z.flatten(), 
                              bins=[0, p25, p50, p75, p90, np.max(valid_values) * 1.1],
                              labels=['MUY BAJO', 'BAJO', 'MEDIO', 'ALTO', 'MUY ALTO'],
                              include_lowest=True)
        })
        
        resultados_df = resultados_df.dropna()
        resultados_df.to_csv('prediccion_idw_merida.csv', index=False)
        
        # 8. ESTADÍSTICAS FINALES
        areas_por_categoria = {}
        total_celdas = np.sum(Z_categorias >= 0)
        
        for i, categoria in enumerate(['MUY BAJO', 'BAJO', 'MEDIO', 'ALTO', 'MUY ALTO']):
            count = np.sum(Z_categorias == i)
            porcentaje = (count / total_celdas) * 100
            areas_por_categoria[categoria] = {
                'celdas': count,
                'porcentaje': porcentaje,
                'area_km2': (count * (resolution ** 2)) / 1000000
            }
        
        print(f"\nRESUMEN DE PREDICCIÓN:")
        print(f"   - Área total analizada: {total_celdas:,} celdas")
        print(f"   - Superficie: {total_celdas * (resolution ** 2) / 1000000:.1f} km²")
        print(f"   - Rango predicho: {np.min(valid_values):.1f} - {np.max(valid_values):.1f} huevos")
        
        print(f"\n DISTRIBUCIÓN TERRITORIAL:")
        for categoria, stats in areas_por_categoria.items():
            print(f"   - {categoria}: {stats['celdas']:,} celdas ({stats['porcentaje']:.1f}%)")
        
        # GUARDAR METADATOS
        metadatos = {
            'fecha_procesamiento': pd.Timestamp.now().isoformat(),
            'resolucion_metros': resolution,
            'radio_interpolacion_metros': 4000,
            'potencia_idw': 2,
            'total_puntos_interpolados': len(points_inside),
            'categorias_riesgo': categorias,
            'estadisticas_areas': areas_por_categoria,
            'crs': 'EPSG:32616 (UTM 16N)'
        }
        
        with open('metadatos_prediccion.json', 'w') as f:
            json.dump(metadatos, f, indent=2)
        
        print(f"\n PREDICCIÓN COMPLETADA EXITOSAMENTE")
        print(f" Archivos generados:")
        print(f"   - prediccion_ovitrampas_merida_idw.png (Mapa principal)")
        print(f"   - prediccion_idw_merida.csv (Datos de predicción)")
        print(f"   - metadatos_prediccion.json (Metadatos técnicos)")
        
        # Limpiar memoria (Más optimo)
        gc.collect()
        
        return True
        
    except Exception as e:
        print(f"Error en la predicción: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 80)
    print("PREDICCIÓN ESPACIAL DE OVITRAMPAS - MÉRIDA YUCATÁN")
    print("=" * 80)
    
    # Ejecutar predicción principal
    success = crear_mapa_semaforizacion_merida()
    
    if success:
        print("\n" + "=" * 80)
        print(" ANÁLISIS COMPLETADO - PREDICCIÓN IDW REALIZADA")
        print("=" * 80)
        print(" El mapa muestra la predicción de densidad de huevos en áreas no muestreadas")
        print("   utilizando la técnica de interpolación IDW (Inverse Distance Weighting)")
        print("   - Fondo blanco con cuadrícula visible")
        print("   - Colores sólidos para las categorías de riesgo")
        print("   - Título específico solicitado")
    else:
        print("\n No se pudo completar la predicción")