import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import json
import math
from shapely.geometry import Point

def xgb_spatial_predict(points_gdf, aoi_gdf, value_col="eggs", res_m=100, k_folds=5):
    """
    XGBoost espacial -
    """
    
    XGB_PARAMS = {
        'n_estimators': 300,
        'max_depth': 10,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1
    }
    
    # Proyección a UTM 16N
    target_crs = "EPSG:32616"
    aoi_proj = aoi_gdf.to_crs(target_crs)
    pts_proj = points_gdf.to_crs(target_crs)
    
    # Filtrar puntos dentro del área de interés
    pts_in_aoi = gpd.sjoin(pts_proj, aoi_proj, how='inner', predicate='within')
    
    if len(pts_in_aoi) == 0:
        raise ValueError("No hay puntos dentro del área de interés")
    
    print(f" Puntos dentro del AOI: {len(pts_in_aoi)}")
    
    
    pts_in_aoi = pts_in_aoi.assign(
        X=pts_in_aoi.geometry.x,
        Y=pts_in_aoi.geometry.y
    )
    
    # Preparación de datos
    base_feats = ['X', 'Y']
    temporal_feats = [f for f in ['year', 'week', 'month', 'season'] if f in pts_in_aoi.columns]
    all_feats = base_feats + temporal_feats
    
    valid_data = pts_in_aoi.dropna(subset=all_feats + [value_col])
    X = valid_data[all_feats].values
    y = valid_data[value_col].values
    
    if len(valid_data) < k_folds:
        raise ValueError(f"Insuficientes datos: {len(valid_data)} puntos para {k_folds} folds")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = XGBRegressor(**XGB_PARAMS)
    y_pred = cross_val_predict(model, X_scaled, y, 
                              cv=KFold(n_splits=k_folds, shuffle=True, random_state=42))
    
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y, y_pred)),
        'mae': mean_absolute_error(y, y_pred),
        'r2': r2_score(y, y_pred),
        'n_samples': len(y)
    }
    
    # Entrenamiento final
    final_model = XGBRegressor(**XGB_PARAMS).fit(X_scaled, y)
    
    
    bounds = aoi_proj.total_bounds
    print(f"Bounds del área: {bounds}")
    
   
    x_res = int((bounds[2] - bounds[0]) / res_m)
    y_res = int((bounds[3] - bounds[1]) / res_m)
    
    gx = np.linspace(bounds[0], bounds[2], x_res)
    gy = np.linspace(bounds[1], bounds[3], y_res)
    
    GX, GY = np.meshgrid(gx, gy)
    
    # Crear todos los puntos del grid
    grid_points = [Point(x, y) for x, y in zip(GX.ravel(), GY.ravel())]
    grid_gdf = gpd.GeoDataFrame(geometry=grid_points, crs=target_crs)
    
    # Filtrar solo puntos dentro del shapefile
    points_in_aoi = gpd.sjoin(grid_gdf, aoi_proj, how='inner', predicate='within')
    
    print(f" Puntos en grid dentro del AOI: {len(points_in_aoi)}")
    
    # Crear DataFrame para predicción
    grid_df = pd.DataFrame({
        'X': points_in_aoi.geometry.x,
        'Y': points_in_aoi.geometry.y
    })
    
    # Predicción en grid
    if len(grid_df) > 0:
        grid_scaled = scaler.transform(grid_df[all_feats].values)
        Z_values = final_model.predict(grid_scaled)
        
        # Creación de matriz completa
        Z_full = np.full(GX.shape, np.nan)
        
        # Mapear predicciones a la matriz
        x_idx = np.digitize(grid_df['X'], gx) - 1
        y_idx = np.digitize(grid_df['Y'], gy) - 1
        
        valid_mask = (x_idx >= 0) & (x_idx < GX.shape[1]) & (y_idx >= 0) & (y_idx < GX.shape[0])
        x_idx = x_idx[valid_mask]
        y_idx = y_idx[valid_mask]
        Z_values = Z_values[valid_mask]
        
        Z_full[y_idx, x_idx] = Z_values
        
        print(f" Rango de predicción: {np.nanmin(Z_full):.1f} - {np.nanmax(Z_full):.1f}")
        
    else:
        Z_full = np.full(GX.shape, np.nan)
        print(" No hay puntos en el grid para predecir")
    
    # Crear y guardar mapa
    create_prediction_map(Z_full, gx, gy, bounds, metrics, aoi_proj, "prediccion_xgb_merida_.png")
    
    # Guardar resultados
    save_results(metrics, final_model, all_feats, valid_data, y_pred)
    
    return metrics, Z_full

def create_prediction_map(Z, gx, gy, bounds, metrics, aoi_gdf, filename="prediccion_xgb_merida.png"):
    """Generar mapa de predicción XGB"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Configuración de colores
    cmap = 'YlOrRd'  
    vmin = 0
    vmax = 300  # Forzar el rango a 0-300
    
    # Mostrar shapefile como base 
    aoi_gdf.plot(ax=ax, color='none', edgecolor='black', alpha=0.8, linewidth=1.5)
    
    # Mostrar predicción con extent correcto
    im = ax.imshow(Z, extent=[gx[0], gx[-1], gy[0], gy[-1]], 
                   cmap=cmap, origin='lower',  # IMPORTANTE: origin='lower' para coincidir con coordenadas
                   vmin=vmin, vmax=vmax, alpha=0.9)
    
    # Configuración del mapa 
    ax.set_title('Predicción Espacial de Ovitrampas Mérida - XGBoost', 
                fontsize=14, fontweight='bold', pad=15)
    
    # Etiquetas de ejes
    ax.set_xlabel('Coordenada Este (UTM 16N)', fontsize=10)
    ax.set_ylabel('Coordenada Norte (UTM 16N)', fontsize=10)
    
    # Barra de color
    cbar = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label('Densidad de Huevos Predicha', fontsize=10, fontweight='bold')
    
    # ESCALA DE COLORES 
    cbar.set_ticks([0, 50, 100, 150, 200, 250, 300])
    
    # Métricas en posición 
    metric_text = f"R² = {metrics['r2']:.3f}\nRMSE = {metrics['rmse']:.2f}\nMAE = {metrics['mae']:.2f}"
    ax.text(0.02, 0.98, metric_text, transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, 
                     edgecolor='black', linewidth=1),
            verticalalignment='top', fontsize=9, fontweight='bold')
    
    # Resolución
    info_text = f"Resolución: 100m\nMuestra: {metrics['n_samples']} puntos"
    ax.text(0.98, 0.02, info_text, transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
            verticalalignment='bottom', horizontalalignment='right', fontsize=8)
    
    # BARRA DE ESCALA --opcional, sino se puede quitar sin afectar al resto del codigo
    add_manual_scalebar(ax, bounds, length_km=15)
    
   
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    
    # Ajustar límites
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])
    
    # Añadir coordenadas en los bordes
    ax.tick_params(axis='both', which='major', labelsize=8)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Mapa guardado: {filename}")
    print(f"Estadísticas de predicción:")
    print(f"   - Mínimo: {np.nanmin(Z):.1f}")
    print(f"   - Máximo: {np.nanmax(Z):.1f}")
    print(f"   - Media: {np.nanmean(Z):.1f}")
    print(f"   - Percentil 95: {np.nanpercentile(Z, 95):.1f}")

def add_manual_scalebar(ax, bounds, location='lower left', length_km=15):
    """
    Añadir barra de escala de 15 --opcional, se puede quitar sin afectar el resto del código
    """
    x_range = bounds[2] - bounds[0]
    y_range = bounds[3] - bounds[1]
    
    # Posición en la esquina inferior izquierda
    x_pos = bounds[0] + 0.08 * x_range
    y_pos = bounds[1] + 0.08 * y_range
    
    # Longitud en metros (15 km)
    length_m = length_km * 1000
    
    # Dibujar barra de escala principal
    ax.plot([x_pos, x_pos + length_m], [y_pos, y_pos], 
            color='black', linewidth=4, solid_capstyle='butt')
    
    # Dibujar marcas divisorias
    ax.plot([x_pos, x_pos], [y_pos-1000, y_pos+1000], 
            color='black', linewidth=2)
    ax.plot([x_pos + length_m/2, x_pos + length_m/2], [y_pos-1000, y_pos+1000], 
            color='black', linewidth=2)
    ax.plot([x_pos + length_m, x_pos + length_m], [y_pos-1000, y_pos+1000], 
            color='black', linewidth=2)
    
    # Añadir texto
    ax.text(x_pos + length_m/2, y_pos + 0.02 * y_range, 
            f'{length_km} km', 
            ha='center', va='bottom', fontsize=9, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9))

def save_results(metrics, model, features, data, y_pred):
    """Guardar resultados del modelo"""
    # Métricas
    with open('xgb_metrics_mejorado.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Importancia de características
    importance_df = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    importance_df.to_csv('xgb_feature_importance_mejorado.csv', index=False)
    
    print(" Resultados guardados:")
    print(f"   - xgb_metrics_mejorado.json")
    print(f"   - xgb_feature_importance_merida.csv")

def load_data(csv_path, shp_path):
    """Cargar y preparar datos"""
    # Cargar puntos de huevos
    df = pd.read_csv(csv_path)
    
    # Verificar columnas disponibles
    print("Columnas disponibles:", df.columns.tolist())
    
    # Buscar columnas de coordenadas
    x_col = next((col for col in df.columns if col.lower() in ['x', 'long', 'longitude', 'este']), 'x')
    y_col = next((col for col in df.columns if col.lower() in ['y', 'lat', 'latitude', 'norte']), 'y')
    
    points_gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[x_col], df[y_col]),
        crs="EPSG:4326"
    )
    
    # Cargar shapefile de Mérida
    aoi_gdf = gpd.read_file(shp_path)
    
    print(f"Datos cargados:")
    print(f"   - Puntos: {len(points_gdf)}")
    print(f"   - Shapefile: {len(aoi_gdf)} polígonos")
    print(f"   - CRS shapefile: {aoi_gdf.crs}")
    
    return points_gdf, aoi_gdf

def run_xgb_analysis(csv_path="eggs_data.csv", shp_path="Merida/2024_31050_L22102025_2228.shp"):
    """
    Ejecutar análisis XGBoost
    """
    print("INICIANDO ANÁLISIS XGBOOST PARA MÉRIDA")
    print("=" * 60)
    
    # Cargar datos
    points, aoi = load_data(csv_path, shp_path)
    
    # Ejecuta la predicción
    try:
        results, prediction_grid = xgb_spatial_predict(
            points, 
            aoi, 
            value_col="eggs", 
            res_m=100,
            k_folds=5
        )
        
        print(f"\n RESULTADOS FINALES:")
        print(f"    R²: {results['r2']:.3f}")
        print(f"   RMSE: {results['rmse']:.2f}")
        print(f"   MAE: {results['mae']:.2f}")
        print(f"   Muestra: {results['n_samples']} puntos")
        print(f"   Rango predicción: {np.nanmin(prediction_grid):.1f} - {np.nanmax(prediction_grid):.1f}")
        print(f"   Mapa: prediccion_xgb_merida.png")
        
    except Exception as e:
        print(f" Error: {e}")
        import traceback
        traceback.print_exc()

# Ejecución y llamado del main
if __name__ == "__main__":
    run_xgb_analysis()