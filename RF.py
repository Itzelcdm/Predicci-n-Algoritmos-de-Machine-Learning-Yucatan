import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
from rasterio.transform import from_origin
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def advanced_rf_predict(points_gdf, poly_gdf, target_col="eggs", res_m=250, 
                       temporal_cols=None, n_estimators=200, cv_folds=5):
    """
    Random Forest 
    """
    # Configuración 
    temporal_cols = temporal_cols or []
    coords = np.column_stack([points_gdf.geometry.x, points_gdf.geometry.y])
    
    # Creación de variables que mejoran la capacidad predictiva del modelo
    df = points_gdf.assign(
        coord_x = points_gdf.geometry.x,
        coord_y = points_gdf.geometry.y,
        x2 = points_gdf.geometry.x ** 2,
        y2 = points_gdf.geometry.y ** 2,
        xy = points_gdf.geometry.x * points_gdf.geometry.y,
        dist_centroid = points_gdf.geometry.distance(points_gdf.unary_union.centroid),
        point_density = 1 / (KDTree(coords).query(coords, k=6)[0][:, 1:].mean(axis=1) + 1e-8)
    )
    
    # Features temporales dinámicas
    if 'week' in df.columns:
        df = df.assign(
            sin_week = np.sin(2 * np.pi * df.week / 52),
            cos_week = np.cos(2 * np.pi * df.week / 52)
        )
    
    # Selección automática de features
    base_feats = ['coord_x', 'coord_y', 'x2', 'y2', 'xy', 'dist_centroid', 'point_density']
    temp_feats = [f for f in ['sin_week', 'cos_week', 'year'] if f in df.columns]
    all_feats = base_feats + temp_feats + [c for c in temporal_cols if c in df.columns]
    available_feats = [f for f in all_feats if f in df.columns]
    
    X, y = df[available_feats].values, df[target_col].values
    
    # Modelo con validación espacial inteligente
    rf = RandomForestRegressor(
        n_estimators=min(n_estimators, len(X)),
        max_depth=None if len(X) > 100 else 10,
        random_state=42,
        n_jobs=-1
    )
    
    # Cross-validation espacial por clustering
    n_folds = min(cv_folds, len(X))
    if len(X) > n_folds * 3:
        clusters = KMeans(n_clusters=n_folds, random_state=42).fit_predict(coords)
        y_pred = np.full_like(y, np.nan)
        for cluster in range(n_folds):
            train_idx, test_idx = clusters != cluster, clusters == cluster
            if train_idx.sum() > 5 and test_idx.sum() > 1:
                rf.fit(X[train_idx], y[train_idx])
                y_pred[test_idx] = rf.predict(X[test_idx])
    else:
        y_pred = cross_val_predict(rf, X, y, cv=min(cv_folds, len(X)), n_jobs=-1)
    
    # Métricas solo con predicciones válidas
    valid_mask = ~np.isnan(y_pred)
    metrics = {
        'r2': r2_score(y[valid_mask], y_pred[valid_mask]) if valid_mask.sum() > 1 else 0,
        'rmse': np.sqrt(mean_squared_error(y[valid_mask], y_pred[valid_mask])) if valid_mask.sum() > 1 else np.nan,
        'n_valid': valid_mask.sum()
    } if valid_mask.sum() > 1 else {'error': 'Validación falló'}
    
    # Entrenar el modelo final
    rf_final = RandomForestRegressor(
        n_estimators=min(n_estimators, len(X)),
        random_state=42,
        n_jobs=-1
    ).fit(X, y)
    
    # Crear grid de predicción optimizado
    bounds = poly_gdf.total_bounds
    minx, miny, maxx, maxy = bounds
    
    nx, ny = int(np.ceil((maxx - minx) / res_m)), int(np.ceil((maxy - miny) / res_m))
    gx, gy = np.meshgrid(
        np.linspace(minx + res_m/2, maxx - res_m/2, nx),
        np.linspace(maxy - res_m/2, miny + res_m/2, ny)
    )
    
    # Predicción eficiente en grid
    grid_points = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(gx.ravel(), gy.ravel()),
        crs=poly_gdf.crs
    )
    
    points_inside = grid_points[grid_points.within(poly_gdf.unary_union)]
    
    if len(points_inside) > 0:
        # Preparar features de predicción
        pred_df = points_inside.assign(
            coord_x = points_inside.geometry.x,
            coord_y = points_inside.geometry.y,
            x2 = points_inside.geometry.x ** 2,
            y2 = points_inside.geometry.y ** 2,
            xy = points_inside.geometry.x * points_inside.geometry.y,
            dist_centroid = points_inside.geometry.distance(poly_gdf.unary_union.centroid),
            point_density = 0.1  # Valor por defecto para densidad
        )
        
        # Features faltantes
        for feat in available_feats:
            if feat not in pred_df.columns:
                pred_df[feat] = 0
        
        Z = np.full(gx.shape, np.nan, dtype=np.float32)
        Z.ravel()[points_inside.index] = rf_final.predict(pred_df[available_feats].values)
    else:
        Z = np.full(gx.shape, np.nan, dtype=np.float32)
    
    return {
        'grid': Z,
        'transform': from_origin(minx, maxy, res_m, res_m),
        'metrics': metrics,
        'model': rf_final,
        'features': available_feats,
        'importance': dict(zip(available_feats, rf_final.feature_importances_)),
        'bounds': bounds
    }

# CARGAR Y PREPARAR DATOS
def load_and_prepare_data(csv_path: str, shp_path: str, target_crs: str = "EPSG:32616"):
    """
    Cargar y preparar datos con configuración específica para Mérida, Yucatán
    """
    print("Cargando datos...")
    
    # 1. Cargar datos de huevos
    eggs_df = pd.read_csv(csv_path)
    eggs_gdf = gpd.GeoDataFrame(
        eggs_df,
        geometry=gpd.points_from_xy(eggs_df.x, eggs_df.y),
        crs="EPSG:4326"  # WGS84
    )
    
    # 2. Cargar shapefile de Mérida
    merida_gdf = gpd.read_file(shp_path)
    
    print(f" Datos cargados:")
    print(f"   - Puntos de huevos: {len(eggs_gdf)}")
    print(f"   - Polígonos de Mérida: {len(merida_gdf)}")
    print(f"   - CRS original Mérida: {merida_gdf.crs}")
    
    # 3. Proyectar a UTM 16N (Yucatán)
    print(f"Proyectando a {target_crs}...")
    
    eggs_proj = eggs_gdf.to_crs(target_crs)
    merida_proj = merida_gdf.to_crs(target_crs)
    
    print(f"Proyección completada")
    print(f"   - Bounds Mérida: {merida_proj.total_bounds}")
    print(f"   - Bounds puntos: {eggs_proj.total_bounds}")
    
    # 4. Filtrar puntos que están dentro de Mérida
    puntos_en_merida = gpd.sjoin(eggs_proj, merida_proj, how='inner', predicate='within')
    
    print(f" Puntos dentro de Mérida: {len(puntos_en_merida)}/{len(eggs_proj)}")
    
    return puntos_en_merida, merida_proj

# ANÁLISIS EXPLORATORIO 
def exploratory_analysis(gdf, target_col="eggs"):
    """Análisis exploratorio de los datos"""
    print("\n ANÁLISIS EXPLORATORIO:")
    print(f"   - Total de registros: {len(gdf)}")
    print(f"   - Rango temporal: {gdf['year'].min()}-{gdf['year'].max()}")
    print(f"   - Semanas cubiertas: {sorted(gdf['week'].unique())}")
    print(f"   - Valores de {target_col}:")
    print(f"     Mín: {gdf[target_col].min()}, Máx: {gdf[target_col].max()}")
    print(f"     Media: {gdf[target_col].mean():.1f}, Mediana: {gdf[target_col].median():.1f}")
    print(f"     Puntos con 0 huevos: {(gdf[target_col] == 0).sum()} ({(gdf[target_col] == 0).mean()*100:.1f}%)")
    
    # Análisis por semana
    if 'week' in gdf.columns:
        weekly_stats = gdf.groupby('week')[target_col].agg(['count', 'mean', 'sum'])
        print(f"   - Conteo por semana:")
        for week, stats in weekly_stats.iterrows():
            print(f"     Semana {week}: {stats['count']} puntos, {stats['mean']:.1f} huevos/punto")

# GENERAR MAPA PNG 
def generate_prediction_map(resultado, poly_gdf, output_filename="prediccion_ovitrampas_merida.png"):
    """Generar mapa de predicción en formato PNG sin puntos observados"""
    print("\n GENERANDO MAPA DE PREDICCIÓN...")
    
    # Configuración de estilo 
    plt.style.use('default')
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    Z = resultado['grid']
    bounds = resultado['bounds']
    
    # Crear colormap personalizado
    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad('lightgray', 1.0)
    
    # Mostrar predicción
    im = ax.imshow(Z, extent=[bounds[0], bounds[2], bounds[1], bounds[3]], 
                   cmap=cmap, origin='upper', vmin=0, vmax=np.nanmax(Z)*0.8)
    
    # Añadir contorno del municipio 
    poly_gdf.boundary.plot(ax=ax, color='black', linewidth=2.5, alpha=0.9)
    
    # Título y etiquetas
    ax.set_title('Predicción espacial de las ovitrampas Mérida Yucatán mediante Random Forest', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Coordenada Este (UTM 16N)', fontsize=12)
    ax.set_ylabel('Coordenada Norte (UTM 16N)', fontsize=12)
    
    # Barra de color 
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.03)
    cbar.set_label('Densidad de huevos predicha', fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)
    
    # Añadir métricas del modelo en el mapa
    metrics_text = f"R² = {resultado['metrics']['r2']:.3f}\nRMSE = {resultado['metrics']['rmse']:.2f}"
    ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, 
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9, edgecolor='black'),
            verticalalignment='top', fontsize=11, fontweight='bold')
    
    # Añadir información de resolución
    res_text = f"Resolución: 500m\nPíxeles válidos: {np.sum(~np.isnan(Z)):,}"
    ax.text(0.98, 0.02, res_text, transform=ax.transAxes, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            verticalalignment='bottom', horizontalalignment='right', fontsize=10)
    
    # Grid sutil
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    
    # Mejorar los ticks
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    # Guardar mapa
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor='white', 
                edgecolor='none', transparent=False)
    print(f"    Mapa guardado: {output_filename}")
    
    plt.close()

# EJECUCIÓN PRINCIPAL MEJORADA
def main():
    # CONFIGURACIÓN
    csv_path = "eggs_data.csv"
    merida_shp = "Merida/2024_31050_L22102025_2228.shp" 
    target_crs = "EPSG:32616"  # UTM 16N para Yucatán
    
    print("INICIANDO ANÁLISIS ESPACIAL DE OVITRAMPAS EN MÉRIDA")
    print("=" * 50)
    
    try:
        # 1. Cargar y preparar datos
        puntos_merida, merida_poly = load_and_prepare_data(csv_path, merida_shp, target_crs)
        
        # 2. Análisis exploratorio
        exploratory_analysis(puntos_merida)
        
        # 3. Verificar que tenemos datos suficientes
        if len(puntos_merida) < 10:
            print(" ERROR: Muy pocos puntos para análisis espacial")
            return
        
        # 4. Ejecutar Random Forest espacial
        print("\n EJECUTANDO RANDOM FOREST ESPACIAL...")
        
        resultado = advanced_rf_predict(
            puntos_merida, 
            merida_poly,
            target_col="eggs",
            res_m=500,  # 500m resolution para Mérida
            temporal_cols=["week", "year"],
            n_estimators=200,
            cv_folds=5
        )
        
        # 5. Mostrar resultados
        print("\n RESULTADOS DEL MODELO:")
        print(f"    R²: {resultado['metrics']['r2']:.3f}")
        print(f"   RMSE: {resultado['metrics']['rmse']:.2f} huevos")
        print(f"   Puntos válidos: {resultado['metrics']['n_valid']}")
        print(f"   Features utilizados: {len(resultado['features'])}")
        
        print(f"\n IMPORTANCIA DE VARIABLES:")
        for feat, imp in sorted(resultado['importance'].items(), key=lambda x: x[1], reverse=True):
            print(f"   - {feat}: {imp:.3f}")
        
        # 6. Generar mapa PNG 
        generate_prediction_map(resultado, merida_poly, "prediccion_ovitrampas_merida.png")
        
        # 7. Guardar resultados adicionales
        print("\n GUARDANDO RESULTADOS ADICIONALES...")
        
        # Guardar raster de predicción
        try:
            from rasterio import open as rasterio_open
            
            with rasterio_open('prediccion_huevos_merida.tif', 'w', 
                              driver='GTiff', 
                              height=resultado['grid'].shape[0], 
                              width=resultado['grid'].shape[1], 
                              count=1, 
                              dtype=str(resultado['grid'].dtype), 
                              crs=merida_poly.crs,
                              transform=resultado['transform'],
                              nodata=np.nan) as dst:
                dst.write(resultado['grid'], 1)
            
            print("   Predicción guardada como 'prediccion_huevos_merida.tif'")
            
        except ImportError:
            print("     Instala rasterio: pip install rasterio")
        
        # Guardar métricas del modelo
        metrics_df = pd.DataFrame([resultado['metrics']])
        metrics_df.to_csv('metricas_modelo.csv', index=False)
        print("    Métricas guardadas como 'metricas_modelo.csv'")
        
        # Guardar importancia de variables
        importance_df = pd.DataFrame(list(resultado['importance'].items()), 
                                   columns=['Variable', 'Importancia'])
        importance_df.to_csv('importancia_variables.csv', index=False)
        print("    Importancia de variables guardada como 'importancia_variables.csv'")
        
        print("\n ANÁLISIS COMPLETADO EXITOSAMENTE!")
        
        # 8. Resumen final
        grid_stats = resultado['grid']
        valid_pixels = np.sum(~np.isnan(grid_stats))
        if valid_pixels > 0:
            print(f"\n RESUMEN PREDICCIÓN:")
            print(f"   - Píxeles con predicción: {valid_pixels:,}")
            print(f"   - Huevos mínimos predichos: {np.nanmin(grid_stats):.1f}")
            print(f"   - Huevos máximos predichos: {np.nanmax(grid_stats):.1f}")
            print(f"   - Huevos promedio predichos: {np.nanmean(grid_stats):.1f}")
            print(f"   - Resolución: 500m")
            print(f"   - Área cubierta: {valid_pixels * 0.25:.1f} km²")
            print(f"   - Mapa generado: prediccion_ovitrampas_merida.png")
        
    except Exception as e:
        print(f" ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()