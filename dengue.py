# ¿cómo fue la distribucion de ovitrampas en 2024 en el municipio de Merida?
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

class DengueVisualizer:
    """Visualización de datos de ovitrampas"""
    
    def __init__(self, config):
        self.config = config
        
    def load_and_validate_data(self):
        """Carga y valida los datos básicos"""
        print("Cargando datos...")
        
        if not Path(self.config.csv_path).exists():
            raise FileNotFoundError(f"No se encuentra: {self.config.csv_path}")
        if not Path(self.config.merida_shp).exists():
            raise FileNotFoundError(f"No se encuentra: {self.config.merida_shp}")
        
        # Cargar shapefile de Mérida
        merida = gpd.read_file(self.config.merida_shp)
        if merida.crs is None:
            merida.set_crs("EPSG:4326", inplace=True)
        
        #datos_ovitrampas
        df = pd.read_csv(self.config.csv_path)
        
        # Validar columnas
        required_cols = ['x', 'y', 'eggs']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Faltan columnas: {missing_cols}")
        
        # Crear GeoDataFrame
        gdf = gpd.GeoDataFrame(
            df, 
            geometry=gpd.points_from_xy(df.x, df.y),
            crs="EPSG:4326"
        )
        
        # Proyectar a UTM
        gdf = gdf.to_crs(self.config.target_crs)
        merida = merida.to_crs(self.config.target_crs)
        
        # Filtrar puntos dentro de Mérida
        puntos_dentro = gdf[gdf.within(merida.unary_union)]
        puntos_fuera = gdf[~gdf.within(merida.unary_union)]
        
        print(f"Datos cargados:")
        print(f"   - Puntos dentro de Mérida: {len(puntos_dentro)}")
        print(f"   - Puntos fuera de Mérida: {len(puntos_fuera)}")
        print(f"   - Total de puntos: {len(gdf)}")
        
        return puntos_dentro, merida
    
    def create_basic_maps(self):
        """Crea mapas básicos de visualización"""
        print("Creando visualizaciones básicas...")
        
        gdf, merida = self.load_and_validate_data()
        
        # 1. Mapa de calor
        self._create_heatmap(gdf, merida, "mapa_calor.png")
        
        # 2. Mapa por categorías (si hay muchos datos)
        self._create_categorized_map(gdf, merida, "mapa_categorias.png")
        
        # 3. Estadística
        self._show_basic_stats(gdf)
        
        print("Visualizaciones completadas!")
    
    def _create_points_map(self, gdf, merida, output_path):
        """Crea un mapa simple de puntos"""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Fondo del polígono de Mérida
        merida.plot(ax=ax, color='lightgray', alpha=0.5, edgecolor='black')
        merida.boundary.plot(ax=ax, color='black', linewidth=2)
        
        # Puntos coloreados por cantidad de huevos
        scatter = ax.scatter(
            gdf.geometry.x, 
            gdf.geometry.y, 
            c=gdf['eggs'], 
            cmap='YlOrRd', 
            s=50, 
            alpha=0.7,
            edgecolors='black',
            linewidth=0.5
        )
        
        # Colorbar
        plt.colorbar(scatter, ax=ax, label='Cantidad de Huevos')
        
        # Configuración
        ax.set_title('Distribución de Ovitrampas en Mérida', fontsize=16, fontweight='bold')
        ax.set_xlabel('Coordenada X (UTM)')
        ax.set_ylabel('Coordenada Y (UTM)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Mapa de puntos: {output_path}")
    
    def _create_heatmap(self, gdf, merida, output_path):
        """Crea un mapa de calor básico"""
        from scipy.stats import gaussian_kde
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Fondo del polígono
        merida.plot(ax=ax, color='white', alpha=0.8, edgecolor='black')
        
        # Preparar datos para KDE
        x = gdf.geometry.x.values
        y = gdf.geometry.y.values
        weights = gdf['eggs'].values
        
        # Crear grid para el heatmap
        xi = np.linspace(x.min(), x.max(), 100)
        yi = np.linspace(y.min(), y.max(), 100)
        xi, yi = np.meshgrid(xi, yi)
        
        # KDE con pesos (cantidad de huevos)
        try:
            # Aplanar los datos para KDE
            coordinates = np.vstack([x, y])
            kde = gaussian_kde(coordinates, weights=weights)
            zi = kde(np.vstack([xi.flatten(), yi.flatten()]))
            zi = zi.reshape(xi.shape)
            
            # Plot heatmap
            im = ax.contourf(xi, yi, zi, levels=15, alpha=0.6, cmap='YlOrRd')
            plt.colorbar(im, ax=ax, label='Densidad de Huevos')
            
        except Exception as e:
            print(f"⚠️ No se pudo generar heatmap KDE: {e}")
            # Fallback: scatter plot simple
            scatter = ax.scatter(x, y, c=weights, cmap='YlOrRd', s=30, alpha=0.6)
            plt.colorbar(scatter, ax=ax, label='Cantidad de Huevos')
        
        # Puntos originales
        ax.scatter(x, y, color='blue', s=10, alpha=0.5, label='Ovitrampas')
        
        ax.set_title('Mapa de Calor - Densidad de Huevos', fontsize=16, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Mapa de calor: {output_path}")
    
    def _create_categorized_map(self, gdf, merida, output_path):
        """Crea mapa con puntos categorizados"""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Fondo
        merida.plot(ax=ax, color='lightgray', alpha=0.3, edgecolor='black')
        
        # Categorizar los datos
        gdf_copy = gdf.copy()
        gdf_copy['categoria'] = pd.cut(
            gdf['eggs'], 
            bins=[0, 10, 50, 100, gdf['eggs'].max()], 
            labels=['Muy Bajo', 'Bajo', 'Medio', 'Alto']
        )
        
        # Colores para categorías
        colors = {'Muy Bajo': 'green', 'Bajo': 'yellow', 'Medio': 'orange', 'Alto': 'red'}
        
        for categoria, color in colors.items():
            mask = gdf_copy['categoria'] == categoria
            if mask.any():
                subset = gdf_copy[mask]
                ax.scatter(
                    subset.geometry.x, 
                    subset.geometry.y, 
                    c=color, 
                    label=f'{categoria} (n={len(subset)})',
                    s=50, 
                    alpha=0.7,
                    edgecolors='black'
                )
        
        ax.set_title('Ovitrampas por Categoría de Huevos', fontsize=16, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Mapa categorizado: {output_path}")
    
    def _show_basic_stats(self, gdf):
        """Muestra estadísticas básicas de los datos"""
        print("\n ESTADÍSTICAS BÁSICAS:")
        print(f"   - Total de ovitrampas: {len(gdf)}")
        print(f"   - Huevos mínimo: {gdf['eggs'].min()}")
        print(f"   - Huevos máximo: {gdf['eggs'].max()}")
        print(f"   - Huevos promedio: {gdf['eggs'].mean():.2f}")
        print(f"   - Huevos mediana: {gdf['eggs'].median()}")
        print(f"   - Desviación estándar: {gdf['eggs'].std():.2f}")
        
        # Distribución por cuartiles
        print(f"   - Cuartiles: {gdf['eggs'].quantile([0.25, 0.5, 0.75]).to_dict()}")