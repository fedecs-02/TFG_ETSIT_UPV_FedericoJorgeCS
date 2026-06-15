#Modelo K-means clustering para dividir los equipos en distintos grupos con características similares
import os
os.environ["OMP_NUM_THREADS"] = "1"

#Librerías a importar:
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from math import pi
from sklearn.preprocessing import RobustScaler
import joblib

#CONFIGURACIÓN:
OUTDIR = "feb_data"
INPUT_PARTIDOS = os.path.join(OUTDIR, "match_stats_global.csv")
OUTPUT_CLUSTERS = os.path.join(OUTDIR, "match_stats_clusterizado.csv")
OUTPUT_CODO = os.path.join(OUTDIR, "clustering_codo.png")
OUTPUT_PCA = os.path.join(OUTDIR, "clustering_pca_mapa.png")
OUTPUT_HEATMAP = os.path.join(OUTDIR,"matriz_correlacion.png")
OUTPUT_BOXPLOT = os.path.join(OUTDIR, "analisis_distancia_boxplot.png")
OUTPUT_TOP20 = os.path.join(OUTDIR, "top20_distancias_centroide.png")
DIR_MODELOS = os.path.join(OUTDIR, "modelos_entrenados")
if not os.path.exists(DIR_MODELOS):
    os.makedirs(DIR_MODELOS)
#Determinamos el K óptimo con el codo:
K_OPTIMO = 6
#CÁLCULO DE MÉTRICAS AVANZADAS: Los 4 factores de Dean Oliver para el clustering
def calculo_metricas_avanzadas(df, ventana=6):
    print("Calculando métricas avanzadas:")

    df_loc = df[['id_partido', 'equipo_local', 'equipo_visitante', 
                 'loc_t2_met', 'loc_t2_int', 'loc_t3_met', 'loc_t3_int', 
                 'loc_tl_met', 'loc_tl_int', 'loc_reb_of', 'loc_reb_def', 'loc_perd',
                 'vis_reb_of', 'vis_reb_def']].copy()
    df_loc.columns = ['id_partido', 'equipo', 'rival', 
                      't2_met', 't2_int', 't3_met', 't3_int', 
                      'tl_met', 'tl_int', 'reb_of', 'reb_def', 'perd',
                      'avg_rival_reb_of', 'avg_rival_reb_def']
    df_vis = df[['id_partido', 'equipo_visitante', 'equipo_local', 
                 'vis_t2_met', 'vis_t2_int', 'vis_t3_met', 'vis_t3_int', 
                 'vis_tl_met', 'vis_tl_int', 'vis_reb_of', 'vis_reb_def', 'vis_perd',
                 'loc_reb_of', 'loc_reb_def']].copy()
    df_vis.columns = ['id_partido', 'equipo', 'rival', 
                      't2_met', 't2_int', 't3_met', 't3_int', 
                      'tl_met', 'tl_int', 'reb_of', 'reb_def', 'perd',
                      'avg_rival_reb_of', 'avg_rival_reb_def']
    # Unimos ambos y ordenamos cronológicamente
    df_unificado = pd.concat([df_loc, df_vis]).sort_values(by=['equipo', 'id_partido']).reset_index(drop=True)
    # Definimos las variables base
    columnas_base = [
        't2_int', 't2_met', 't3_int', 't3_met', 'tl_int', 'tl_met', 
        'perd', 'reb_of', 'reb_def', 'avg_rival_reb_def', 'avg_rival_reb_of'
    ]
    # Aplicamos la ventana móvil
    for col in columnas_base:
        if col in df_unificado.columns:
            df_unificado[f'roll_{col}'] = df_unificado.groupby('equipo')[col].transform(
                lambda x: x.shift(1).rolling(window=ventana, min_periods=3).mean()
            )
    # Eliminamos los partidos iniciales donde la media móvil no se ha podido realizar
    df_clean = df_unificado.dropna(subset=['roll_t2_int']).copy()
    # Cálculo de los 4 factores
    # PACE (Ritmo)
    df_clean['PACE'] = df_clean['roll_t2_int'] + df_clean['roll_t3_int'] + df_clean['roll_perd'] + (0.44 * df_clean['roll_tl_int']) - df_clean['roll_reb_of']
    # eFG% (Tiro Efectivo)
    tiros_campo = df_clean['roll_t2_int'] + df_clean['roll_t3_int']
    df_clean['EFG'] = np.where(tiros_campo > 0, 100 * (df_clean['roll_t2_met'] + 1.5 * df_clean['roll_t3_met']) / tiros_campo, 0)
    # TOV% (Pérdidas)
    df_clean['TOV'] = np.where(df_clean['PACE'] > 0, 100 * (df_clean['roll_perd'] / df_clean['PACE']), 0)
    # ORB% y DRB% (Rebotes)
    total_reb_of_disp = df_clean['roll_reb_of'] + df_clean['roll_avg_rival_reb_def']
    df_clean['ORB'] = np.where(total_reb_of_disp > 0, 100 * (df_clean['roll_reb_of'] / total_reb_of_disp), 0)
    total_reb_def_disp = df_clean['roll_reb_def'] + df_clean['roll_avg_rival_reb_of']
    df_clean['DRB'] = np.where(total_reb_def_disp > 0, 100 * (df_clean['roll_reb_def'] / total_reb_def_disp), 0)
    # FT/FGA (Tiros Libres)
    df_clean['FT/FGA'] = np.where(tiros_campo > 0, 100 * df_clean['roll_tl_met'] / tiros_campo, 0)
    return df_clean

def ejecucion_clustering():
    INPUT_PARTIDOS = os.path.join(OUTDIR, "match_stats_global.csv") 
    df = pd.read_csv(INPUT_PARTIDOS)
    # Cálculo de métricas avanzadas en las ventanas
    df_fiables = calculo_metricas_avanzadas(df, ventana=6)
    # Selección de variables a emplear en el modelo
    variables = ['EFG','TOV','ORB','DRB','FT/FGA','PACE']
    df_fiables[variables] = df_fiables[variables].fillna(0)
    print(f"Momentos de forma válidos para entrenar el modelo: {len(df_fiables)}")
    # Estandarización de variables para no desvirutar resultados (PACE es mucho mayor que ORB por ejemplo).
    scaler = RobustScaler()
    variables_limpias_escaladas = scaler.fit_transform(df_fiables[variables])
    #Generación gráfico del codo y método silueta para determinar el K óptimo:
    inertia = []
    silhouette_avg = []
    rango_k = range(2,12) #Probamos de 1 a 10 clusters
    for k in rango_k:
        kmeans_test = KMeans(n_clusters=k, random_state=42, n_init=30)
        cluster_labels = kmeans_test.fit_predict(variables_limpias_escaladas)
        inertia.append(kmeans_test.inertia_)
        silhouette_avg.append(silhouette_score(variables_limpias_escaladas, cluster_labels))
    fig, ax1 = plt.subplots(figsize=(10, 6))
    color = 'tab:blue'
    ax1.set_xlabel('Número de Clusters (K)')
    ax1.set_ylabel('Inercia (Suma de distancias intra-cluster al cuadrado)', color=color)
    ax1.plot(rango_k, inertia, 'bo-', color=color, markersize=8,linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Silhouette Score (Calidad)', color=color)
    ax2.plot(rango_k, silhouette_avg, 's--', color=color, markersize=8)
    ax2.tick_params(axis='y', labelcolor=color)
    plt.savefig(OUTPUT_CODO)
    plt.title('Determinación de K óptimo: Método del Codo vs Método de Silueta')
    plt.grid(True, alpha=0.3)
    print("Gráfica de validación guardada.")
    print(f"Aplicando K-Means con k={K_OPTIMO}")
    #K-Means
    kmeans = KMeans(n_clusters=K_OPTIMO, random_state=42, n_init=50)
    #Aprendizaje de grupos
    clusters_fiables = kmeans.fit_predict(variables_limpias_escaladas)
    df_fiables['cluster_equipo'] = clusters_fiables
    dists_fiables_all = kmeans.transform(variables_limpias_escaladas)
    df_fiables['distancia_centroide'] = [dists_fiables_all[i, lbl] for i, lbl in enumerate(kmeans.labels_)]
    joblib.dump(kmeans, os.path.join(DIR_MODELOS, "kmeans_model.pkl"))
    joblib.dump(scaler, os.path.join(DIR_MODELOS, "scaler_cluster.pkl"))
    print(f"Modelos guardados en: {DIR_MODELOS}")
    tabla_clusters = df_fiables[['id_partido', 'equipo', 'cluster_equipo', 'distancia_centroide']].copy()
    df_final = df_fiables.merge(
        tabla_clusters,
        left_on=['id_partido', 'rival'],
        right_on=['id_partido', 'equipo'],
        suffixes=('', '_del_rival')
    )
    df_final = df_final.drop(columns=['equipo_del_rival'])
    df_final.rename(columns={
        'cluster_equipo_del_rival': 'cluster_rival',
        'distancia_centroide_del_rival': 'dist_centroide_rival'
    }, inplace=True)
    # Resultados:
    print("\nPerfil de los estilos de juego (Centroides):")
    perfil = df_final.groupby('cluster_rival')[variables].mean() 
    perfil['num_estados_forma'] = df_final['cluster_rival'].value_counts() # Contamos cuántas "ventanas" hay en cada grupo
    # Renombramos variables
    perfil.rename(columns={
        'EFG': 'eFG% (Tiro)',
        'TOV': 'TOV% (Pérdidas)',
        'ORB': 'ORB% (Rebote Of)',
        'DRB' : 'DRB% (Rebote Def)',
        'FT/FGA': 'FT/FGA (Tiros Lib)',
        'PACE': 'Pace (Ritmo)'
    }, inplace=True)
    print(perfil.round(3).to_string())
    # Mapa 2D - PCA
    variables_totales_escaladas = scaler.transform(df_final[variables])
    pca = PCA(n_components=2)
    variables_pca = pca.fit_transform(variables_totales_escaladas)    
    plt.figure(figsize=(12,8))
    sns.scatterplot(x=variables_pca[:,0], y=variables_pca[:,1], hue=df_final['cluster_equipo'], palette='viridis', s=50, alpha=0.5)
    plt.title(f'Mapa de Estilos (PCA) - {K_OPTIMO} Clusters Temporales')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.legend(title='Cluster')
    plt.grid(True, alpha=0.3)
    plt.savefig(OUTPUT_PCA)
    print("Mapa PCA guardado")
    # Matriz de Correlación
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_final[variables].corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlación entre Factores (Ventanas Móviles)')
    plt.savefig(OUTPUT_HEATMAP)
    print("Mapa Heatmap guardado")
    # Guardado de partidos con clusters
    df_final.to_csv(OUTPUT_CLUSTERS, index=False, encoding='utf-8-sig')
    # Visualización de distancias al centroide y outliers
    plt.figure(figsize=(10, 6))
    sns.boxplot(y='distancia_centroide', data=df_final, palette='Set2')
    plt.title('Distribución de Distancias al Centroide')
    plt.ylabel('Distancia Euclidiana (Rareza del estilo)') 
    plt.grid(True, axis='y', alpha=0.3)
    plt.savefig(OUTPUT_BOXPLOT)
    print("Mapa Boxplot guardado")
    # Ranking Outliers (Los estados de forma más extraños de la liga)
    top_outliers = df_final.sort_values('distancia_centroide', ascending=False).head(20)  
    plt.figure(figsize=(12, 6))
    top_outliers['etiqueta_outlier'] = top_outliers['equipo'] + " (J" + top_outliers['id_partido'].astype(str) + ")"
    sns.barplot(x='distancia_centroide', y='etiqueta_outlier', data=top_outliers, palette='viridis')
    plt.title('Top 20 Momentos de Forma más inusuales (Outliers)')
    plt.xlabel('Distancia al centroide')
    plt.ylabel('Equipo (Partido ID)')    
    # Línea de corte
    umbral = df_final['distancia_centroide'].mean() + 2 * df_final['distancia_centroide'].std()
    plt.axvline(umbral, color='red', linestyle='--', label=f'Umbral de anomalía ({umbral:.2f})')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(OUTPUT_TOP20)
    print("Gráfico Top 20 guardado")
if __name__ == "__main__":
    ejecucion_clustering()