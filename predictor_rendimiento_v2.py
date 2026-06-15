
import pandas as pd # Para manipulación de datos
import numpy as np # Para operaciones matemáticas
import os # Para gestión de archivos y carpetas
import joblib # Para guardar los modelos entrenados
import xgboost as xgb # Algoritmo de árboles de decisión a implementar en los modelos
import matplotlib.pyplot as plt # Para generar gráficos estáticos
import seaborn as sns # Para estética de los gráficos
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score, roc_auc_score, confusion_matrix,roc_curve, auc, precision_recall_curve #Métricas de evaluación del modelo
from sklearn.model_selection import train_test_split # Para dividir la muestra de entrenamiento y la muestra de test
from sklearn.calibration import calibration_curve # Para la evaluación de la fiabilidad de las probabilidades 
from sklearn.model_selection import RandomizedSearchCV #Optimizador inteligente de hiperparámetros
import shap # Para interpretabilidad avanzada de IA
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 0. CONFIGURACIÓN
DIR_DATA = "feb_data"
DIR_MODELOS = os.path.join(DIR_DATA, "modelos_entrenados")
if not os.path.exists(DIR_MODELOS): 
    os.makedirs(DIR_MODELOS)
FILE_JUGADORAS = os.path.join(DIR_DATA, "estadisticas_jugadoras_vf.csv")
FILE_EQUIPOS_CLUSTERED = os.path.join(DIR_DATA, "match_stats_clusterizado.csv") 

# 1. MÉTRICA PTC (PLAYER TOTAL CONTRIBUTION)
PESOS_PTC = {
    'puntos': 1.0,'tap_fav': 0.91,'reb_def': 0.58,'reb_of': 0.92,'recup': 0.86,        
    'asist': 0.48,'faltas_r': 0.23,'tc_fail': -0.91,'tl_fail': -0.57,'perd': -0.86,'faltas_c': -0.23}

def calcular_ptc_row(row):
    # Tiros fallados
    tc_f = (row['t2_int'] - row['t2_met']) + (row['t3_int'] - row['t3_met'])
    tl_f = row['tl_int'] - row['tl_met']
    # Ponderación de variables
    return (row['puntos'] * PESOS_PTC['puntos'] + row['tap_fav'] * PESOS_PTC['tap_fav'] +
            row['reb_def'] * PESOS_PTC['reb_def'] + row['reb_of'] * PESOS_PTC['reb_of'] +
            row['recup'] * PESOS_PTC['recup'] + row['asist'] * PESOS_PTC['asist'] +
            row['faltas_r'] * PESOS_PTC['faltas_r'] + tc_f * PESOS_PTC['tc_fail'] +
            tl_f * PESOS_PTC['tl_fail'] + row['perd'] * PESOS_PTC['perd'] + row['faltas_c'] * PESOS_PTC['faltas_c'])

def main():
    print("Inicio del entrenamiento del Modelo:")
    
    # 3. CARGA Y LIMPIADO DE DATOS
    df = pd.read_csv(FILE_JUGADORAS).fillna(0) #Elimina los NaN
    equipos = pd.read_csv(FILE_EQUIPOS_CLUSTERED)
    # 4. IDENTIFICACIÓN DEL EQUIPO RIVAL
    partidos = df.groupby('id_partido')['equipo'].unique()    
    def get_rival(pid, my_team):
        teams = partidos.get(pid, [])
        if len(teams) == 2:
            return teams[0] if teams[1] == my_team else teams[1]
        return None
    # Nueva columna con el equipo rival
    df['rival'] = df.apply(lambda x: get_rival(x['id_partido'], x['equipo']), axis=1)
    # Eliminamos registros erróneos
    df = df.dropna(subset=['rival'])
    # Unimos columnas del estilo del equipo rival a cada jugadora
    # Como la tabla de equipos ya trae "equipo" y "rival", cruzamos directamente por nuestro propio equipo
    info_partido = equipos[['id_partido', 'equipo', 'cluster_rival', 'dist_centroide_rival']].rename(
        columns={'cluster_rival': 'cluster_def_rival'}
    )
    
    # Cruzamos para que cada jugadora tenga el estilo de su rival
    df = df.merge(info_partido, on=['id_partido', 'equipo'], how='left').dropna(subset=['cluster_def_rival'])
    
    # Para el PACE (Ritmo), sí necesitamos ir a buscar el 'PACE' de la fila del rival
    info_ritmo = equipos[['id_partido', 'equipo', 'PACE']].rename(
        columns={'equipo': 'rival', 'PACE': 'pace_rival'}
    )
    df = df.merge(info_ritmo, on=['id_partido', 'rival'], how='left')
    
    # 5. CONFIGURACIÓN DE VARIABLES OBJETIVOS
    df['PTC'] = df.apply(calcular_ptc_row, axis=1)
    # Normalizamos minutos jugados y multiplicamos por el ritmo del rival para calcular el nº de posesiones ajustado al ritmo de partido
    posesiones = (df['minutos_float'].replace(0, 0.1) / 40) * df['pace_rival']
    # Calculamos el PTC normalizado sobre 100 (%) como estándar analítico
    df['PTC_mp'] = (df['PTC'] / posesiones) * 100
    # Calculamos el umbral de éxito (percentil 50)
    UMBRAL_EXITO = df['PTC_mp'].quantile(0.5)
    print(f" El Percentil 50 del PTC_mp es {UMBRAL_EXITO:.2f}")
    print(f" Toda actuación superior a {UMBRAL_EXITO:.2f} será considerada 'Éxito'.")
    # Creamos la variable binaria de éxito
    df['es_exito'] = np.where(df['PTC_mp'] >= UMBRAL_EXITO, 1, 0)
    # Calculamos el uso absoluto de la jugadora en el partido
    uso_jugadora = df['t2_int'] + df['t3_int'] + (0.44 * df['tl_int']) + df['perd'] #0.44 es una ponderación estadística para las posesiones que acaban en tiros libres
    # Evitamos divisiones por cero por si no hay tiros ni pérdidas de la jugadora
    uso_jugadora = uso_jugadora.replace(0, 0.1) 
    df['PTS_mpu'] = (df['puntos'] / (df['minutos_float'].replace(0, 0.1) * uso_jugadora)) * 40 #puntos esperados por 40 minutos de uso continuo
    # eFG% (Porcentaje de Tiro Efectivo)
    tiros_totales = df['t2_int'] + df['t3_int']
    df['eFG'] = np.where(tiros_totales > 0, 
                         ((df['t2_met'] + 1.5 * df['t3_met']) / tiros_totales) * 100, 
                         0)                  
    # USG% (Porcentaje de Uso de la Jugadora)
    df['USG'] = (uso_jugadora / posesiones) * 100

    # 6. DEFINICIÓN DE VARIABLES PREDICTORAS
    df = df.sort_values(['nombre', 'temporada', 'id_partido']) # Ordenamos cronológicamente
    # Lista de variables de las que queremos calcular el histórico reciente de la jugadora
    cols_stats = [
        'puntos', 'reb_tot', 'asist', 'PTC_mp', 'minutos_float','PTS_mpu', 'eFG', 'USG', 
        'recup', 'tap_fav', 'perd', 'reb_of', 'reb_def', 'faltas_c', 'faltas_r', 
        't2_int', 't2_met', 't3_int', 't3_met', 'tl_int', 'tl_met']
    for col in cols_stats:
        # L6 (Last 6): Racha Reciente (Media Móvil Exponencial EMA).
        df[f'L6_{col}'] = df.groupby('nombre')[col].transform(lambda x: x.shift().ewm(span=6, min_periods=1).mean())
        # Sn (Season): Rendimiento Global.
        # expanding().mean()para la media acumulada desde el inicio de la temporada hasta el partido anterior.
        df[f'Sn_{col}'] = df.groupby(['nombre', 'temporada'])[col].transform(lambda x: x.shift().expanding().mean())
        if col in ['PTC_mp', 'puntos']:
            df[f'L6_{col}_std'] = df.groupby('nombre')[col].transform(lambda x: x.shift().rolling(6, min_periods=2).std().fillna(0))
            df[f'L6_{col}_min'] = df.groupby('nombre')[col].transform(lambda x: x.shift().rolling(6, min_periods=1).min().fillna(0))
            
            if col == 'PTC_mp':
                df['es_partido_malo'] = (df['PTC_mp'] < 0).astype(int)
                df['Sn_ratio_malos'] = df.groupby(['nombre', 'temporada'])['es_partido_malo'].transform(lambda x: x.shift().expanding().mean().fillna(0))
    # Eliminamos las filas del primer partido
    df_train = df.dropna(subset=['L6_puntos', 'Sn_puntos']).copy()
    # Determinamos factor cancha en formato binario
    if df_train['es_local'].dtype == 'object':
        df_train['es_local'] = np.where(df_train['es_local'].str.lower().str.contains('local'), 1, 0) 
    # Convertimos el clúster numérico en variables categóricas (One-Hot Encoding)
    df_train['cluster_def_rival'] = df_train['cluster_def_rival'].astype(int)
    df_train = pd.get_dummies(df_train, columns=['cluster_def_rival'], drop_first=False)
    for i in range(6): 
        col_name = f'cluster_def_rival_{i}'
        if col_name not in df_train.columns:
            df_train[col_name] = 0.0
    # Definimos los inputs para el modelo
    features = [
        'minutos_float','Sn_puntos', 'Sn_reb_tot', 'Sn_asist', 'Sn_PTC_mp', 'Sn_minutos_float','Sn_PTS_mpu', 
        'cluster_def_rival_0', 'cluster_def_rival_1', 'cluster_def_rival_2', 
        'cluster_def_rival_3', 'cluster_def_rival_4', 'cluster_def_rival_5', 'dist_centroide_rival', 'es_local', 'Sn_eFG', 'Sn_USG', 'Sn_recup', 'Sn_tap_fav', 'Sn_perd', 'pace_rival',
        'L6_puntos', 'L6_reb_tot', 'L6_asist', 'L6_PTC_mp', 'L6_minutos_float','L6_PTS_mpu', 
        'L6_eFG', 'L6_USG', 'L6_recup', 'L6_tap_fav', 'L6_perd', 'Sn_reb_of', 'Sn_reb_def', 'Sn_faltas_c', 'Sn_faltas_r', 'Sn_t2_int', 'Sn_t2_met', 
        'Sn_t3_int', 'Sn_t3_met', 'Sn_tl_int', 'Sn_tl_met', 'L6_PTC_mp_std', 'L6_PTC_mp_min', 'L6_puntos_std', 'L6_puntos_min', 'Sn_ratio_malos'
    ]
    # Lo que vamos a intentar predecir
    targets = ['puntos', 'asist', 'reb_of', 'reb_def', 'recup', 'tap_fav', 
    'perd', 'faltas_c', 'faltas_r', 't2_int', 't2_met', 
    't3_int', 't3_met', 'tl_int', 'tl_met','eFG','USG']

    # 7. SEPARACIÓN DATOS DE ENTRENAMIENTO Y TEST   
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(df_train, groups=df_train['id_partido']))
    df_test_completo = df_train.iloc[test_idx].copy()
    X_train = df_train.iloc[train_idx][features]
    y_train = df_train.iloc[train_idx]['es_exito']
    X_test = df_test_completo[features]
    y_test = df_test_completo['es_exito']

    # 8. ENTRENAMIENTO SISTEMAS DE REGRESIÓN
    regresores = {}
    resultados_regresion = []
    print("\n--- EVALUACIÓN DE PREDICCIONES VS BASELINE (Error Absoluto Medio - MAE) ---")
    for t in targets: 
        model = xgb.XGBRegressor(n_estimators=150, max_depth=5, learning_rate=0.04, n_jobs=-1, random_state=42)
        model.fit(X_train, df_train.iloc[train_idx][t])
        regresores[t] = model
        # Hacemos las predicciones con la IA en el grupo de Test
        predicciones_ia = model.predict(X_test)
        y_test_real = df_test_completo[t]
    
        # Calculamos el BASELINE (Media de la temporada)
        columna_baseline = f"Sn_{t}"
        if columna_baseline in X_test.columns:
            predicciones_baseline = X_test[columna_baseline]
            
            # Errores Absolutos (MAE - Unidades Reales)
            mae_ia = mean_absolute_error(y_test_real, predicciones_ia)
            mae_baseline = mean_absolute_error(y_test_real, predicciones_baseline)
            mejora_mae = mae_baseline - mae_ia 
            # RMSE
            rmse_ia = np.sqrt(mean_squared_error(y_test_real, predicciones_ia))
            rmse_baseline = np.sqrt(mean_squared_error(y_test_real, predicciones_baseline))
            mejora_rmse = rmse_baseline - rmse_ia

            resultados_regresion.append({
                    "Variable": t, 
                    "MAE_IA_uds": round(mae_ia, 3), 
                    "MAE_Media_uds": round(mae_baseline, 3), 
                    "Mejora_MAE_uds": round(mejora_mae, 3),
                    "RMSE_IA_uds": round(rmse_ia, 3),
                    "RMSE_Media_uds": round(rmse_baseline, 3),
                    "Mejora_RMSE_uds": round(mejora_rmse, 3)
                })
            # 6. Imprimimos la comparativa completa
            print(f"[{t.upper()}]")
            print(f"  > Absoluto (Uds) : IA se equivoca por {mae_ia:.2f} | Media se equivoca por {mae_baseline:.2f} -> MEJORA IA: {mejora_mae:+.2f}")
            print(f"  > RMSE           : IA = {rmse_ia:.2f} | Media = {rmse_baseline:.2f} -> MEJORA IA: {mejora_rmse:+.2f}\n")
        else:
            # Si no hay baseline directo (ej. PTS_mpu)
            mae_ia = mean_absolute_error(y_test_real, predicciones_ia)
            rmse_ia = np.sqrt(mean_squared_error(y_test_real, predicciones_ia))
            print(f"[{t.upper()}] MAE IA: {mae_ia:.2f} | RMSE IA: {rmse_ia:.2f} (Sin baseline)")
    df_metricas_reg = pd.DataFrame(resultados_regresion)
    df_metricas_reg.to_csv(os.path.join(DIR_DATA, "metricas_regresion_baseline.csv"), index=False)
    # =====================================================================
    # 8.5 COMPARATIVA DE LOS 4 ESCENARIOS JUGADORA A JUGADORA
    # =====================================================================
    print("\n--- INICIANDO SIMULADOR DE ESCENARIOS (JUGADORA A JUGADORA) ---")

    # 1. Función de optimización adaptada al entorno de test
    def optimizar_minutos_simulador(df_partido, col_eficiencia):
        mins_seguros = df_partido['minutos_float'].replace(0, 0.1)
        eficiencia = df_partido[col_eficiencia] / mins_seguros
        pesos = eficiencia.clip(lower=0).values

        if np.sum(pesos) == 0:
            return df_partido['minutos_float'].values

        mins_asignados = np.zeros(len(df_partido))
        mins_restantes = 200.0
        tope = 35.0

        while mins_restantes > 0.1 and np.sum(pesos) > 0:
            cuotas = (pesos / np.sum(pesos)) * mins_restantes
            for i in range(len(df_partido)):
                if pesos[i] > 0:
                    if mins_asignados[i] + cuotas[i] > tope:
                        mins_restantes -= (tope - mins_asignados[i])
                        mins_asignados[i] = tope
                        pesos[i] = 0
                    else:
                        mins_asignados[i] += cuotas[i]
                        mins_restantes -= cuotas[i]
        return mins_asignados

    # 2. Generar TODAS las predicciones del modelo para el test set
    for t in targets:
        df_test_completo[f'{t}_pred'] = regresores[t].predict(X_test).clip(min=0)

    # 3. Calcular el PTC Proyectado por la IA para cada jugadora
    fallos_t2_pred = (df_test_completo['t2_int_pred'] - df_test_completo['t2_met_pred']).clip(lower=0)
    fallos_t3_pred = (df_test_completo['t3_int_pred'] - df_test_completo['t3_met_pred']).clip(lower=0)
    fallos_tl_pred = (df_test_completo['tl_int_pred'] - df_test_completo['tl_met_pred']).clip(lower=0)

    df_test_completo['PTC_Proy_IA'] = (
        df_test_completo['puntos_pred'] * PESOS_PTC['puntos'] +
        df_test_completo['tap_fav_pred'] * PESOS_PTC['tap_fav'] +
        df_test_completo['reb_def_pred'] * PESOS_PTC['reb_def'] +
        df_test_completo['reb_of_pred'] * PESOS_PTC['reb_of'] +
        df_test_completo['recup_pred'] * PESOS_PTC['recup'] +
        df_test_completo['asist_pred'] * PESOS_PTC['asist'] +
        df_test_completo['faltas_r_pred'] * PESOS_PTC['faltas_r'] +
        (fallos_t2_pred + fallos_t3_pred) * PESOS_PTC['tc_fail'] +
        fallos_tl_pred * PESOS_PTC['tl_fail'] +
        df_test_completo['perd_pred'] * PESOS_PTC['perd'] +
        df_test_completo['faltas_c_pred'] * PESOS_PTC['faltas_c']
    )

    # 4. Bucle Partido a Partido y Jugadora a Jugadora
    resultados_escenarios = []
    
    vars_a_comparar = ['puntos', 'asist', 'reb_of', 'reb_def', 'recup', 'tap_fav', 'perd', 'faltas_c', 'faltas_r']

    # Primero, calculamos los minutos óptimos de CADA PARTIDO para todas las jugadoras de ese equipo
    df_test_completo['Min_Optimos_IA'] = 0.0
    for partido_id, df_partido in df_test_completo.groupby(['id_partido', 'equipo']):
        min_optimos = optimizar_minutos_simulador(df_partido, 'PTC_Proy_IA')
        df_test_completo.loc[df_partido.index, 'Min_Optimos_IA'] = min_optimos

    # Ahora sí, iteramos fila por fila (jugadora a jugadora)
    for index, row in df_test_completo.iterrows():
        fila_resultado = {
            'ID_Partido': row['id_partido'],
            'Equipo': row['equipo'],
            'Jugadora': row['nombre'], # AÑADIMOS EL NOMBRE
            'Min_Reales': round(row['minutos_float'], 1),
            'Min_Optimos_IA': round(row['Min_Optimos_IA'], 1)
        }
        
        mins_reales = row['minutos_float']
        mins_seguros = max(mins_reales, 0.1)
        min_optimos = row['Min_Optimos_IA']
        
        # --- CÁLCULO DEL PTC ---
        e1_ptc = row['PTC']
        e2_ptc = row['PTC_Proy_IA']
        # E3: Proyectamos el PTC en base a los nuevos minutos óptimos
        e3_ptc = (row['PTC_Proy_IA'] / mins_seguros) * min_optimos
        
        # E4: Media Histórica (Normalizada por minutos reales)
        eficiencia_media_ptc = (row['Sn_PTC_mp'] / 100) * (row['pace_rival'] / 40)
        e4_ptc = eficiencia_media_ptc * mins_reales
        
        fila_resultado.update({
            'E1_PTC_Real': round(e1_ptc, 2),
            'E2_PTC_IA_MinReales': round(e2_ptc, 2),
            'E3_PTC_IA_MinOptimos': round(e3_ptc, 2),
            'E4_PTC_Media_MinReales': round(e4_ptc, 2)
        })
        
        # --- CÁLCULO PARA EL RESTO DE VARIABLES (Puntos, Asistencias, etc) ---
        for var in vars_a_comparar:
            e1_val = row[var]
            e2_val = row[f'{var}_pred']
            
            # E3: Proyectamos la variable multiplicando su eficiencia por minuto por los mins óptimos
            eficiencia_ia_var = row[f'{var}_pred'] / mins_seguros
            e3_val = eficiencia_ia_var * min_optimos
            
            # E4: Media histórica de la variable * mins reales
            if f'Sn_{var}' in row.index:
                mins_hist_seguros = max(row['Sn_minutos_float'], 0.1)
                eficiencia_media_var = row[f'Sn_{var}'] / mins_hist_seguros
                e4_val = eficiencia_media_var * mins_reales
            else:
                e4_val = 0.0
                
            fila_resultado.update({
                f'E1_{var}_Real': round(e1_val, 2),
                f'E2_{var}_IA': round(e2_val, 2),
                f'E3_{var}_Optimo': round(e3_val, 2),
                f'E4_{var}_Media': round(e4_val, 2)
            })
            
        resultados_escenarios.append(fila_resultado)

    # 5. Guardar resultados y sacar resumen por consola
    df_escenarios = pd.DataFrame(resultados_escenarios)
    df_escenarios.to_csv(os.path.join(DIR_DATA, "comparativa_4_escenarios_jugadoras.csv"), index=False)

    print("✅ SIMULACIÓN JUGADORA A JUGADORA COMPLETADA.")
    print("Archivo generado: 'comparativa_4_escenarios_jugadoras.csv'")
    # 9. OPTIMIZACIÓN DE CLASIFICADORES (RANDOMIZED SEARCH CV)    
    param_dist = {
        'max_depth': [3, 4, 5, 6], # Para evitar el sobreajuste, se limita el crecimiento del árbol
        'learning_rate': [0.01, 0.03, 0.05, 0.1],
        'n_estimators': [100, 150, 200, 250],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0] #Añadir un factor aleatorio en las variables para ganar robustez
    }
    xgb_base = xgb.XGBClassifier(random_state=42, n_jobs=-1)
    print("Configuración Modelo")
    random = RandomizedSearchCV(estimator=xgb_base, param_distributions=param_dist, 
                                      n_iter=25, scoring='roc_auc', cv=3, verbose=0, random_state=42)
    random.fit(X_train, y_train)
    clasificador = random.best_estimator_
    print(f"Parámetros Modelo: {random.best_params_}")

    # 10. TEST DE VALIDACIÓN
    # Predicciones de Test
    y_pred = clasificador.predict(X_test)
    y_prob = clasificador.predict_proba(X_test)[:,1]
    # Función de KPIs de Validación del modelo
    def evaluar_modelo(y_true, y_pred, y_prob, nombre):
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        roc = roc_auc_score(y_true, y_prob)
        print(f"Resultados Test Modelo {nombre}")
        print(f"Accuracy  : {acc:.3f}")
        print(f"Precisión : {prec:.3f}")
        print(f"Recall    : {rec:.3f}")
        print(f"F1-score  : {f1:.3f}")
        print(f"ROC-AUC   : {roc:.3f}")
        return acc, prec, rec, f1, roc
    # Aplicamos la fórmula
    metricas = evaluar_modelo(y_test, y_pred, y_prob, "Modelo Predictivo")
    # Guardamos los resultados en un CSV 
    tabla_metricas = pd.DataFrame({
        "Modelo": ["Modelo Predictivo"],
        "Accuracy": [metricas[0]],
        "Precisión": [metricas[1]],
        "Recall": [metricas[2]],
        "F1": [metricas[3]],
        "ROC-AUC": [metricas[4]]
    })
    tabla_metricas.to_csv(os.path.join(DIR_DATA, "metricas_validacion_modelo_predictivo.csv"), index=False)

    # 11. GENERACIÓN DE GRÁFICAS EXPLICATIVAS

    # Gráfica 1: Histograma y Umbral de éxito
    plt.figure(figsize=(10, 6))
    sns.histplot(df['PTC_mp'], bins=50, kde=True, color='skyblue')
    plt.axvline(UMBRAL_EXITO, color='red', linestyle='--', linewidth=2, label=f'Umbral de Éxito: {UMBRAL_EXITO:.1f}')
    plt.title('Distribución del Rendimiento (PTC por cada 100 posesiones)')
    plt.xlim(-60, 60)
    plt.legend()
    plt.savefig(os.path.join(DIR_DATA, "distribucion_umbral_exito.png"))
    plt.close()

    # Gráfica 2: Comparativa Importancia de Variables
    plt.figure(figsize=(10, 8))
    df_imp = pd.DataFrame({'Variable': features, 'Importancia': clasificador.feature_importances_}).sort_values('Importancia', ascending=False).head(15)
    sns.barplot(x='Importancia', y='Variable', data=df_imp, palette='Greens_r')
    plt.title('Importancia Variables')
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_DATA, "feature_importance_comparativa.png"))
    plt.close()

    # Gráfica 3: Matriz de Confusión
    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Greens', cbar=False,
                xticklabels=['Fracaso Previsto', 'Éxito Previsto'],
                yticklabels=['Fracaso Real', 'Éxito Real'])
    plt.title('Matriz Confusión (Test)')
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_DATA, "matriz_confusion_test.png"))
    plt.close()

    # Gráfica 4: Valores SHAP
    explainer = shap.TreeExplainer(clasificador)
    shap_values = explainer.shap_values(X_test)
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig(os.path.join(DIR_DATA, "shap_summary.png"), bbox_inches='tight')
    plt.close()

    # Gráfica 5: Curva ROC
    # Visualiza el trade-off entre Sensibilidad y Especificidad
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Azar (0.5)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Capacidad Discriminatoria del Modelo')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_DATA, "roc_curve.png"))
    plt.close()

    # 12. GUARDADO DE MODELOS
    print("Modelo entrenado. Procedemos a guardarlo")
    joblib.dump(regresores, os.path.join(DIR_MODELOS, "regresores.pkl"))
    joblib.dump(clasificador, os.path.join(DIR_MODELOS, "clasificador.pkl"))
    joblib.dump(features, os.path.join(DIR_MODELOS, "features.pkl"))

    print(f"Modelo guardado en {DIR_MODELOS}")

if __name__ == "__main__":
    main()