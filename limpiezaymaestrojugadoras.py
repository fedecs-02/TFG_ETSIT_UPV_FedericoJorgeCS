import pandas as pd #Para manipulación de datos y guardado en csv
import numpy as np
import os #Para gestión de archivos

#CONFIGURACIÓN
OUTDIR = "feb_data"
INPUT_CSV = os.path.join(OUTDIR, "players_stats_detailed.csv")
#CSV sin jugadoras con 0 minutos jugados
OUTPUT_CLEAN = os.path.join(OUTDIR, "estadisticas_jugadoras_vf.csv")
#Maestro jugadoras
OUTPUT_MASTER = os.path.join(OUTDIR, "maestro_jugadoras.csv")

#Cambio de minutos a valor numérico
def time_to_float(time_str):
    if not isinstance(time_str, str): return 0.0
    time_str = str(time_str).strip()
    
    # Si es "00:00" o vacío, devuelve 0
    if time_str in ["00:00", "0", "", "nan"]: return 0.0
    
    # Si tiene dos puntos, convertimos minutos y segundos
    if ":" in time_str:
        try:
            parts = time_str.split(":")
            return int(parts[0]) + (int(parts[1]) / 60)
        except:
            return 0.0
    try:
        return float(time_str) #Si no tiene decimales, devolvemos el valor sin decimal
    except:
        return 0.0


#FUNCIÓN PRINCIPAL
def procesar_datos():
    #Carga de datos
    if not os.path.exists(INPUT_CSV):
        print(f"ERROR: No se encuentra el archivo {INPUT_CSV}")
        return
    df = pd.read_csv(INPUT_CSV)
    total_filas = len(df)
    print(f"Registros leídos: {total_filas}")

    #Limpieza de datos
    df['minutos'] = df['minutos'].astype(str).str.strip()
    
    #Aplicamos filtro para eliminar las filas con 00:00,0, vacíos o nulos
    df_clean = df[
        (df['minutos'] != '00:00') & 
        (df['minutos'] != '0') & 
        (df['minutos'] != '') & 
        (df['minutos'] != 'nan')
    ].copy() #Crea una copia para evitar errores inesperados

    #Cálculo de filas eliminadas
    eliminadas = total_filas - len(df_clean)
    print(f"Filas eliminadas: {eliminadas}")
    print(f"Registros válidos: {len(df_clean)}")

    #PREPARACIÓN DE DATOS Y CÁLCULOS PREVIOS AL MAESTRO
    # Normalizamos nombres para evitar duplicados en el maestro (convertimos a mayúsculas y quitamos espacios)
    df_clean['nombre'] = df_clean['nombre'].str.upper().str.strip()
    #Creación de nueva columna para las métricas sobre minutos
    df_clean['minutos_float'] = df_clean['minutos'].apply(time_to_float)

    # Calculamos % de tiro por partido, con np.where para evitar división por cero si se ha tirado
    df_clean['pct_t2'] = np.where(df_clean['t2_int'] > 0, (df_clean['t2_met'] / df_clean['t2_int']) * 100, 0)
    df_clean['pct_t3'] = np.where(df_clean['t3_int'] > 0, (df_clean['t3_met'] / df_clean['t3_int']) * 100, 0)
    df_clean['pct_tl'] = np.where(df_clean['tl_int'] > 0, (df_clean['tl_met'] / df_clean['tl_int']) * 100, 0)

    #Guardamos el nuevo csv limpiado 
    df_clean.to_csv(OUTPUT_CLEAN, index=False, encoding='utf-8-sig')
    print(f"estadisticas_jugadoras_vf guardado en: {OUTPUT_CLEAN}")

    #CREACIÓN DEL MAESTRO
    #Ordenamos por temporada los datos previamente
    df_clean = df_clean.sort_values(by=['temporada', 'id_partido'], ascending=[True, True])
    # Definimos las operaciones a realizar en cada columna
    reglas = {
        # Datos descriptivos, nos quedamos con el último registrado
        'equipo': 'last',
        'liga': 'last',
        'dorsal': 'last',
        'id_partido': 'count',  # Partidos jugados
        # Medias de rendimiento
        'puntos': 'mean',
        'val': 'mean',
        'mas_menos': 'mean',
        'minutos_float': ['max', 'mean'],  

        # Medias de Tiro (Totales y Porcentajes)
        't2_met': 'mean', 't2_int': 'mean', 'pct_t2': 'mean',
        't3_met': 'mean', 't3_int': 'mean', 'pct_t3': 'mean',
        'tl_met': 'mean', 'tl_int': 'mean', 'pct_tl': 'mean',
        # Medias de Rebote
        'reb_of': 'mean', 'reb_def': 'mean', 'reb_tot': 'mean',
        # Medias de otras estadísticas
        'asist': 'mean', 'recup': 'mean', 'perd': 'mean',
        'tap_fav': 'mean', 'tap_con': 'mean', 
        'faltas_c': 'mean', 'faltas_r': 'mean'
    }

    #Agrupación por nombres
    maestro = df_clean.groupby('nombre').agg(reglas).reset_index()

    #Diferenciamos las dos columnas de minutos medios y máximos de cada jugador
    maestro.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in maestro.columns.values]
    maestro = maestro.reset_index()

    # Añadimos a los datos 'avg_' o 'total_' para que sea más comprensible
    nombres_cols = {
        'id_partido': 'total_partidos_jugados',
        'minutos_float_max': 'max_minutos_partido',
        'minutos_float_mean': 'avg_minutos_partido',
        'puntos': 'avg_puntos',
        'val': 'avg_valoracion',
        'mas_menos': 'avg_plus_minus',
        # Tiros
        't2_met': 'avg_t2_met', 't2_int': 'avg_t2_intentos', 'pct_t2': 'avg_pct_t2',
        't3_met': 'avg_t3_met', 't3_int': 'avg_t3_intentos', 'pct_t3': 'avg_pct_t3',
        'tl_met': 'avg_tl_met', 'tl_int': 'avg_tl_intentos', 'pct_tl': 'avg_pct_tl',
        # Otras estadísticas
        'reb_of': 'avg_reb_of', 'reb_def': 'avg_reb_def', 'reb_tot': 'avg_reb_tot',
        'asist': 'avg_asist', 'recup': 'avg_robos', 'perd': 'avg_perdidas',
        'tap_fav': 'avg_tap_favor', 'tap_con': 'avg_tap_contra',
        'faltas_c': 'avg_faltas_com', 'faltas_r': 'avg_faltas_rec'
    }
    maestro.rename(columns=nombres_cols, inplace=True)

    # Redondeo de decimales
    cols_numericas = maestro.select_dtypes(include=['float64']).columns
    maestro[cols_numericas] = maestro[cols_numericas].round(1)

    #Guardado del maestro
    maestro.to_csv(OUTPUT_MASTER, index=False, encoding='utf-8-sig')
    print(f" Maestro generado con {len(maestro)} jugadoras.")
    print(f" Archivo guardado en: {OUTPUT_MASTER}")

if __name__ == "__main__":
    procesar_datos()