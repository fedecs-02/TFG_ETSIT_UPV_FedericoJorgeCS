import streamlit as st # Paquete para crear la interfaz web
import pandas as pd # Para manejar dataframes
import numpy as np #Para operaciones matemáticas
import requests # Para hacer peticiones a páginas web
from bs4 import BeautifulSoup # Para leer y extraer datos del HTML
import joblib # Para cargar modelos de IA entrenados
import os # Para gestión documental y rutas
import plotly.graph_objects as go # Librerías gráficas
import plotly.express as px # Para gráficos más sencillos
import matplotlib.pyplot as plt # Para gráficos matemáticos 
import shap
# Configuración de dimensiones de la página
st.set_page_config(page_title="Simulador LF Endesa", layout="wide", initial_sidebar_state="collapsed")
# 1. Carga de Modelos
@st.cache_resource 
def cargar_modelos_ia():
    DIR_MODELOS = r"modelos_entrenados"
    modelos = {}
    try:
        modelos['clf'] = joblib.load(os.path.join(DIR_MODELOS, "clasificador.pkl"))
        modelos['reg'] = joblib.load(os.path.join(DIR_MODELOS, "regresores.pkl"))
        modelos['features'] = joblib.load(os.path.join(DIR_MODELOS, "features.pkl"))
        modelos['kmeans'] = joblib.load(os.path.join(DIR_MODELOS, "kmeans_model.pkl"))
        modelos['scaler_kmeans'] = joblib.load(os.path.join(DIR_MODELOS, "scaler_cluster.pkl"))
        return modelos
    except Exception as e:
        st.error(f"Error cargando los modelos: {e}. Comprueba la ruta.")
        return None
# Carga de la IA al iniciar la app
ia_models = cargar_modelos_ia()

# 2. Scrapping de datos de la FEB
@st.cache_data(ttl=3600)
def obtener_equipos_clasificacion(): #Extracción de nombres, logos y balances de los equipos
    url = "https://www.feb.es/lfendesa/clasificacion.aspx"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        # Selección del desplegable de jornada actual
        jornada_actual = "Jornada Desconocida"
        select_jornadas = soup.find('select', id='_ctl0_jornadasDropDownList')
        if select_jornadas:
            # Búsqueda de la opción que tiene el atributo 'selected'
            opcion_seleccionada = select_jornadas.find('option', selected=True)
            if opcion_seleccionada:
                jornada_actual = opcion_seleccionada.text.strip()     
        # Guardamdo de la jornada en la memoria de la interfaz web
        st.session_state['jornada_actual'] = jornada_actual
        #Extracción de equipos
        equipos_data = []
        filas = soup.find_all('tr')
        for fila in filas:
            celda_equipo = fila.find('td', class_='equipo')
            if celda_equipo:
                # Extracción del enlace que contiene el ID y el Nombre
                enlace = celda_equipo.find('a')
                if not enlace: continue
                nombre = enlace.text.strip()
                id_equipo = "".join(filter(str.isdigit, enlace['href'])) 
                img_tag = celda_equipo.find('img', class_='escudo') # Extracción del logo del equipo
                if img_tag and 'src' in img_tag.attrs:
                    logo = img_tag['src']
                    if logo.startswith('/'):
                        logo = "https://www.feb.es" + logo
                else:
                    logo = "https://www.feb.es/images/escudoFEB.png"
                # Extracción del balance de partidos
                columnas = fila.find_all('td')
                if len(columnas) >= 5:
                    pj = columnas[2].text.strip()
                    pg = columnas[3].text.strip()
                    pp = columnas[4].text.strip()
                    #Generación de información limpiada
                    equipos_data.append({
                        'equipo': nombre,
                        'id_equipo': id_equipo,
                        'logo': logo,
                        'PJ': pj,
                        'PG': pg,
                        'PP': pp
                    })  
        return pd.DataFrame(equipos_data)
    except Exception as e:
        st.error(f"Error al conectar con la web de la FEB: {e}")
        return pd.DataFrame(columns=['equipo', 'id_equipo', 'logo', 'PJ', 'PG', 'PP'])

# 3. Extracción estadísticas medias de equipos
@st.cache_data(ttl=3600)
def obtener_estadisticas_equipos():
    url = "https://baloncestoenvivo.feb.es/estadisticas/lfendesa/4/2025"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36"}   
    try:
        session = requests.Session()
        response = session.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        viewstate = soup.find('input', {'id': '__VIEWSTATE'})
        viewstate = viewstate['value'] if viewstate else ''
        viewstategen = soup.find('input', {'id': '__VIEWSTATEGENERATOR'})
        viewstategen = viewstategen['value'] if viewstategen else ''
        eventvalidation = soup.find('input', {'id': '__EVENTVALIDATION'})
        eventvalidation = eventvalidation['value'] if eventvalidation else ''
        payload = {
            '__EVENTTARGET': '_ctl0:MainContentPlaceHolderMaster:fasesGruposDropDownList',
            '__EVENTARGUMENT': '',
            '__VIEWSTATE': viewstate,
            '__VIEWSTATEGENERATOR': viewstategen,
            '__EVENTVALIDATION': eventvalidation,
            '_ctl0:MainContentPlaceHolderMaster:fasesGruposDropDownList': '88870'
        }
        response_post = session.post(url, headers=headers, data=payload)
        response_post.raise_for_status()
        soup_post = BeautifulSoup(response_post.content, 'html.parser')
        filas = soup_post.find_all('tr')
        datos_equipos = [] #Genera dataframe vacío
        # Extracción de información de los equipos
        for fila in filas:
            td_nombre = fila.find('td', class_='nombre equipo')
            if not td_nombre: continue
            nombre = td_nombre.text.strip()
            partidos_str = fila.find('td', class_='partidos').text.strip()
            partidos = int(partidos_str) if partidos_str.isdigit() else 1 #Partidos jugados
            # Valores medios
            def get_med(clase):
                try:
                    val = fila.find('td', class_=clase).find('span', class_='med').text
                    return float(val.replace(',', '.'))
                except: return 0.0
            # Valores totales
            def get_tiros_med(clase):
                try:
                    tot = fila.find('td', class_=clase).find('span', class_='tot').text
                    met, intt = map(int, tot.split('/'))
                    return met / partidos, intt / partidos
                except: return 0.0, 0.0
            t2_m, t2_i = get_tiros_med('tiros dos')
            t3_m, t3_i = get_tiros_med('tiros tres')
            tl_m, tl_i = get_tiros_med('tiros libres')
            reb_of = get_med('rebotes ofensivos')
            reb_def = get_med('rebotes defensivos')
            perd = get_med('perdidas')
            pts_equipo = (t2_m * 2) + (t3_m * 3) + tl_m
            reb_tot_equipo = reb_of + reb_def
            ast_equipo = get_med('asistencias')
            val_equipo = get_med('valoracion')
            # Cálculo de los 4 Factores y posesiones a excepción de ORB y DRB ya que necesitamos información del rival
            pace = t2_i + t3_i + perd + (0.44 * tl_i) - reb_of
            fga = t2_i + t3_i
            efg = 100 * (t2_m + 1.5 * t3_m) / fga if fga > 0 else 0
            tov_pct = 100 * (perd / pace) if pace > 0 else 0
            ftr = 100 * tl_m / fga if fga > 0 else 0            
            datos_equipos.append({
                'equipo': nombre,
                'PACE': round(pace, 1),
                'eFG%': round(efg, 1),
                'TOV%': round(tov_pct, 1),
                'FTR': round(ftr, 1),
                'ORB_med': round(reb_of, 1),
                'DRB_med': round(reb_def, 1),
                'PTS_med': pts_equipo,
                'REB_med': reb_tot_equipo,
                'AST_med': ast_equipo,
                'VAL_med': val_equipo
            })   
        return pd.DataFrame(datos_equipos) 
    except Exception as e:
        st.error(f"Error extrayendo estadísticas de equipos: {e}")
        return pd.DataFrame()
# Cálculo PTC
PESOS_PTC = {
    'puntos': 1.0, 'tap_fav': 0.91, 'reb_def': 0.58, 'reb_of': 0.92,
    'recup': 0.86, 'asist': 0.48, 'faltas_r': 0.23, 'tc_fail': -0.91,
    'tl_fail': -0.57, 'perd': -0.86, 'faltas_c': -0.23
}
@st.cache_data(ttl=3600)
# 4. Busca las estadísticas medias de las jugadoras
def obtener_jugadoras_sn(id_equipo, pace_propio): 
    url = f"https://baloncestoenvivo.feb.es/estadisticasacumuladas/{id_equipo}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36"}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')   
        filas = soup.find_all('tr')
        datos_jugadoras = []
        for fila in filas:
            # Busca la clase "nombre jugador"
            td_nombre = fila.find('td', class_='nombre jugador')
            if not td_nombre: continue
            nombre = td_nombre.text.strip()
            if not nombre or 'Equipos' in nombre: continue
            # Extracción de partidos jugados
            td_partidos = fila.find('td', class_='partidos')
            if not td_partidos or not td_partidos.text.strip().isdigit(): continue
            partidos = int(td_partidos.text.strip())
            if partidos == 0: continue
            # Funciones auxiliares
            def get_tot(clase): # Extracción de totales
                td = fila.find('td', class_=clase)
                if td and td.text.strip():
                    try:
                        return float(td.text.split()[0].replace(',', '.'))
                    except: return 0.0
                return 0.0
            def get_tiros_tot(clase): #Extracción de tiros intentados y tiros metidos
                td = fila.find('td', class_=clase)
                if td:
                    texto = td.text.strip().split(' ')[0]
                    if '/' in texto:
                        met, intt = map(float, texto.split('/'))
                        return met, intt
                return 0.0, 0.0
            # Cambio de minutos a formato decimal
            td_minutos = fila.find('td', class_='minutos')
            minutos_totales = 0.0
            if td_minutos:
                min_str = td_minutos.text.strip()
                if ':' in min_str:
                    m, s = min_str.split(':')
                    minutos_totales = float(m) + (float(s) / 60.0)
            if minutos_totales == 0: continue
            # Extracción de totales
            puntos_tot = get_tot('puntos')
            t2_m_tot, t2_i_tot = get_tiros_tot('tiros dos')
            t3_m_tot, t3_i_tot = get_tiros_tot('tiros tres')
            tl_m_tot, tl_i_tot = get_tiros_tot('tiros libres')
            reb_of_tot = get_tot('rebotes ofensivos')
            reb_def_tot = get_tot('rebotes defensivos')
            reb_tot_tot = get_tot('rebotes total')
            asist_tot = get_tot('asistencias')
            recup_tot = get_tot('recuperaciones')
            perd_tot = get_tot('perdidas')
            tap_fav_tot = get_tot('tapones favor')
            faltas_c_tot = get_tot('faltas cometidas')
            faltas_r_tot = get_tot('faltas recibidas')
            # Cálculo de medias
            minutos = minutos_totales / partidos
            puntos = puntos_tot / partidos
            t2_m, t2_i = t2_m_tot / partidos, t2_i_tot / partidos
            t3_m, t3_i = t3_m_tot / partidos, t3_i_tot / partidos
            tl_m, tl_i = tl_m_tot / partidos, tl_i_tot / partidos
            reb_of = reb_of_tot / partidos
            reb_def = reb_def_tot / partidos
            reb_tot = reb_tot_tot / partidos
            asist = asist_tot / partidos
            recup = recup_tot / partidos
            perd = perd_tot / partidos
            tap_fav = tap_fav_tot / partidos
            faltas_c = faltas_c_tot / partidos
            faltas_r = faltas_r_tot / partidos
            # Cálculo de métricas
            tc_f = (t2_i - t2_m) + (t3_i - t3_m)
            tl_f = (tl_i - tl_m)
            # Cálculo PTC Bruto
            ptc = (puntos * PESOS_PTC['puntos'] + tap_fav * PESOS_PTC['tap_fav'] +
                   reb_def * PESOS_PTC['reb_def'] + reb_of * PESOS_PTC['reb_of'] +
                   recup * PESOS_PTC['recup'] + asist * PESOS_PTC['asist'] +
                   faltas_r * PESOS_PTC['faltas_r'] + tc_f * PESOS_PTC['tc_fail'] +
                   tl_f * PESOS_PTC['tl_fail'] + perd * PESOS_PTC['perd'] + faltas_c * PESOS_PTC['faltas_c'])
            # Posesiones jugadas y PTC Normalizado
            posesiones = (minutos / 40) * pace_propio if pace_propio > 0 else 1
            ptc_mp = (ptc / posesiones) * 100
            # Uso y Puntos Esperados por Uso
            uso_jugadora = t2_i + t3_i + (0.44 * tl_i) + perd
            uso_jugadora = 0.1 if uso_jugadora == 0 else uso_jugadora
            pts_mpu = (puntos / (minutos * uso_jugadora)) * 40 if minutos > 0 else 0
            # eFG% y USG%
            tiros_totales = t2_i + t3_i
            efg = ((t2_m + 1.5 * t3_m) / tiros_totales) * 100 if tiros_totales > 0 else 0
            usg = (uso_jugadora / posesiones) * 100 if posesiones > 0 else 0
            # Guardado de diccionario para el modelo XGBOOST
            datos_jugadoras.append({
                'Jugadora': nombre,
                'Sn_puntos': puntos,
                'Sn_reb_tot': reb_tot,
                'Sn_asist': asist,
                'Sn_PTC': ptc,
                'Sn_PTC_mp': ptc_mp,
                'Sn_minutos_float': minutos,
                'Sn_PTS_mpu': pts_mpu,
                'Sn_eFG': efg,
                'Sn_USG': usg,
                'Sn_recup': recup,
                'Sn_tap_fav': tap_fav,
                'Sn_perd': perd
            })
        return pd.DataFrame(datos_jugadoras)
    except Exception as e:
        st.error(f"Error en la extracción de jugadoras del equipo {id_equipo}: {e}")
        return pd.DataFrame()
    
# 5. Obtención de estadísticas de las jugadoras de los últimos 6 partidos
@st.cache_data(ttl=3600)
def obtener_jugadoras_l6(id_equipo, pace_propio):
    url_racha = f"https://baloncestoenvivo.feb.es/racha/{id_equipo}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36"}
    try:
        # Obtención de enlaces a partidos
        response = requests.get(url_racha, headers=headers)
        response.raise_for_status()
        soup_racha = BeautifulSoup(response.content, 'html.parser')
        enlaces_partidos = []
        filas_racha = soup_racha.find_all('tr')
        for fila in filas_racha:
            enlace = fila.find('a', href=lambda h: h and 'Partido.aspx?p=' in h)
            if enlace and '-' in enlace.text:
                href = enlace['href']
                if href.startswith('http'):
                    url_partido = href
                else:
                    url_partido = "https://baloncestoenvivo.feb.es" + (href if href.startswith('/') else '/' + href)
                enlaces_partidos.append(url_partido)
        ultimos_6_enlaces = enlaces_partidos[-6:]
        if not ultimos_6_enlaces:
            return pd.DataFrame()
        # Extracción de estadísticas de cada partido
        datos_partidos = []
        for url_p in ultimos_6_enlaces:
            res_p = requests.get(url_p, headers=headers)
            soup_p = BeautifulSoup(res_p.content, 'html.parser')
            filas_p = soup_p.find_all('tr')
            for fila in filas_p:
                # Verificación de ID de jugadoras y equipos
                td_nombre = fila.find('td', class_='nombre jugador')
                if not td_nombre: continue
                enlace_jugadora = td_nombre.find('a', href=True)
                if not enlace_jugadora or f"i={id_equipo}" not in enlace_jugadora['href']: 
                    continue
                nombre = td_nombre.text.strip()
                if not nombre or 'Equipos' in nombre: continue
                # Extracción de minutos, excluyendo las convocadas que no jugaron
                td_min = fila.find('td', class_='minutos')
                minutos_decimales = 0.0
                if td_min and ':' in td_min.text:
                    m, s = td_min.text.split(':')
                    minutos_decimales = float(m) + (float(s) / 60.0)
                if minutos_decimales == 0: continue
                # Funciones auxiliares
                def get_val(clase):
                    td = fila.find('td', class_=clase)
                    try: return float(td.text.split()[0].replace(',', '.'))
                    except: return 0.0
                def get_tiros(clase):
                    td = fila.find('td', class_=clase)
                    if td and '/' in td.text:
                        texto = td.text.strip().split(' ')[0]
                        met, intt = map(float, texto.split('/'))
                        return met, intt
                    return 0.0, 0.0
                # Extracción de estadísticas del partido
                puntos = get_val('puntos')
                t2_m, t2_i = get_tiros('tiros dos')
                t3_m, t3_i = get_tiros('tiros tres')
                tl_m, tl_i = get_tiros('tiros libres')
                reb_of = get_val('rebotes ofensivos')
                reb_def = get_val('rebotes defensivos')
                reb_tot = get_val('rebotes total')
                asist = get_val('asistencias')
                recup = get_val('recuperaciones')
                perd = get_val('perdidas')
                tap_fav = get_val('tapones favor')
                faltas_c = get_val('faltas cometidas')
                faltas_r = get_val('faltas recibidas')
                # Cálculo de estadísticas avanzadas del encuentro
                tc_f = (t2_i - t2_m) + (t3_i - t3_m)
                tl_f = (tl_i - tl_m)
                ptc = (puntos * PESOS_PTC['puntos'] + tap_fav * PESOS_PTC['tap_fav'] +
                       reb_def * PESOS_PTC['reb_def'] + reb_of * PESOS_PTC['reb_of'] +
                       recup * PESOS_PTC['recup'] + asist * PESOS_PTC['asist'] +
                       faltas_r * PESOS_PTC['faltas_r'] + tc_f * PESOS_PTC['tc_fail'] +
                       tl_f * PESOS_PTC['tl_fail'] + perd * PESOS_PTC['perd'] + faltas_c * PESOS_PTC['faltas_c'])
                posesiones = (minutos_decimales / 40) * pace_propio if pace_propio > 0 else 1
                ptc_mp = (ptc / posesiones) * 100
                uso_j = t2_i + t3_i + (0.44 * tl_i) + perd
                uso_j = 0.1 if uso_j == 0 else uso_j
                pts_mpu = (puntos / (minutos_decimales * uso_j)) * 40 if minutos_decimales > 0 else 0
                tiros_tot = t2_i + t3_i
                efg = ((t2_m + 1.5 * t3_m) / tiros_tot) * 100 if tiros_tot > 0 else 0
                usg = (uso_j / posesiones) * 100 if posesiones > 0 else 0
                # Guardado de los datos del partido en formato fila
                datos_partidos.append({
                    'Jugadora': nombre, 'minutos_float': minutos_decimales, 'puntos': puntos,
                    'reb_tot': reb_tot, 'asist': asist,'PTC': ptc, 'PTC_mp': ptc_mp, 'PTS_mpu': pts_mpu,
                    'eFG': efg, 'USG': usg, 'recup': recup, 'tap_fav': tap_fav, 'perd': perd
                })
        # Conversión a Data Frame y cálculo de la Media Móvil Exponencial
        df_partidos = pd.DataFrame(datos_partidos)
        if df_partidos.empty: return pd.DataFrame()
        # Definición de variables necesarias para la Media Móvil Exponencial
        cols_a_ema = ['minutos_float', 'puntos', 'reb_tot', 'asist','PTC', 'PTC_mp', 'PTS_mpu', 'eFG', 'USG', 'recup', 'tap_fav', 'perd']
        # Cálculo de la Media Móvil Exponencial para cada jugadora de los últimos 6 partidos
        for col in cols_a_ema:
            df_partidos[f'L6_{col}'] = df_partidos.groupby('Jugadora')[col].transform(lambda x: x.ewm(span=6, min_periods=1).mean())
            if col in ['PTC_mp', 'puntos']:
                df_partidos[f'L6_{col}_std'] = df_partidos.groupby('Jugadora')[col].transform(lambda x: x.rolling(6, min_periods=2).std().fillna(0))
                df_partidos[f'L6_{col}_min'] = df_partidos.groupby('Jugadora')[col].transform(lambda x: x.rolling(6, min_periods=1).min().fillna(0))
                
        df_l6_final = df_partidos.drop_duplicates(subset=['Jugadora'], keep='last').reset_index(drop=True)

        columnas_finales = ['Jugadora'] + [f'L6_{col}' for col in cols_a_ema] + ['L6_PTC_mp_std', 'L6_PTC_mp_min', 'L6_puntos_std', 'L6_puntos_min']
        return df_l6_final[columnas_finales]
    except Exception as e:
        st.error(f"Error al procesar racha del equipo {id_equipo}: {e}")
        return pd.DataFrame()
# 6. Configuración del motor de la IA
def predecir_rendimiento(df_unificado, modelos_ia, cluster_rival, dist_centroide_rival, es_local, pace_rival):
    if df_unificado.empty or modelos_ia is None:
        return pd.DataFrame()
    df_pred = df_unificado.copy()
    df_pred_ia = df_unificado.copy()
    col_min = 'Minutos_Asignados' if 'Minutos_Asignados' in df_pred.columns else 'Sn_minutos_float'
    mins_hoy = df_pred_ia[col_min].astype(float)
    mins_hist = df_pred_ia['Sn_minutos_float'].astype(float).replace(0, 0.1)
    ratio_mins = mins_hoy / mins_hist
    vars_volumen = ['Sn_puntos', 'Sn_reb_tot', 'Sn_asist', 'Sn_recup', 'Sn_tap_fav', 'Sn_perd',
                        'L6_puntos', 'L6_reb_tot', 'L6_asist', 'L6_recup', 'L6_tap_fav', 'L6_perd']
        
    for v in vars_volumen:
        if v in df_pred_ia.columns:
            df_pred_ia[v] = df_pred_ia[v].astype(float) * ratio_mins
    df_pred_ia['Sn_minutos_float'] = mins_hoy
    if 'L6_minutos_float' in df_pred_ia.columns:
        df_pred_ia['L6_minutos_float'] = mins_hoy
    df_pred_ia['minutos_float'] = mins_hoy
    # Incluimos las variables del partido de hoy al modelo
    df_pred_ia['es_local'] = es_local
    df_pred_ia['pace_rival'] = pace_rival if (pace_rival and pace_rival > 0) else 70.0
    df_pred_ia['dist_centroide_rival'] = dist_centroide_rival
    # Seleccionamos únicamente las variabes con las que fue entrenado el modelo en origen
    features = modelos_ia['features']
    #Eliminamos clusteres viejos:
    for col in df_pred_ia.columns:
        if 'cluster_def_rival_' in col:
            df_pred_ia[col] = 0.0
    # Rellenamos las columnas con 0 si no hay nada        
    for col in features:
        if col not in df_pred_ia.columns: 
            df_pred_ia[col] = 0.0
    col_cluster = f'cluster_def_rival_{int(cluster_rival)}'
    if col_cluster in df_pred_ia.columns:
        df_pred_ia[col_cluster] = 1.0       
    X_pred = df_pred_ia[features]
    # Cálculo del modelo clasificador de XGBoost que nos dará el % de probabilidad de éxito de la jugadora
    probs = modelos_ia['clf'].predict_proba(X_pred)[:, 1]
    proyecciones = {t: modelos_ia['reg'][t].predict(X_pred) for t in modelos_ia['reg']} # Para todas las proyecciones del XGBoost
    # Generamos las variables de salida para el modelo regresor de XGBoost
    puntos_f, rebotes_f, asist_f, ptc_f, usg_f, efg_f, recup_f = [], [], [], [], [], [], []
    tap_f, perd_f, falt_c_f, falt_r_f = [], [], [], []
    t2i_f, t2m_f, t3i_f, t3m_f, tli_f, tlm_f = [], [], [], [], [], []
    reb_of_f, reb_def_f = [], []
    for i, (idx, row) in enumerate(df_pred.iterrows()):
        m_hoy = float(row.get(col_min, 0.0))
        if m_hoy == 0:
            puntos_f.append(0.0); rebotes_f.append(0.0); asist_f.append(0.0); ptc_f.append(0.0)
            usg_f.append(0.0); efg_f.append(0.0); recup_f.append(0.0); tap_f.append(0.0)
            perd_f.append(0.0); falt_c_f.append(0.0); falt_r_f.append(0.0)
            t2i_f.append(0.0); t2m_f.append(0.0); t3i_f.append(0.0); t3m_f.append(0.0)
            tli_f.append(0.0); tlm_f.append(0.0); reb_of_f.append(0.0); reb_def_f.append(0.0)
            continue
        # Cálculo PTC   
        p = {t: proyecciones[t][i] for t in proyecciones}
        fallos_t2 = max(0, p['t2_int'] - p['t2_met'])
        fallos_t3 = max(0, p['t3_int'] - p['t3_met'])
        tc_fail = fallos_t2 + fallos_t3
        tl_fail = max(0, p['tl_int'] - p['tl_met'])
        valor_ptc = (
            p['puntos'] * 1.0 + p['tap_fav'] * 0.91 + p['reb_def'] * 0.58 + 
            p['reb_of'] * 0.92 + p['recup'] * 0.86 + p['asist'] * 0.48 + 
            p['faltas_r'] * 0.23 + tc_fail * -0.91 + tl_fail * -0.57 + 
            p['perd'] * -0.86 + p['faltas_c'] * -0.23
        )
        puntos_f.append(max(0, p['puntos'])); rebotes_f.append(max(0, p['reb_of'] + p['reb_def']))
        asist_f.append(max(0, p['asist'])); ptc_f.append(valor_ptc)
        usg_f.append(max(0, p['USG'])); efg_f.append(max(0, p['eFG']))   
        recup_f.append(max(0, p['recup'])); tap_f.append(max(0, p['tap_fav']))
        perd_f.append(max(0, p['perd'])); falt_c_f.append(max(0, p['faltas_c']))
        falt_r_f.append(max(0, p['faltas_r'])); 
        t2i_f.append(max(0, p['t2_int'])); t2m_f.append(max(0, p['t2_met']))
        t3i_f.append(max(0, p['t3_int'])); t3m_f.append(max(0, p['t3_met']))
        tli_f.append(max(0, p['tl_int'])); tlm_f.append(max(0, p['tl_met']))
        reb_of_f.append(max(0, p['reb_of'])); reb_def_f.append(max(0, p['reb_def']))
    # Guardado en el dataset
    df_pred['Puntos_IA'] = np.round(puntos_f, 1); df_pred['Rebotes_IA'] = np.round(rebotes_f, 1)
    df_pred['Asistencias_IA'] = np.round(asist_f, 1); df_pred['PTC_Proy'] = np.round(ptc_f, 1)
    df_pred['USG_IA'] = np.round(usg_f, 1); df_pred['eFG_IA'] = np.round(efg_f, 1)
    df_pred['recup_IA'] = np.round(recup_f, 1); df_pred['tap_IA'] = np.round(tap_f, 1)
    df_pred['perd_IA'] = np.round(perd_f, 1); df_pred['faltas_c_IA'] = np.round(falt_c_f, 1)
    df_pred['faltas_r_IA'] = np.round(falt_r_f, 1)
    df_pred['t2i_IA'] = np.round(t2i_f, 1); df_pred['t2m_IA'] = np.round(t2m_f, 1)
    df_pred['t3i_IA'] = np.round(t3i_f, 1); df_pred['t3m_IA'] = np.round(t3m_f, 1)
    df_pred['tli_IA'] = np.round(tli_f, 1); df_pred['tlm_IA'] = np.round(tlm_f, 1)
    df_pred['reb_of_IA'] = np.round(reb_of_f, 1); df_pred['reb_def_IA'] = np.round(reb_def_f, 1)
    # Semáforo
    def semaforo(pr, mins):
        if mins == 0: return "No juega"
        if pr >= 0.60: return f"Alta ({pr*100:.0f}%)"
        if pr >= 0.35: return f"Dudosa ({pr*100:.0f}%)"
        return f"Baja ({pr*100:.0f}%)"
    df_pred['Prob_Exito'] = [semaforo(pr, m) for pr, m in zip(probs, df_pred[col_min])]
    df_pred['Prob_Num'] = np.round(probs * 100, 1)
    cols_finales = ['Jugadora', 'Puntos_IA', 'Rebotes_IA', 'Asistencias_IA', 'PTC_Proy', 'USG_IA', 'eFG_IA', 
                    'recup_IA', 'tap_IA', 'perd_IA', 'faltas_c_IA', 'faltas_r_IA', 
                    't2i_IA', 't2m_IA', 't3i_IA', 't3m_IA', 'tli_IA', 'tlm_IA', 'reb_of_IA', 'reb_def_IA',
                    'Prob_Exito', 'Prob_Num', 'L6_PTC', 'Sn_PTC', 'Sn_minutos_float', 'Minutos_Asignados',
                    'Sn_puntos', 'Sn_reb_tot', 'Sn_asist']
                    
    return df_pred[[c for c in cols_finales if c in df_pred.columns]]

# 6.2. Optimización de minutos
def optimizar_minutos_plantilla(df, convocadas=None):
    df_opt = df.copy()
    # Cálculo de la eficiencia por minuto
    mins_seguros = df_opt['Minutos_Asignados'].replace(0, 0.1)
    eficiencia_ia = df_opt['PTC_Proy'] / mins_seguros
    # Filtro de aquellas jugadoras con eficiencia negativa
    pesos = eficiencia_ia.clip(lower=0).to_numpy(copy=True)    
    if convocadas is not None:
        for i, jugadora in enumerate(df_opt['Jugadora']):
            if jugadora not in convocadas:
                pesos[i] = 0
    # Uso de los minutos reales en caso de que la mayoría tengan eficiencia negativa
    if np.sum(pesos) == 0:
        return df_opt['Sn_minutos_float'].values  
    n_jugadoras = len(df_opt)
    minutos_asignados = np.zeros(n_jugadoras)
    minutos_restantes = 200.0
    tope_minutos = 35.0  # Límite máximo 
    # Bucle de reparto de minutos
    while minutos_restantes > 0.1 and np.sum(pesos) > 0:
        suma_pesos = np.sum(pesos)
        cuotas = (pesos / suma_pesos) * minutos_restantes
        for i in range(n_jugadoras):
            if pesos[i] > 0:
                # Tope de las jugadoras que han llego a 35 mins para que ya no le sumen
                if minutos_asignados[i] + cuotas[i] > tope_minutos:
                    minutos_restantes -= (tope_minutos - minutos_asignados[i])
                    minutos_asignados[i] = tope_minutos
                    pesos[i] = 0
                else:
                    minutos_asignados[i] += cuotas[i]
                    minutos_restantes -= cuotas[i]
                    
    # Redondeo a un decimal 
    return np.round(minutos_asignados, 1)


# 7. Interfaz gráfica
# 7.1. Cabezera
col_logo, col_tit = st.columns([1, 6])
with col_logo:
    st.image("https://www.ibaetabasket.com/wp-content/uploads/2020/11/LF-ENDESA_horizontal_fondo-blanco.png", width=400)
with col_tit:
    st.title("Simulador de Partidos")
    if 'jornada_actual' in st.session_state:
        st.caption(f"{st.session_state['jornada_actual']}")
st.markdown('<hr style="border: none; height: 5px; background-color: #004080; margin-top: -10px; margin-bottom: 25px;">', unsafe_allow_html=True)
# 7.2. Desplegables de selección de equipos
df_equipos = obtener_equipos_clasificacion()
lista_equipos = df_equipos['equipo'].tolist()
c_loc, c_vis, c_btn = st.columns([2, 2, 1.5])
with c_loc:
    equipo_local = st.selectbox("Equipo Local:", lista_equipos, index=0)
with c_vis:
    equipo_visitante = st.selectbox("Equipo Visitante:", lista_equipos, index=1)
with c_btn:
    st.write("") 
    st.write("") 
    # Creación de botón de simular partido
    btn_simular = st.button("SIMULAR PARTIDO", type="primary")
# 7.3. Fusión de medias globales y de los últimos 6 partidos
def preparar_dataset_unificado(df_sn, df_l6):
    if df_sn.empty: return pd.DataFrame()
    df = pd.merge(df_sn, df_l6, on='Jugadora', how='left')
    # Rellenado de datos de racha por si no hubiera (Ej: Lesión)
    for c in df.columns:
        if c.startswith('L6_'):
            # Si no hay datos en los últimos 6 partidos, su racha es CERO
            df[c] = df[c].fillna(0.0)
            
    df = df.fillna(0)
    return df
# 7.4. Lógica del botón de simulación
if btn_simular:
    if equipo_local == equipo_visitante:
        st.error("El equipo local y visitante no pueden ser el mismo")
    else:
        with st.spinner("Descargando estadísticas y aplicando modelos predictivos"):
            # scrapping de datos
            datos_local_basicos = df_equipos[df_equipos['equipo'] == equipo_local].iloc[0]
            datos_visit_basicos = df_equipos[df_equipos['equipo'] == equipo_visitante].iloc[0]
            df_stats_avanzadas = obtener_estadisticas_equipos()
            if df_stats_avanzadas.empty:
                st.error("Error: No se pudieron descargar las estadísticas")
                st.stop() # Detiene la ejecución aquí para no generar más errores
            match_local = df_stats_avanzadas[df_stats_avanzadas['equipo'].str.contains(equipo_local[:10], case=False, na=False)]
            if match_local.empty:
                st.error(f"Error: No se encontraron las estadísticas del equipo local ({equipo_local}). Puede que el nombre difiera entre páginas.")
                st.stop()
            stats_local = match_local.iloc[0]
            match_visitante = df_stats_avanzadas[df_stats_avanzadas['equipo'].str.contains(equipo_visitante[:10], case=False, na=False)]
            if match_visitante.empty:
                st.error(f"Error: No se encontraron las estadísticas del equipo visitante ({equipo_visitante}). Puede que el nombre difiera entre páginas.")
                st.stop()
            stats_visit = match_visitante.iloc[0]
            df_sn_local = obtener_jugadoras_sn(datos_local_basicos['id_equipo'], stats_local['PACE'])
            df_sn_visit = obtener_jugadoras_sn(datos_visit_basicos['id_equipo'], stats_visit['PACE'])
            
            df_l6_local = obtener_jugadoras_l6(datos_local_basicos['id_equipo'], stats_local['PACE'])
            df_l6_visit = obtener_jugadoras_l6(datos_visit_basicos['id_equipo'], stats_visit['PACE'])
            uni_loc = preparar_dataset_unificado(df_sn_local, df_l6_local)
            uni_vis = preparar_dataset_unificado(df_sn_visit, df_l6_visit)
            # Guardado en memoria de los datos extraídos
            st.session_state['datos_simulacion'] = {
                'equipo_local': equipo_local, 'equipo_visitante': equipo_visitante,
                'basicos_local': datos_local_basicos, 'basicos_visit': datos_visit_basicos,
                'stats_local': stats_local, 'stats_visit': stats_visit,
                'df_sn_local': df_sn_local, 'df_sn_visit': df_sn_visit,
                'df_l6_local': df_l6_local, 'df_l6_visit': df_l6_visit,
                'uni_loc': uni_loc, 
                'uni_vis': uni_vis
            }
# 8. Interfaz de resultado
if 'datos_simulacion' in st.session_state:
    d = st.session_state['datos_simulacion']
    st.markdown("---")
    # Asignación de cluster
    def predecir_cluster(stats_equipo, stats_rival, modelos_ia):
        if modelos_ia is None or 'kmeans' not in modelos_ia or 'scaler_kmeans' not in modelos_ia:
            return -1, 0.0
        try:
            orb = round(100 * stats_equipo['ORB_med'] / (stats_equipo['ORB_med'] + stats_rival['DRB_med']),1)
            drb = round(100 * stats_equipo['DRB_med'] / (stats_equipo['DRB_med'] + stats_rival['ORB_med']),1)
            # Diccionario de variables
            datos_modelo = {
                'EFG': stats_equipo['eFG%'],
                'TOV': stats_equipo['TOV%'], 
                'ORB': orb,
                'DRB': drb,
                'FT/FGA': stats_equipo['FTR'],
                'PACE': stats_equipo['PACE']
            }
            df_temp = pd.DataFrame([datos_modelo])
            orden_correcto = modelos_ia['scaler_kmeans'].feature_names_in_
            df_temp = df_temp[orden_correcto]
            datos_escalados = modelos_ia['scaler_kmeans'].transform(df_temp)
            num_cluster = modelos_ia['kmeans'].predict(datos_escalados)[0]
            distancias = modelos_ia['kmeans'].transform(datos_escalados)
            distancia_centroide = distancias[0, num_cluster]
            return int(num_cluster), float(distancia_centroide)
        except Exception as e:
            return -1, 0.0
    num_cluster_local, dist_local = predecir_cluster(d['stats_local'], d['stats_visit'], ia_models)
    num_cluster_visit, dist_visit = predecir_cluster(d['stats_visit'], d['stats_local'], ia_models)
    # Asignación de Cluster asignado
    NOMBRES_CLUSTERS = {
        0: "Acelerado y Volátil",
        1: "Estructurado y Controlador",
        2: "Transicional y Arriesgado",
        3: "Físico y Posicional",
        4: "Equilibrado y Regular",
        5: "Eficiente y Sistemático",
        -1: "Desconocido"
    }
    cluster_local_txt = NOMBRES_CLUSTERS.get(num_cluster_local, f"Cluster {num_cluster_local}")
    cluster_visit_txt = NOMBRES_CLUSTERS.get(num_cluster_visit, f"Cluster {num_cluster_visit}")
    #Bloque de Clusters y información básica de los equipos
    top_izq, separador, top_der = st.columns([4, 0.5, 4])
    with top_izq:
        st.subheader(f"🏠 {d['equipo_local']}")
        c1, c2 = st.columns([1, 2])
        with c1: st.image(d['basicos_local']['logo'], width=100)
        with c2:
            st.metric("Balance de Partidos", f"{d['basicos_local']['PJ']} | {d['basicos_local']['PG']} V - {d['basicos_local']['PP']} D")
            orb_loc_pct = round(100 * d['stats_local']['ORB_med'] / (d['stats_local']['ORB_med'] + d['stats_visit']['DRB_med']),1)
            drb_loc_pct = round(100 * d['stats_local']['DRB_med'] / (d['stats_local']['DRB_med'] + d['stats_visit']['ORB_med']),1)
            f1, f2, f3 = st.columns(3)
            with f1: st.info(f"**eFG%:** {d['stats_local']['eFG%']}%")
            with f2: st.info(f"**TOV%:** {d['stats_local']['TOV%']}%")
            with f3: st.info(f"**FT/FGA:** {d['stats_local']['FTR']}%")
            f4, f5, f6 = st.columns(3)
            with f4: st.success(f"**ORB%:** {orb_loc_pct}%")
            with f5: st.success(f"**DRB%:** {drb_loc_pct}%")
            with f6: st.success(f"**PACE:** {d['stats_local']['PACE']}")
            st.warning(f"**Estilo de Juego:** {cluster_local_txt}")
    with separador:
        st.markdown("<h1 style='text-align: center; margin-top: 50px;'>VS</h1>", unsafe_allow_html=True)
    with top_der:
        st.subheader(f"✈️ {d['equipo_visitante']}")
        c1, c2 = st.columns([1, 2])
        with c1: st.image(d['basicos_visit']['logo'], width=100)
        with c2:
            st.metric("Balance de Partidos", f"{d['basicos_visit']['PJ']} | {d['basicos_visit']['PG']} V - {d['basicos_visit']['PP']} D")
            orb_vis_pct = round(100 * d['stats_visit']['ORB_med'] / (d['stats_visit']['ORB_med'] + d['stats_local']['DRB_med']),1)
            drb_vis_pct = round(100 * d['stats_visit']['DRB_med'] / (d['stats_visit']['DRB_med'] + d['stats_local']['ORB_med']),1)
            f1, f2, f3 = st.columns(3)
            with f1: st.info(f"**eFG%:** {d['stats_visit']['eFG%']}%")
            with f2: st.info(f"**TOV%:** {d['stats_visit']['TOV%']}%")
            with f3: st.info(f"**FT/FGA:** {d['stats_visit']['FTR']}%")
            f4, f5, f6 = st.columns(3)
            with f4: st.success(f"**ORB%:** {orb_vis_pct}%")
            with f5: st.success(f"**DRB%:** {drb_vis_pct}%")
            with f6: st.success(f"**PACE:** {d['stats_visit']['PACE']}")
        
            st.warning(f"**Estilo de Juego:** {cluster_visit_txt}")
    st.markdown("---")
    # 9. Activación de modelos de IA
    if ia_models:
        import datetime # librería de tiempo
        # Asignación de minutos de los slicers
        def inyectar_minutos_sliders(df_base, df_target, equipo, col_minutos_defecto):
            df_copia = df_target.copy()
            minutos_asignados = []
            dict_minutos = {}
            for i, row in df_base.iterrows():
                key_slider = f"slider_{equipo}_{i}"
                if key_slider in st.session_state:
                    val = st.session_state[key_slider]
                    if isinstance(val, datetime.time):
                        val = val.minute + val.second / 60.0
                    dict_minutos[row['Jugadora']] = val
                else:
                    dict_minutos[row['Jugadora']] = row['Sn_minutos_float'] 
            for _, row in df_copia.iterrows():
                jugadora = row['Jugadora']
                minutos = dict_minutos.get(jugadora, row.get(col_minutos_defecto, 0.0))
                minutos_asignados.append(minutos)
            df_copia['Minutos_Asignados'] = minutos_asignados
            return df_copia
        # Aplicamos los mi utos seleccionados en los slicers
        d['uni_loc'] = inyectar_minutos_sliders(d['uni_loc'], d['uni_loc'], 'loc', 'Sn_minutos_float')
        d['uni_vis'] = inyectar_minutos_sliders(d['uni_vis'], d['uni_vis'], 'vis', 'Sn_minutos_float')
        # Ejecución del modelo de XGBoost para ambos equipos
        pred_loc = predecir_rendimiento(d['uni_loc'], ia_models, num_cluster_visit, dist_visit, 1, d['stats_visit']['PACE'])
        pred_vis = predecir_rendimiento(d['uni_vis'], ia_models, num_cluster_local, dist_local, 0, d['stats_local']['PACE'])
    else:
        df_ia_sn_loc = df_ia_l6_loc = df_ia_sn_vis = df_ia_l6_vis = pd.DataFrame()
    # 10. Bloque de visualización de estadísticas
    st.markdown("### Estadísticas generales")
    c_filtro, c_vacia1, c_vacia2 = st.columns([1, 2, 1])
    with c_filtro:
        stat_seleccionada = st.selectbox(
            "Métrica de comparación:",
            ['PTC (Impacto Total)', 'Puntos', 'Rebotes', 'Asistencias']
        )
    # Diccionario de variables
    dic_stats = {
        'PTC (Impacto Total)': ('Sn_PTC', '#e74c3c', 'PTC'), 
        'Puntos': ('Sn_puntos', '#3498db', 'PTS'),             
        'Rebotes': ('Sn_reb_tot', '#2ecc71', 'REB'),           
        'Asistencias': ('Sn_asist', '#f1c40f', 'AST')           
    }
    col_stat, color_stat, nombre_stat = dic_stats[stat_seleccionada]
    st.write("") 
    col_graf_izq, col_graf_cen, col_graf_der = st.columns([1.5, 1.2, 1.5])
    # Equipo Local
    with col_graf_izq:
        st.markdown(f"<h5 style='text-align: center; color: #1f77b4;'>Top 8 {nombre_stat} - {d['equipo_local']}</h5>", unsafe_allow_html=True)
        if not d['df_sn_local'].empty:
            df_rot_loc = d['df_sn_local'].nlargest(8, col_stat).sort_values(col_stat, ascending=True)
            
            fig_loc = px.bar(df_rot_loc, x=col_stat, y='Jugadora', orientation='h', 
                             text_auto='.1f', color_discrete_sequence=[color_stat])
            max_val = df_rot_loc[col_stat].max()
            fig_loc.update_layout(
                margin=dict(l=0, r=0, t=10, b=0), height=350,
                xaxis=dict(range=[0, max_val * 1.15], showgrid=False, showticklabels=False, title=None),
                yaxis=dict(title=None, tickfont=dict(size=11))
            )
            
            fig_loc.update_traces(textposition='outside', textfont_size=11)
            st.plotly_chart(fig_loc, use_container_width=True, config={'displayModeBar': False})
    # Gráfico Spider Comparativo
    with col_graf_cen:
        st.markdown("<h5 style='text-align: center;'>Choque de Estilos</h5>", unsafe_allow_html=True)
        orb_loc_pct = round(100 * d['stats_local']['ORB_med'] / (d['stats_local']['ORB_med'] + d['stats_visit']['DRB_med']), 1)
        drb_loc_pct = round(100 * d['stats_local']['DRB_med'] / (d['stats_local']['DRB_med'] + d['stats_visit']['ORB_med']), 1)
        orb_vis_pct = round(100 * d['stats_visit']['ORB_med'] / (d['stats_visit']['ORB_med'] + d['stats_local']['DRB_med']), 1)
        drb_vis_pct = round(100 * d['stats_visit']['DRB_med'] / (d['stats_visit']['DRB_med'] + d['stats_local']['ORB_med']), 1)
        categorias = ['eFG%', 'TOV%', 'FT/FGA', 'ORB%', 'DRB%']
        v_loc_raw = [d['stats_local']['eFG%'], d['stats_local']['TOV%'], d['stats_local']['FTR'], orb_loc_pct, drb_loc_pct]
        v_vis_raw = [d['stats_visit']['eFG%'], d['stats_visit']['TOV%'], d['stats_visit']['FTR'], orb_vis_pct, drb_vis_pct]
        v_max = [max(l, v) if max(l, v) > 0 else 1 for l, v in zip(v_loc_raw, v_vis_raw)]
        r_loc = [(l / m) * 100 for l, m in zip(v_loc_raw, v_max)]
        r_vis = [(v / m) * 100 for v, m in zip(v_vis_raw, v_max)]
        r_loc += [r_loc[0]]; r_vis += [r_vis[0]]
        v_loc_raw += [v_loc_raw[0]]; v_vis_raw += [v_vis_raw[0]]
        cat_cerradas = categorias + [categorias[0]]
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=r_loc, theta=cat_cerradas, fill='toself', name='Local',
            line_color='#1f77b4', fillcolor='rgba(31, 119, 180, 0.4)',
            customdata=v_loc_raw, hovertemplate="%{theta}: <b>%{customdata}</b><extra></extra>"
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=r_vis, theta=cat_cerradas, fill='toself', name='Visitante',
            line_color='#ff7f0e', fillcolor='rgba(255, 127, 14, 0.4)',
            customdata=v_vis_raw, hovertemplate="%{theta}: <b>%{customdata}</b><extra></extra>"
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=False, range=[0, 110]), 
                angularaxis=dict(tickfont=dict(size=11))
            ),
            showlegend=True, 
            legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5, title=None), # Arriba y centrada
            height=370, margin=dict(t=30, b=20, l=40, r=40)
        )
        st.plotly_chart(fig_radar, use_container_width=True, config={'displayModeBar': False})
    # Equipo visitante
    with col_graf_der:
        st.markdown(f"<h5 style='text-align: center; color: #ff7f0e;'>Top 8 {nombre_stat} - {d['equipo_visitante']}</h5>", unsafe_allow_html=True)
        if not d['df_sn_visit'].empty:
            df_rot_vis = d['df_sn_visit'].nlargest(8, col_stat).sort_values(col_stat, ascending=True)
            
            fig_vis = px.bar(df_rot_vis, x=col_stat, y='Jugadora', orientation='h', 
                             text_auto='.1f', color_discrete_sequence=[color_stat])
            
            max_val_vis = df_rot_vis[col_stat].max()
            fig_vis.update_layout(
                margin=dict(l=0, r=0, t=10, b=0), height=350,
                xaxis=dict(range=[0, max_val_vis * 1.15], showgrid=False, showticklabels=False, title=None), 
                yaxis=dict(title=None, tickfont=dict(size=11), side='right') # Simetría espejo
            )
            
            fig_vis.update_traces(textposition='outside', textfont_size=11)
            st.plotly_chart(fig_vis, use_container_width=True, config={'displayModeBar': False})

    st.markdown("---")
    # 11. Bloque de Jugadoras en Racha
    st.markdown("### Jugadoras en Racha")
    clocal,cvisit = st.columns(2)
    with clocal:
        st.markdown(f"#### {d['equipo_local']}")
        top_loc = pred_loc.sort_values('L6_PTC', ascending=False).head(3)
        c1_loc, c2_loc, c3_loc = st.columns(3)
        columnas_loc = [c1_loc, c2_loc, c3_loc]
        for i, (_, r) in enumerate(top_loc.iterrows()):
            with columnas_loc[i]:
                st.success(f"**{r['Jugadora']}**\n\n"
                        f"Racha (L6): **{r['L6_PTC']:.1f}**\n\n"
                        f"PTC Proy: **{r['PTC_Proy']:.1f}**\n\n"
                        f"Prob. Éxito: **{r['Prob_Exito']}**\n\n"
                        f"PTS: **{r['Puntos_IA']:.1f}** | AST: **{r['Asistencias_IA']:.1f}**")
    with cvisit:
        st.markdown(f"#### {d['equipo_visitante']}")
        top_vis = pred_vis.sort_values('L6_PTC', ascending=False).head(3)
        c1_vis, c2_vis, c3_vis = st.columns(3)
        columnas_vis = [c1_vis, c2_vis, c3_vis]
        for i, (_, r) in enumerate(top_vis.iterrows()):
            with columnas_vis[i]:
                st.warning(f"**{r['Jugadora']}**\n\n"
                        f"Racha (L6): **{r['L6_PTC']:.1f}**\n\n"
                        f"PTC Proy: **{r['PTC_Proy']:.1f}**\n\n"
                        f"Prob. Éxito: **{r['Prob_Exito']}**\n\n"
                        f"PTS: **{r['Puntos_IA']:.1f}** | AST: **{r['Asistencias_IA']:.1f}**")
    st.markdown("---")
    
    # 12. Bloque de simulación de rotaciones
    st.markdown("### Ajuste de Rotaciones")
    equipo_a_rotar = st.radio("Equipo:", [d['equipo_local'], d['equipo_visitante']], horizontal=True) 
    # Conexión con memoria y racha de jugadoras
    prefijo_memoria = 'loc' if equipo_a_rotar == d['equipo_local'] else 'vis'
    df_base_sim = d['uni_loc'].copy() if equipo_a_rotar == d['equipo_local'] else d['uni_vis'].copy()
    df_base_sim['Sn_minutos_float'] = df_base_sim['Sn_minutos_float'].round(1)
    if not df_base_sim.empty and ia_models:
        col_sliders, col_resultados = st.columns([1, 1])
        with col_sliders:
                import datetime
                st.markdown("#### Rotaciones:")
                def float_to_time(f_mins):
                    m = int(f_mins)
                    s = int(round((f_mins - m) * 60))
                    if s >= 60:
                        m += 1
                        s -= 60
                    m = min(m, 40)
                    return datetime.time(0, m, s)
                todas_jugadoras = df_base_sim['Jugadora'].tolist()
                top_12_default = df_base_sim.sort_values('Sn_minutos_float', ascending=False).head(12)['Jugadora'].tolist()
                convocadas = st.multiselect(
                    "Selecciona las jugadoras convocadas:",
                    options=todas_jugadoras,
                    default=top_12_default,
                    max_selections=12
                )
                is_local_scout = (equipo_a_rotar == d['equipo_local'])
                cluster_rival = num_cluster_visit if is_local_scout else num_cluster_local
                dist_rival = dist_visit if is_local_scout else dist_local
                pace_rival = d['stats_visit']['PACE'] if is_local_scout else d['stats_local']['PACE']
                factor_cancha = 1 if is_local_scout else 0
                df_scout = df_base_sim.copy()
                df_scout['Minutos_Asignados'] = df_scout['Sn_minutos_float']
                df_pred_scout = predecir_rendimiento(df_scout, ia_models, cluster_rival, dist_rival, factor_cancha, pace_rival)
                df_base_sim['Minutos_Optimos'] = optimizar_minutos_plantilla(df_pred_scout, convocadas) 
                if st.button("Calcular Rotación Óptima", use_container_width=True):
                    for i, row in df_base_sim.iterrows():
                        st.session_state[f"slider_{prefijo_memoria}_{i}"] = float_to_time(float(row['Minutos_Optimos']))
                st.markdown("<br>", unsafe_allow_html=True)

                df_base_sim['Minutos_Asignados'] = df_base_sim['Minutos_Optimos']
                for i, row in df_base_sim.iterrows():
                    jugadora = row['Jugadora']
                    es_convocada = jugadora in convocadas 
                    val_optimo = float(row['Minutos_Optimos']) 
                    valor_actual = st.session_state.get(f"slider_{prefijo_memoria}_{i}")
                    if not es_convocada:
                        valor_actual = float_to_time(0.0)
                    elif isinstance(valor_actual, float) or isinstance(valor_actual, int):
                        valor_actual = float_to_time(valor_actual)
                    elif valor_actual is None:
                        valor_actual = float_to_time(val_optimo) 
                    nuevo_tiempo = st.slider(
                        jugadora,
                        min_value=datetime.time(0, 0),
                        max_value=datetime.time(0, 40),
                        value=valor_actual,
                        step=datetime.timedelta(seconds=1), 
                        format="mm:ss",
                        key=f"slider_{prefijo_memoria}_{i}",
                        disabled=not es_convocada # LA MAGIA: Bloquea el slider si no juega
                    )
                    nuevo_minuto_float = nuevo_tiempo.minute + nuevo_tiempo.second / 60.0
                    df_base_sim.at[i, 'Minutos_Asignados'] = nuevo_minuto_float
                # Advertencia de minutos totales para ajustar bien las rotaciones
                minutos_totales = df_base_sim['Minutos_Asignados'].sum()
                mt_m = int(minutos_totales)
                mt_s = int(round((minutos_totales - mt_m) * 60))
                if mt_s == 60: mt_m += 1; mt_s = 0
                if minutos_totales > 201.0 or minutos_totales < 199.0:
                    st.warning(f"{mt_m:02d}:{mt_s:02d} minutos asignados. Deben ser 200:00 ")
                else:
                    st.info(f"Minutos repartidos: {mt_m:02d}:{mt_s:02d} / 200:00 minutos posibles.")
        with col_resultados:
            st.markdown("#### Proyección de rendimiento individual")
            # Asignación de variable de contexto
            is_local = (equipo_a_rotar == d['equipo_local'])
            cluster_rival = num_cluster_visit if is_local else num_cluster_local
            dist_rival = dist_visit if is_local else dist_local
            pace_rival = d['stats_visit']['PACE'] if is_local else d['stats_local']['PACE']
            factor_cancha = 1 if is_local else 0
            # Cálculo de la predicción con los minutos asignados en los sliders
            df_sim_ia = predecir_rendimiento(df_base_sim, ia_models, cluster_rival, dist_rival, factor_cancha, pace_rival)
            # Cálculo de la predicción base con minutos medios de las jugadoras
            df_reset = df_base_sim.copy()
            df_reset['Minutos_Asignados'] = df_reset['Minutos_Optimos']
            df_base_ia = predecir_rendimiento(df_reset, ia_models, cluster_rival, dist_rival, factor_cancha, pace_rival)
            if not df_sim_ia.empty and not df_base_ia.empty:
                df_comparacion = pd.merge(df_sim_ia, df_base_ia, on='Jugadora', suffixes=('_proy', '_base'), how='left')
                df_comparacion = df_comparacion.sort_values(by='Puntos_IA_proy', ascending=False) 
                # Configuración delta comparativo con valores medios
                def formato_delta(nuevo, antiguo):
                    if pd.isna(antiguo): return f"{nuevo:.1f}"
                    delta = nuevo - antiguo
                    if abs(delta) < 0.1: return f"{nuevo:.1f} (=)"
                    signo = "+" if delta > 0 else ""
                    return f"{nuevo:.1f} ({signo}{delta:.1f})"     
                df_comparacion['Puntos'] = df_comparacion.apply(lambda x: formato_delta(x['Puntos_IA_proy'], x['Puntos_IA_base']), axis=1)
                df_comparacion['Rebotes'] = df_comparacion.apply(lambda x: formato_delta(x['Rebotes_IA_proy'], x['Rebotes_IA_base']), axis=1)
                df_comparacion['Asist.'] = df_comparacion.apply(lambda x: formato_delta(x['Asistencias_IA_proy'], x['Asistencias_IA_base']), axis=1)
                df_comparacion['PTC'] = df_comparacion.apply(lambda x: formato_delta(x['PTC_Proy_proy'], x['PTC_Proy_base']), axis=1)
                df_comparacion['Prob. Éxito'] = df_comparacion['Prob_Exito_proy']
                cols_mostrar = ['Jugadora', 'Puntos', 'Rebotes', 'Asist.', 'PTC', 'Prob. Éxito']
                df_visual = df_comparacion[cols_mostrar].copy()
                # Configuración de estilo de la tabla de visualización de datos
                def color_deltas(val):
                    if isinstance(val, str):
                        if '(+' in val: return 'color: #2ecc71; font-weight: bold;' 
                        elif '(-' in val: return 'color: #e74c3c; font-weight: bold;' 
                        elif '(=)' in val: return 'color: #7f8c8d;' 
                    return ''
                def color_probabilidad(val):
                    if isinstance(val, str):
                        if 'Alta' in val: return 'color: #2ecc71; font-weight: bold;'
                        if 'Dudosa' in val: return 'color: #f1c40f; font-weight: bold;'
                        if 'Baja' in val: return 'color: #e74c3c; font-weight: bold;'
                        if 'No juega' in val: return 'color: #7f8c8d; font-style: italic;'
                    return ''
                try:
                    df_estilizado = df_visual.style\
                        .map(color_deltas, subset=['Puntos', 'Rebotes', 'Asist.', 'PTC'])\
                        .map(color_probabilidad, subset=['Prob. Éxito'])
                except AttributeError:
                    df_estilizado = df_visual.style\
                        .applymap(color_deltas, subset=['Puntos', 'Rebotes', 'Asist.', 'PTC'])\
                        .applymap(color_probabilidad, subset=['Prob. Éxito'])
                # Matriz estratégica y tabla
                tab_matriz, tab_usg, tab_tabla = st.tabs(["Matriz Estratégica","Matriz Uso vs Eficiencia", "Tabla Detallada"])
                with tab_matriz:
                    df_graf = df_sim_ia[df_sim_ia['Minutos_Asignados'] > 0].copy()
                    if not df_graf.empty:
                        colores = ['#2ecc71' if p >= 60 else '#f1c40f' if p >= 35 else '#e74c3c' for p in df_graf['Prob_Num']]
                        fig_mat = px.scatter(
                            df_graf, x='PTC_Proy', y='Prob_Num', text='Jugadora', size='Puntos_IA', 
                            color_discrete_sequence=colores, hover_name='Jugadora',
                            hover_data={'Prob_Num': False, 'PTC_Proy': True, 'Puntos_IA': True}
                        )
                        # Cálculo de la media para la división vertical
                        media_ptc = df_graf['PTC_Proy'].median()
                        if media_ptc == 0: media_ptc = 5.0 
                        # Cuadrantes
                        fig_mat.add_hline(y=50, line_dash="dot", line_color="#bdc3c7")
                        fig_mat.add_vline(x=media_ptc, line_dash="dot", line_color="#bdc3c7")
                        max_ptc = df_graf['PTC_Proy'].max() * 1.1 if df_graf['PTC_Proy'].max() > 0 else 10
                        fig_mat.add_annotation(x=1, y=1, xref="paper", yref="paper", text="ÉXITO ASEGURADO", showarrow=False, font=dict(color="#2ecc71", size=10, weight="bold"), xanchor="right", yanchor="top")
                        fig_mat.add_annotation(x=0, y=1, xref="paper", yref="paper", text="APOYO FIABLE", showarrow=False, font=dict(color="#f1c40f", size=10, weight="bold"), xanchor="left", yanchor="top")
                        fig_mat.add_annotation(x=1, y=0, xref="paper", yref="paper", text="ÉXITO VOLÁTIL", showarrow=False, font=dict(color="#e74c3c", size=10, weight="bold"), xanchor="right", yanchor="bottom")
                        fig_mat.add_annotation(x=0, y=0, xref="paper", yref="paper", text="ROTACIÓN PROFUNDA", showarrow=False, font=dict(color="#7f8c8d", size=10, weight="bold"), xanchor="left", yanchor="bottom")
                        fig_mat.update_traces(textposition='top center', marker=dict(color=colores))
                        ptc_min, ptc_max = df_graf['PTC_Proy'].min(), df_graf['PTC_Proy'].max()
                        prob_min, prob_max = df_graf['Prob_Num'].min(), df_graf['Prob_Num'].max()
                        ptc_pad, prob_pad = max((ptc_max - ptc_min) * 0.15, 1), max((prob_max - prob_min) * 0.15, 5)

                        fig_mat.update_layout(
                            height=400, margin=dict(l=0, r=0, t=30, b=0),
                            xaxis_title="Impacto Proyectado (PTC)", yaxis_title="Probabilidad de Éxito (%)",
                            xaxis=dict(range=[ptc_min - ptc_pad, ptc_max + ptc_pad]),
                            yaxis=dict(range=[prob_min - prob_pad, prob_max + prob_pad])
                        )
                        st.plotly_chart(fig_mat, use_container_width=True, config={'displayModeBar': False})
                    else:
                        st.info("No hay jugadoras en pista para mostrar la matriz.")

                with tab_usg:
                    if not df_graf.empty:
                        # Cálculo medianas
                        media_usg = df_graf['USG_IA'].median()
                        media_efg = df_graf['eFG_IA'].median()
                        fig_usg = px.scatter(
                            df_graf, x='USG_IA', y='eFG_IA', text='Jugadora', size='PTC_Proy',
                            color_discrete_sequence=colores, hover_name='Jugadora',
                            hover_data={'Prob_Num': False, 'USG_IA': True, 'eFG_IA': True, 'PTC_Proy': True}
                        )
                        # Cuadrantes
                        fig_usg.add_hline(y=media_efg, line_dash="dot", line_color="#bdc3c7")
                        fig_usg.add_vline(x=media_usg, line_dash="dot", line_color="#bdc3c7")
                        max_usg = df_graf['USG_IA'].max() * 1.1 if df_graf['USG_IA'].max() > 0 else 30
                        max_efg = df_graf['eFG_IA'].max() * 1.1 if df_graf['eFG_IA'].max() > 0 else 60
                        fig_usg.add_annotation(x=1, y=1, xref="paper", yref="paper", text="EFICIENTES Y PRODUCTIVAS", showarrow=False, font=dict(color="#2ecc71", size=10, weight="bold"), xanchor="right", yanchor="top")
                        fig_usg.add_annotation(x=0, y=1, xref="paper", yref="paper", text="CATCH & SHOOT", showarrow=False, font=dict(color="#f1c40f", size=10, weight="bold"), xanchor="left", yanchor="top")
                        fig_usg.add_annotation(x=1, y=0, xref="paper", yref="paper", text="AMASADORAS INEFICIENTES", showarrow=False, font=dict(color="#e74c3c", size=10, weight="bold"), xanchor="right", yanchor="bottom")
                        fig_usg.add_annotation(x=0, y=0, xref="paper", yref="paper", text="PERFIL BAJO / DEFENSIVO", showarrow=False, font=dict(color="#7f8c8d", size=10, weight="bold"), xanchor="left", yanchor="bottom")
                        fig_usg.update_traces(textposition='top center', marker=dict(color=colores))
                        usg_min, usg_max = df_graf['USG_IA'].min(), df_graf['USG_IA'].max()
                        efg_min, efg_max = df_graf['eFG_IA'].min(), df_graf['eFG_IA'].max()
                        usg_pad, efg_pad = max((usg_max - usg_min) * 0.15, 1), max((efg_max - efg_min) * 0.15, 1)

                        fig_usg.update_layout(
                            height=400, margin=dict(l=0, r=0, t=30, b=0),
                            xaxis_title="Volumen Ofensivo Proyectado (USG%)",
                            yaxis_title="Eficiencia de Tiro Proyectada (eFG%)",
                            xaxis=dict(range=[usg_min - usg_pad, usg_max + usg_pad]), 
                            yaxis=dict(range=[efg_min - efg_pad, efg_max + efg_pad]))
                        st.plotly_chart(fig_usg, use_container_width=True, config={'displayModeBar': False})
                    else:
                        st.info("No hay jugadoras en pista para mostrar la matriz.")        
                with tab_tabla:
                    st.dataframe(df_estilizado, use_container_width=True, hide_index=True)
                st.divider()
                st.markdown(f"#### Proyección del equipo ({equipo_a_rotar})")
                # Usamos las estadísisticas optimas como valores para la comparación
                base_pts = df_base_ia['Puntos_IA'].sum()
                base_reb = df_base_ia['Rebotes_IA'].sum()
                base_ast = df_base_ia['Asistencias_IA'].sum()
                base_ptc = df_base_ia['PTC_Proy'].sum() 
                # Proyección de la IA
                new_pts = df_sim_ia['Puntos_IA'].sum()
                new_reb = df_sim_ia['Rebotes_IA'].sum()
                new_ast = df_sim_ia['Asistencias_IA'].sum()
                new_ptc = df_sim_ia['PTC_Proy'].sum() 
                c_pt, c_rb, c_as, c_ptc_card = st.columns(4)
                c_pt.metric("Puntos Esperados", f"{new_pts:.1f}", delta=f"{new_pts - base_pts:.1f}")
                c_rb.metric("Rebotes Esperados", f"{new_reb:.1f}", delta=f"{new_reb - base_reb:.1f}")
                c_as.metric("Asist. Esperadas", f"{new_ast:.1f}", delta=f"{new_ast - base_ast:.1f}")
                c_ptc_card.metric("PTC Total", f"{new_ptc:.1f}", delta=f"{new_ptc - base_ptc:.1f}")
                # Modelo interpretativo con SHAP para explicar el razonamiento del modelo
                st.markdown("---")
                st.markdown("### Claves del Rendimiento de las Jugadoras")
                # Desplegable para seleccionar la jugadora a explicar
                lista_jugadoras = df_base_sim[df_base_sim['Minutos_Asignados'] > 0]['Jugadora'].tolist()
                if lista_jugadoras:
                        st.write("Selecciona una jugadora en pista:")
                        jugadora_shap = st.selectbox("Jugadora a analizar:", lista_jugadoras, label_visibility="collapsed")
                        fila_jugadora = df_base_sim[df_base_sim['Jugadora'] == jugadora_shap].copy()
                        fila_jugadora['cluster_def_rival'] = cluster_rival
                        fila_jugadora['es_local'] = factor_cancha
                        fila_jugadora['pace_rival'] = pace_rival
                        mins_hoy = float(fila_jugadora['Minutos_Asignados'].iloc[0])
                        fila_jugadora['minutos_float'] = mins_hoy  # CRÍTICO: La nueva variable de la IA
                        for col in fila_jugadora.columns:
                            if 'cluster_def_rival_' in col:
                                fila_jugadora[col] = 0.0
                        features = ia_models['features']
                        for col in features:
                            if col not in fila_jugadora.columns: fila_jugadora[col] = 0.0
                        col_cluster_shap = f'cluster_def_rival_{int(cluster_rival)}'
                        if col_cluster_shap in fila_jugadora.columns:
                            fila_jugadora[col_cluster_shap] = 1.0
                        X_jugadora = fila_jugadora[features]
                        # Aplicación de SHAP
                        try:
                            explainer = shap.Explainer(ia_models['clf'])
                            shap_values = explainer(X_jugadora)
                            if len(shap_values.shape) == 3:
                                shap_vals = shap_values[0, :, 1].values
                                base_val = shap_values[0, :, 1].base_values
                            else:
                                shap_vals = shap_values[0].values
                                base_val = shap_values[0].base_values
                            if isinstance(base_val, (np.ndarray, list)):
                                base_val = base_val[1] if len(base_val) > 1 else base_val[0]
                                # Diccionario de variables para una mejor comprensión del entrenador
                            NOMBRES_VARIABLES = {
                                'Sn_puntos': 'Puntos (Temporada)', 'Sn_reb_tot': 'Rebotes (Temporada)', 
                                'Sn_asist': 'Asistencias (Temporada)', 'Sn_recup': 'Robos (Temporada)', 
                                'Sn_tap_fav': 'Tapones (Temporada)', 'Sn_perd': 'Pérdidas (Temporada)',
                                'Sn_PTC': 'Impacto Total PTC', 'Sn_PTC_mp': 'Impacto PTC / 100 Posesiones',
                                'Sn_minutos_float': 'Minutos Habituales', 'Sn_PTS_mpu': 'Puntos por Uso Ofensivo',
                                'Sn_eFG': 'Tiro Efectivo (eFG%)', 'Sn_USG': 'Uso Ofensivo (USG%)',
                                'L6_puntos': 'Racha Puntos (Últ. 6)', 'L6_reb_tot': 'Racha Rebotes (Últ. 6)', 
                                'L6_asist': 'Racha Asist. (Últ. 6)', 'L6_recup': 'Racha Robos (Últ. 6)', 
                                'L6_tap_fav': 'Racha Tapones (Últ. 6)', 'L6_perd': 'Racha Pérdidas (Últ. 6)',
                                'L6_PTC': 'Momento de Forma (Racha PTC)', 'L6_PTC_mp': 'Racha PTC / 100 Pos.',
                                'L6_PTS_mpu': 'Racha Pts por Uso', 'L6_eFG': 'Racha eFG%', 
                                'L6_USG': 'Racha Uso Ofensivo', 'L6_minutos_float': 'Racha Minutos',
                                'cluster_def_rival': 'Estilo Defensivo Rival', 
                                'es_local': 'Factor Cancha (Local/Visitante)', 
                                'pace_rival': 'Ritmo de Partido del Rival (Pace)',
                                'Minutos_Asignados': 'Minutos Asignados Hoy'
                    }
                            # Traducción a Dataframe
                            df_shap = pd.DataFrame({
                                'Factor': X_jugadora.columns,
                                'Impacto': shap_vals
                            })
                            df_shap.loc[df_shap['Factor'].str.startswith('cluster_def_rival'), 'Factor'] = 'cluster_def_rival'
                            df_shap['Factor_Limpio'] = df_shap['Factor'].map(NOMBRES_VARIABLES).fillna(df_shap['Factor'])
                            df_shap = df_shap.groupby('Factor_Limpio', as_index=False)['Impacto'].sum() # Para que los clusteres no salgan divididos
                            df_shap['Magnitud'] = df_shap['Impacto'].abs()
                            df_shap = df_shap.sort_values('Magnitud', ascending=False)
                            # Se visualizan únicamente los 5 valores con mayor peso, luego el resto los agrupamos en "otros"
                            top_n = 5
                            df_top = df_shap.head(top_n)
                            otros_impacto = df_shap.iloc[top_n:]['Impacto'].sum()
                            # Construcción del grásfico de tipo cascada
                            y_labels = ['Base'] + df_top['Factor_Limpio'].tolist() + ['Otros Factores']
                            x_values = [base_val] + df_top['Impacto'].tolist() + [otros_impacto, 0] 
                            medidas = ['absolute'] + ['relative'] * top_n + ['relative', 'total']
                            textos = [f"{v:.2f}" if m == 'absolute' else f"{v:+.2f}" for v, m in zip(x_values[:-1], medidas[:-1])]
                            textos.append(f"{sum(x_values[:-1]):.2f}") 
                            fig = go.Figure(go.Waterfall(
                                name="Análisis Táctico", 
                                orientation="h", 
                                measure=medidas,
                                y=y_labels,      
                                x=x_values,      
                                textposition="outside",
                                text=textos,
                                connector={"line": {"color": "rgb(63, 63, 63)", "dash": "dot"}},
                                decreasing={"marker": {"color": "#e74c3c"}}, # Rojo
                                increasing={"marker": {"color": "#2ecc71"}}, # Verde
                                totals={"marker": {"color": "#3498db"}}      # Azul
                            ))
                            fig.update_layout(
                                title=f"Factores relevantes en la proyección de {jugadora_shap}?",
                                showlegend=False,
                                height=450,
                                margin=dict(l=150, r=40, t=50, b=20), # Más margen izquierdo para que quepan bien los nombres
                                xaxis_title="Índice de Éxito (Impacto Logarítmico)",
                                yaxis=dict(autorange="reversed") # <-- CRUCIAL: Para que se lea de arriba a abajo en formato escalera
                            )
                            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                        except Exception as e:
                            st.error(f" No se pudo generar el gráfico explicativo {e}")
                else:
                    st.warning("Asigna minutos en los sliders a alguna jugadora para poder analizar sus estadísiticas clave ")
        # 13. Informe de Scouting con modelo LLM de Gemini
        st.markdown("---")
        st.markdown("### Informe Táctico")
        col_sel, col_btn, col_vacia = st.columns([1.5, 1.5, 4])
        with col_sel:
            perspectiva = st.selectbox(
                "Seleccionar equipo:",
                [d['equipo_local'], d['equipo_visitante']],
                index=0
            )
        def generar_texto_scouting(df_top):
            texto = ""
            for i, row in df_top.head(3).iterrows():
                delta_pts = row['Puntos_IA'] - row['Sn_puntos']
                estado = "Jugadora en racha" if delta_pts > 2 else ("Jugadora en baja forma" if delta_pts < -2 else "Jugadora con rendimiento normal")
                
                texto += f"  - {row['Jugadora']} ({estado} | Delta Pts: {delta_pts:+.1f}):\n"
                texto += f"    * ATAQUE: {row['Puntos_IA']:.1f} Pts | {row['Asistencias_IA']:.1f} Ast | {row.get('perd_IA', 0):.1f} Pérdidas | {row['USG_IA']:.1f}% USG\n"
                texto += f"    * TIROS: {row.get('t2m_IA',0):.1f}/{row.get('t2i_IA',0):.1f} en T2 | {row.get('t3m_IA',0):.1f}/{row.get('t3i_IA',0):.1f} en T3 | {row.get('tlm_IA',0):.1f}/{row.get('tli_IA',0):.1f} en TL (eFG%: {row['eFG_IA']:.1f}%)\n"
                texto += f"    * REBOTE: {row.get('reb_of_IA',0):.1f} Ofensivos | {row.get('reb_def_IA',0):.1f} Defensivos\n"
                texto += f"    * DEFENSA Y FALTAS: {row['recup_IA']:.1f} Robos | {row.get('tap_IA',0):.1f} Tapones | {row.get('faltas_c_IA',0):.1f} Faltas Cometidas | {row.get('faltas_r_IA',0):.1f} Faltas Recibidas\n\n"
            return texto
        if st.button("Generar Informe", type="primary"):
            with st.spinner("..."):
                try:
                    import google.generativeai as genai
                    # Configuración de la API
                    API_KEY = st.secrets.get("GEMINI_API_KEY", "")
                    if not API_KEY:
                        st.error("Falta la API Key de Gemini")
                    else:
                        genai.configure(api_key=API_KEY)
                        if perspectiva == d['equipo_local']:
                            mi_equipo = d['equipo_local']
                            rival = d['equipo_visitante']
                            mis_stats = d['stats_local']
                            rival_stats = d['stats_visit']
                            mi_estilo = cluster_local_txt
                            rival_estilo = cluster_visit_txt
                            mi_orb_pct = orb_loc_pct
                            mi_drb_pct = drb_loc_pct
                            mis_top = top_loc
                            rival_top = top_vis
                            cancha = "en nuestra casa (como locales)"
                            mi_orb_pct = orb_loc_pct
                            mi_drb_pct = drb_loc_pct
                            rival_orb_pct = orb_vis_pct
                            rival_drb_pct = drb_vis_pct
                            df_mio = pred_loc
                            texto_mis_top = generar_texto_scouting(mis_top)
                            texto_rival_top = generar_texto_scouting(rival_top)
                        else:
                            mi_equipo = d['equipo_visitante']
                            rival = d['equipo_local']
                            mis_stats = d['stats_visit']
                            rival_stats = d['stats_local']
                            mi_estilo = cluster_visit_txt
                            rival_estilo = cluster_local_txt
                            mi_orb_pct = orb_vis_pct
                            mi_drb_pct = drb_vis_pct
                            mis_top = top_vis
                            rival_top = top_loc
                            cancha = "fuera de casa (como visitantes)"
                            mi_orb_pct = orb_vis_pct
                            mi_drb_pct = drb_vis_pct
                            rival_orb_pct = orb_loc_pct
                            rival_drb_pct = drb_loc_pct
                            df_mio = pred_vis
                            texto_mis_top = generar_texto_scouting(mis_top)
                            texto_rival_top = generar_texto_scouting(rival_top)
                        # Listado de la plantilla compelta que va a jugar el partido
                        # Listado de la plantilla completa que va a jugar el partido
                        mi_plantilla_str = ""
                        for _, row in df_mio.iterrows():
                            if row.get('Minutos_Asignados', 1) > 0: 
                                mi_plantilla_str += (
                                    f"- {row['Jugadora']}:\n"
                                    f"  Ataque: {row['Puntos_IA']:.1f} Pts (T3: {row.get('t3m_IA',0):.1f}/{row.get('t3i_IA',0):.1f}), {row['Asistencias_IA']:.1f} Ast, {row.get('perd_IA',0):.1f} Pérdidas.\n"
                                    f"  Rebote: {row.get('reb_of_IA',0):.1f} Of, {row.get('reb_def_IA',0):.1f} Def.\n"
                                    f"  Defensa y Faltas: {row['recup_IA']:.1f} Rob, {row.get('tap_IA',0):.1f} Tap | Faltas: {row.get('faltas_c_IA',0):.1f} Com, {row.get('faltas_r_IA',0):.1f} Rec.\n"
                                    f"  Impacto Global: PTC {row['PTC_Proy']:.1f} | USG {row['USG_IA']:.1f}%\n"
                                )
                        # Construcción del Prompt
                        prompt_tactico = f"""
                        Eres el entrenador asistente y analista de datos avanzado del equipo {mi_equipo}.
                        Hoy jugamos contra {rival} {cancha}.
                        
                        CONTEXTO DE DATOS AVANZADOS DE LOS EQUIPOS:
                        1. Ritmo de Juego (PACE): Nosotros jugamos a {mis_stats['PACE']:.1f} posesiones (con el siguiente estilo de juego: {mi_estilo}). El rival juega a {rival_stats['PACE']:.1f} posesiones (con el siguiente estilo de juego: {rival_estilo}).
                        2. Tiro Efectivo: Nuestro eFG% es {mis_stats['eFG%']:.1f}%. El del rival es {rival_stats['eFG%']:.1f}%.
                        3. Cuidado del Balón (TOV%): Perdemos el {mis_stats['TOV%']:.1f}% de las posesiones. El rival pierde el {rival_stats['TOV%']:.1f}% de las posesiones.
                        4. Agresividad (FTR - Tiros Libres): Nuestro FTR es {mis_stats.get('FTR', 0):.1f} frente al {rival_stats.get('FTR', 0):.1f} del rival.
                        5. Lucha por el Rebote: Capturamos el {mi_orb_pct:.1f}% de los rebotes ofensivos disponibles y aseguramos el {mi_drb_pct:.1f}% de nuestro aro defensivo. Por otro lado, el rival captura el {rival_orb_pct:.1f}% en ataque y asegura el {rival_drb_pct:.1f}% en defensa.

                        NUESTRA PLANTILLA DISPONIBLE HOY:
                        Revisa toda nuestra plantilla y sus perfiles ofensivos y defensivos (robos, tapones, pérdidas y demás estadísticas):
                        {mi_plantilla_str}

                        PROYECCIONES PARA EL PARTIDO DE HOY SEGÚN NUESTRO MODELO PREDICTIVO DESARROLLADO VIA XGBOOST:
                        - NUESTRAS JUGADORAS CLAVE HOY:
                        {texto_mis_top}

                        - Las jugadoras rivales con mayor proyección ofensiva contra nosotras en el partido de hoy son las siguientes (a las cuales debemos poner el foco en defensa):
                        {texto_rival_top}

                        TAREA Y FORMATO ESTRICTO:
                        Redacta el informe de scouting siguiendo EXACTAMENTE la estructura que se detalla a continuación. 
                        
                        REGLAS DE ESTILO OBLIGATORIAS: 
                        1. Usa un tono pragmático, asertivo y vocabulario técnico de baloncesto profesional comprensible por el staff técnico. 
                        2. Justifica cada propuesta de carácter táctico o afirmación que aportes con los datos numéricos provistos arriba.
                        3. Está PROHIBIDO usar saludos iniciales o introducciones genéricas de tipo "Querido entrenador", "Buenas", o "Aquí tienes el informe". Inicia el texto directamente con el primer apartado.
                        4. SÉ EXTREMADAMENTE CONCISO. Prohibido el texto de relleno, la lírica o las oraciones largas. Ve directo al dato y a la orden táctica. Si puedes decirlo en 10 palabras, no uses 20.
                        5. Hazlo visual y claro, utilizando obligatoriamente la siguiente estructura con los titulares en negrita (### marca los textos en negrita):

                        ### 1. ESTILO Y RITMO DE JUEGO
                        Compara nuestro estilo colectivo con el suyo. Basándote en el PACE de ambos equipos, indica a qué ritmo exacto nos interesa llevar el partido hoy para sacar ventaja. Analiza también el eFG% e indica quién tira mejor y cómo condiciona esto al plan de partido.

                        ### 2. DEBILIDADES
                        Identifica en qué métricas colectivas somos inferiores hoy respecto al rival. Propón ajustes tácticos específicos para esconder o minimizar esta carencia durante el partido.

                        ### 3. AMENAZAS
                        Identifica en qué métricas colectivas son ellas superiores hoy. Define un plan táctico claro sobre qué debemos hacer en el partido para lidiar contra esa ventaja matemática que tienen frente a nosotras.

                        ### 4. FORTALEZAS
                        Identifica en qué métricas colectivas somos estadísticamente superiores hoy. Explica qué debemos hacer para fortalecer esta superioridad táctica frente al rival y beneficiarnos de tal manera que saquemos de ello la mayor ventaja posible.

                        ### 5. OPORTUNIDADES
                        Identifica en qué métricas colectivas son ellas inferiores hoy. Propón cómo debemos atacar o plantear la defensa y ataque para sacar el máximo beneficio y castigar esa carencia específica que tienen.

                        ### 6. EMPAREJAMIENTOS CLAVE
                        Analiza detalladamente NUESTRA PLANTILLA COMPLETA facilitada en los datos. Para defender a las 6 AMENAZAS DEL RIVAL sin que sea necesario elegir a nuestras mejores anotadoras; busca en nuestro banquillo perfiles de "especialistas defensivas" frente a ellas y asígnales su marca. Explica brevemente por qué has elegido a cada una de nuestras jugadoras basándote en sus métricas, con foco en las defensivas.
                        """
                        
                        # Llamada al modelo y respuesta del mismo
                        model = genai.GenerativeModel('gemini-2.5-flash') 
                        respuesta = model.generate_content(prompt_tactico)
                        texto_limpio = respuesta.text
                        for i in range(1, 7):
                            texto_limpio = texto_limpio.replace(f"###{i}.", f"### {i}.")
                            texto_limpio = texto_limpio.replace(f"### {i}. ESTILO Y RITMO DE JUEGO", f"### {i}. ESTILO Y RITMO DE JUEGO\n")
                            texto_limpio = texto_limpio.replace(f"### {i}. DEBILIDADES", f"### {i}. DEBILIDADES\n")
                            texto_limpio = texto_limpio.replace(f"### {i}. AMENAZAS", f"### {i}. AMENAZAS\n")
                            texto_limpio = texto_limpio.replace(f"### {i}. FORTALEZAS", f"### {i}. FORTALEZAS\n")
                            texto_limpio = texto_limpio.replace(f"### {i}. OPORTUNIDADES", f"### {i}. OPORTUNIDADES\n")
                            texto_limpio = texto_limpio.replace(f"### {i}. EMPAREJAMIENTOS CLAVE", f"### {i}. EMPAREJAMIENTOS CLAVE\n")
                        texto_limpio = texto_limpio.replace("###  ", "### ")
                        # Mostramos el resultado
                        st.success(" Informe generado con éxito")
                        with st.container(border=True):
                            st.markdown(f"#### Reporte Pre-Partido: {d['equipo_local']} vs {d['equipo_visitante']}")
                            st.markdown("---") # Una línea divisoria elegante
                            st.markdown(respuesta.text) # Aquí el Markdown sí se renderizará perfecto
                except Exception as e:
                    st.error(f"Error de conexión con la IA: {e}")

