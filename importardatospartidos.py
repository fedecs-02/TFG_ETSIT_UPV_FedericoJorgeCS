import os #Para gestión de archivos
import time #Para introducir pausas
import json #Para poder trabaar con archivos JSON
import pandas as pd #Para manipulación de datos y guardado en csv
from bs4 import BeautifulSoup #Para analizar el html estático
#Importaciones concretas de Selenium
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

#CONFIGURACIÓN
OUTDIR = "feb_data" #Carpeta de destino
INPUT_JSON = os.path.join(OUTDIR, "matches_all.json")
OUTPUT_CSV = os.path.join(OUTDIR, "match_stats_global.csv")
#Ruta del driver
CHROMEDRIVER_PATH = r"C:\Users\feder\Downloads\CHROMEDRIVER\chromedriver.exe"

#FUNCIÓN ARRANQUE DEL DRIVER
def init_driver():
    """Inicia el navegador sin que se visualice la pantalla."""
    options = Options() #Objeto de configuración de las opciones
    options.add_argument("--headless=new") #Para que no se visualice la pantalla
    options.add_argument("--window-size=1920,1080") #Para cargar el tamaño habitual del navegador aunque no se vaya a visualizar, y evitar errores.
    #Para mejorar la estabilidad del driver
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36") #User Agent para simular un usuario real y evitar bloqueos
    #Inicio del driver en el path indicado o el del path del sistema en caso de no encontrarlo.
    if os.path.exists(CHROMEDRIVER_PATH):
        service = Service(CHROMEDRIVER_PATH)
        driver = webdriver.Chrome(service=service, options=options)
    else:
        driver = webdriver.Chrome(options=options)
    return driver

#FUNCIÓN AUXILIAR ANÁLISIS DE TABLAS
def parse_team_stats(soup_table, prefix):
    """Analiza la tabla HTML de los equipo y extrae la fila del total de cada uno (row-total)."""
    data = {}
    try:
        row_total = soup_table.find("tr", class_="row-total") #Buscamos la fila que se llame "row-total", nombre que se le ha puesto en el html a la fila total del equipo
        if not row_total: return {} #Si no se encuentra nada, se devuelve vacío

        cols = row_total.find_all("td") #Extrae todas las celdas de la fila
        if len(cols) < 15: return {} #Con esta línea verificamos que la fila tiene más de 15 columnas, que son las que tiene la tabla que nos interesa. Así se evitan errores

        #FUNCIÓN DEPURACIÓN DE ESPACIOS
        #Convierte cadenas de texto como "20/40 50%" en dos datos: metidos e intentados
        def split_shoot(text):
            clean = text.strip().split(' ')[0] #Nos quedamos únicamente con la información previa al espacio
            if '/' in clean: return clean.split('/') #Devuelve los 2 datos separados por "/" 
            return (0, 0) #En caso de error, se devuelve 0,0

        #EXTRACCIÓN DE DATOS SEGÚN LA POSICIÓN EN EL HTML
        data[f"{prefix}_pts"] = int(cols[4].text.strip()) #Puntos totales
        
        t2_m, t2_i = split_shoot(cols[5].text) #Tiros de 2
        data[f"{prefix}_t2_met"] = t2_m
        data[f"{prefix}_t2_int"] = t2_i
        
        t3_m, t3_i = split_shoot(cols[6].text) #Triples
        data[f"{prefix}_t3_met"] = t3_m
        data[f"{prefix}_t3_int"] = t3_i
        
        tl_m, tl_i = split_shoot(cols[8].text) #Tiros libres
        data[f"{prefix}_tl_met"] = tl_m
        data[f"{prefix}_tl_int"] = tl_i

        #Rebotes
        data[f"{prefix}_reb_of"] = int(cols[9].text.strip())
        data[f"{prefix}_reb_def"] = int(cols[10].text.strip())
        data[f"{prefix}_reb_tot"] = int(cols[11].text.strip())
        
        #Más estadísticas
        data[f"{prefix}_asist"]   = int(cols[12].text.strip())
        data[f"{prefix}_recup"]   = int(cols[13].text.strip())
        data[f"{prefix}_perd"]    = int(cols[14].text.strip())

        #Faltas
        data[f"{prefix}_faltas_c"] = int(cols[18].text.strip())
        data[f"{prefix}_faltas_r"] = int(cols[19].text.strip())

        #Valoración
        data[f"{prefix}_val"]     = int(cols[20].text.strip())

    except Exception as e:
        pass # Error silencioso para no ensuciar consola
    
    return data


#FUNCIÓN PRINCIPAL: PROCESADO DE UN PARTIDO
#Entra en la url del partido y obtiene los datos deseados
def process_match(driver, match_info):
    url = match_info.get('link')
    if not url: url = f"https://baloncestoenvivo.feb.es/Partido.aspx?p={match_info['id_partido']}"
    
    match_id = match_info['id_partido']
    print(f"PROCESANDO ID {match_id}...", end="\r") #Visualizamos progreso del procedimento. Luego se borra ara no generar ruido en la terminal

    #Obtiene nombre de local y visitante del JSON
    nombre_local = match_info.get('local', 'Local')
    nombre_visitante = match_info.get('visitante', 'Visitante')
    
    try:
        driver.get(url) #Entramos en la url
        
        # Espera de 8 segundos máximo para obtener un elemento de tipo "responsive-scroll" que garantice que la tabla se ha cargado correctamente antes de obtener los datos
        WebDriverWait(driver, 8).until(
            EC.presence_of_element_located((By.CLASS_NAME, "responsive-scroll"))
        )
        
        soup = BeautifulSoup(driver.page_source, "html.parser") #Pasamos el código html a BeautifulSoup, más rápido que Selenium
        
        #MARCADOR Y GANADOR
        try:
            #Búsqueda del marcador
            res_loc = int(soup.select_one(".equipo.local .resultado").text.strip())
            res_vis = int(soup.select_one(".equipo.visitante .resultado").text.strip())
            #Búsqueda del ganador
            ganador = nombre_local if res_loc > res_vis else (nombre_visitante if res_vis > res_loc else "Empate")
        except:
            return None

        #Línea con datos básicos para que funcione como diccionario
        row = {
            "id_partido": match_id,
            "liga": match_info.get('liga_clave'),
            "temporada": match_info.get('temporada'),
            "equipo_local": match_info.get('local'),
            "equipo_visitante": match_info.get('visitante'),
            "puntos_local": res_loc,
            "puntos_visit": res_vis,
            "ganador": ganador
        }

        #PARCIALES Y GANADORES
        try:
            #Buscamos los puntos de cada cuarto
            spans_l = soup.select(".fila.parciales .columna.equipo.local span")
            spans_v = soup.select(".fila.parciales .columna.equipo.visitante span")
            #Recorremos los cuartos encontrados
            for i in range(len(spans_l)):
                val_l = int(spans_l[i].text.strip() or 0)
                val_v = int(spans_v[i].text.strip() or 0)
                #Guardamos puntos del cuarto por equipos
                row[f"q{i+1}_loc"] = val_l
                row[f"q{i+1}_vis"] = val_v
                #Determinamos ganador de cada cuarto o empate
                if val_l > val_v: row[f"ganador_q{i+1}"] = nombre_local
                elif val_v > val_l: row[f"ganador_q{i+1}"] = nombre_visitante
                else: row[f"ganador_q{i+1}"] = "Empate"
        except:
            pass

        #ESTADÍSTICAS DEL EQUIPO

        #Buscamos los contenedores de tablas "responsive-scroll"
        containers = soup.find_all("div", class_="responsive-scroll")
        valid_tables = []
        for c in containers:
            table = c.find("table")
            #Buscamos las filas "row-total" que contienen los datos totales
            if table and table.find("tr", class_="row-total"):
                valid_tables.append(table)
        #Guardamos los datos de local y visitante
        if len(valid_tables) >= 2:
            row.update(parse_team_stats(valid_tables[0], "loc"))
            row.update(parse_team_stats(valid_tables[1], "vis"))
            print(f"[OK] ID {match_id}: Datos extraídos correctamente.")
            return row
        else:
            print(f"[ERROR] ID {match_id}: Tablas no encontradas.")
            return None

    except Exception as e:
        print(f"[ERROR] ID {match_id}: {e}")
        return None

#BLOQUE DE EJECUCIÓN
if __name__ == "__main__":
    #Comprobar si exite el JSON
    if not os.path.exists(INPUT_JSON):
        print(f"ERROR: Falta {INPUT_JSON}")
        exit()
    #Carga de datos del JSON
    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        all_matches = json.load(f)

    # Ejecuta todos los partidos
    matches_to_run = all_matches

    print(f"Iniciando proceso HEADLESS para {len(matches_to_run)} partidos...")
    print(f"El navegador funcionará en segundo plano.")
    
    #Inicio del driver en el navegador
    driver = init_driver()
    results = []

    #Bucle de iteración partido a partido
    for i, m in enumerate(matches_to_run):
        data = process_match(driver, m)
        if data:
            results.append(data)
            # Guardado del csv en cada iteración
            pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig') #utf-8-sig para leer tildes y ñ en el excel
        
        # Pausa para no saturar
        time.sleep(0.35)
    #Cierre de navegador
    driver.quit()
    print(f"\nPROCESO FINALIZADO. Archivo: {os.path.abspath(OUTPUT_CSV)}")