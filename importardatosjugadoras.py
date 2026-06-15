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
OUTDIR = "feb_data"
INPUT_JSON = os.path.join(OUTDIR, "matches_all.json")
OUTPUT_CSV = os.path.join(OUTDIR, "players_stats_detailed.csv")
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
def parse_players_from_table(soup_table, match_id, team_name, is_local):
    """
    Extrae todas las filas de jugadoras de una tabla específica.
    """
    players_list = []
    
    # Buscamos el cuerpo de la tabla
    tbody = soup_table.find("tbody")
    if not tbody: return []
    
    rows = tbody.find_all("tr")
    
    for row in rows:
        #Saltamos la fila de totales ("row-total") y vacías
        if "row-total" in row.get("class", []) or not row.find("td"):
            continue
        
        cols = row.find_all("td")
        # Si la fila tiene pocas columnas, se descarta porque es un separador o ruido
        if len(cols) < 15: continue 

        #Extracción de datos
        try:
            #Convierte cadenas de texto como "20/40 50%" en dos datos: metidos e intentados
            def split_shoot(text):
                clean = text.strip().split(' ')[0]
                if '/' in clean: return clean.split('/')
                return (0, 0)

            #Extracción datos básicos
            es_titular = 1 if "*" in cols[0].text else 0
            dorsal = cols[1].text.strip()
            nombre = cols[2].text.strip()
            #Saltamos si no hya nombre de la jugadora
            if not nombre: continue
            #Minutos jugados
            minutos = cols[3].text.strip()
            
            # Estadísticas
            pts = int(cols[4].text.strip() or 0)
            #Tiros de 2 y triples
            t2_m, t2_i = split_shoot(cols[5].text)
            t3_m, t3_i = split_shoot(cols[6].text)
            #Tiros de campo
            tc_m, tc_i = split_shoot(cols[7].text)
            #Tiros libres
            tl_m, tl_i = split_shoot(cols[8].text)
            #Rebotes
            reb_of = int(cols[9].text.strip() or 0)
            reb_def = int(cols[10].text.strip() or 0)
            reb_tot = int(cols[11].text.strip() or 0)
            #Otras estadísticas
            asist = int(cols[12].text.strip() or 0)
            recup = int(cols[13].text.strip() or 0)
            perd  = int(cols[14].text.strip() or 0)
            tap_fav = int(cols[15].text.strip() or 0)
            tap_con = int(cols[16].text.strip() or 0)
            mates   = int(cols[17].text.strip() or 0)
            #Faltas
            faltas_c = int(cols[18].text.strip() or 0)
            faltas_r = int(cols[19].text.strip() or 0)
            #Valoración
            val = int(cols[20].text.strip() or 0)
            #Más/Menos
            plus_minus = int(cols[21].text.strip() or 0) if len(cols) > 21 else 0

            # Construimos el objeto Jugadora
            player_data = {
                "id_partido": match_id,
                "equipo": team_name,
                "es_local": "Local" if is_local else "Visitante",
                "nombre": nombre,
                "dorsal": dorsal,
                "titular": es_titular,
                "minutos": minutos,
                "puntos": pts,
                "t2_met": t2_m, "t2_int": t2_i,
                "t3_met": t3_m, "t3_int": t3_i,
                "tl_met": tl_m, "tl_int": tl_i,
                "reb_of": reb_of, "reb_def": reb_def, "reb_tot": reb_tot,
                "asist": asist, "recup": recup, "perd": perd,
                "tap_fav": tap_fav, "tap_con": tap_con, "mates": mates,
                "faltas_c": faltas_c, "faltas_r": faltas_r,
                "val": val, "mas_menos": plus_minus
            }
            #Añadimos la jugadora a la lista
            players_list.append(player_data)

        except Exception as e:
            # Si falla una jugadora concreta, no paramos todo el partido
            continue

    return players_list #Devuelve las jugadoras del partido

#FUNCIÓN PRINCIPAL: PROCESADO DE UN PARTIDO
#Entra en la url del partido y obtiene los datos deseados
def process_match_players(driver, match_info):
    url = match_info.get('link')
    if not url: url = f"https://baloncestoenvivo.feb.es/Partido.aspx?p={match_info['id_partido']}"
    
    ##
    match_id = match_info['id_partido']
    print(f"PROCESANDO ID {match_id}...", end="\r") #Visualizamos progreso del procedimento. Luego se borra ara no generar ruido en la terminal

    #Obtiene nombre de local y visitante del JSON
    nombre_local = match_info.get('local', 'Local')
    nombe_visitante = match_info.get('visitante', 'Visitante')
    
    try:
        driver.get(url)#Entramos en la url
        
        # Espera de 8 segundos máximo para obtener un elemento de tipo "responsive-scroll" que garantice que la tabla se ha cargado correctamente antes de obtener los datos
        WebDriverWait(driver, 8).until(
            EC.presence_of_element_located((By.CLASS_NAME, "responsive-scroll"))
        )
        
        soup = BeautifulSoup(driver.page_source, "html.parser") #Pasamos el código html a BeautifulSoup, más rápido que Selenium
        
        # Buscamos los contenedores de las tablas
        containers = soup.find_all("div", class_="responsive-scroll")
        
        valid_tables = []
        # Filtramos tablas que contengan datos de jugadores
        for c in containers:
            table = c.find("table")
            # Únicamente seleccionamos las tablas que contengan la clase "dorsal"
            if table and table.find("th", class_="dorsal"):
                valid_tables.append(table)
        
        match_players = []
        #Buscamos la tabla local y visitante
        if len(valid_tables) >= 2:
            # Tabla Local
            p_loc = parse_players_from_table(valid_tables[0], match_id, nombre_local, is_local=True)
            match_players.extend(p_loc)
            
            # Tabla Visitante
            p_vis = parse_players_from_table(valid_tables[1], match_id, nombe_visitante, is_local=False)
            match_players.extend(p_vis)
            
            # Añadimos información sobre el partido
            for p in match_players:
                p['liga'] = match_info.get('liga_clave')
                p['temporada'] = match_info.get('temporada')
                p['fecha'] = match_info.get('fecha', '')

            print(f"   [OK] ID {match_id}: {len(match_players)} jugadoras extraídas.")
            return match_players
        else:
            print(f"   [X] ID {match_id}: No se encontraron tablas de jugadoras.")
            return []

    except Exception as e:
        return []

#FUNCIÓN PRINCIPAL
if __name__ == "__main__":
    #Verificamos que se detecte el JSON
    if not os.path.exists(INPUT_JSON):
        print(f"ERROR: Falta {INPUT_JSON}")
        exit()
    #Cargamos datos del JSON
    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        all_matches = json.load(f)
    matches_to_run = all_matches

    print(f"Extrayendo estadísticas individuales de {len(matches_to_run)} partidos")
    print(f"Guardando en: {os.path.abspath(OUTPUT_CSV)}")
    #Iniciamos navegador
    driver = init_driver()
    all_data = [] #Lista destino de los datos

    #Bucle de iteración partido a partido
    for i, m in enumerate(matches_to_run):
        players = process_match_players(driver, m)
        if players:
            all_data.extend(players)
            # Guardado del csv en cada iteración 
            pd.DataFrame(all_data).to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
        
        # Pausa para no saturar
        time.sleep(0.35)
    #Cierre del navegador
    driver.quit()
    print(f"\nPROCESO FINALIZADO. Total registros: {len(all_data)}")