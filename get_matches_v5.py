# get_matches_v5.py
import re # Librería para identificar patrones en textos/URLs
import time # Para gestionar pausas entre peticiones
import json # Para guardar los datos en JSON
import os #Para poder crear carpetas
from typing import List, Dict # Para saber la respuesta de las funciones

import requests #Librería principal para peticiones HTTP
from bs4 import BeautifulSoup #Para parsear y extraer datos de archivos HTML
import pandas as pd #Para manejar bases de datos y exportarlo a csv

from selenium import webdriver #Motor de automatización del navegador
from selenium.webdriver.chrome.service import Service #Configuración del serviicio del driver de Chrome
from selenium.webdriver.common.by import By #Para localizar elementos en la web
from selenium.webdriver.chrome.options import Options #Para configurar opciones del navegador
from selenium.webdriver.support.ui import WebDriverWait #Para implementar esperas en el navegador
from selenium.webdriver.support import expected_conditions as EC #Condiciones en las que selenium espera a un elemento

# CONFIGURACIÓN:
BASE = "https://baloncestoenvivo.feb.es"
OUTDIR = "feb_data" #carpeta de destino de los datos
os.makedirs(OUTDIR, exist_ok=True) #solamente se crea si la carpeta no existe, para evitar errores

#ID de cada una de las ligas en la web de FEB, encontrados analizando los url.
LEAGUES = {
    "LF_ENDESA": 4,
    "LF_CHALLENGE": 67,
    "LF2": 9
}

#Nombre de cada una de las ligas en la web de FEB, encontrados analizando los url.
LEAGUE_NAMES = {
    "LF_ENDESA": "lfendesa",
    "LF_CHALLENGE": "lfchallenge",
    "LF2": "lf2"
}

#Temporadas de las cuales se desea obtener datos de los partidos
SEASONS = [2024, 2023, 2022, 2021]

SELENIUM_TIMEOUT = 12 #Tiempo máximo de espera para cargar un elementode Selenium
HEADLESS = True #El navegador no se abrirá visualmente
CHROMEDRIVER_PATH = r"C:\Users\feder\Downloads\CHROMEDRIVER\chromedriver.exe"


# FUNCIONES AUXILIARES
def _extract_matches_from_html(html: str) -> List[Dict]:
    """Extracción de la información de los partidos del HTML con BeautifulSoup"""
    soup = BeautifulSoup(html, "html.parser")
    matches = []
    #Búsqueda del ID del partido en el enlace
    for a in soup.find_all("a", href=True):
        href = a["href"]
        link = href if href.startswith("http") else BASE + href if href.startswith("/") else BASE + "/" + href
        m = re.search(r'[Pp]artido\.aspx\?p=(\d+)', href) #Extracción del enlace del partido para conseguir el ID
        if m:
            pid = int(m.group(1))
            fecha = local = visitante = None
            tr = a.find_parent("tr")
            if tr:
                tds = tr.find_all("td")
                if len(tds) >= 3: #Extracción de equipos y resultado
                    local = tds[0].get_text(strip=True)
                    marcador = tds[1].get_text(strip=True)
                    visitante = tds[2].get_text(strip=True)
            matches.append({ #Extracción de datos generales del partido
                "id_partido": pid,
                "local": local,
                "marcador": marcador,
                "visitante": visitante,
                "link": link
            })

    # Deduplicar
    unique = {it["id_partido"]: it for it in matches if it["id_partido"]}
    return list(unique.values())


def _selenium_extract_matches(friendly_url: str) -> List[Dict]:
    opts = Options() #Configuración de opciones de Chrome
    if HEADLESS:
        opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")

    driver = webdriver.Chrome(service=Service(CHROMEDRIVER_PATH), options=opts) #Iniciamos el navegador con el Driver y las opciones definidas
    try:
        driver.get(friendly_url)
        wait = WebDriverWait(driver, SELENIUM_TIMEOUT)
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "a[href*='Partido.aspx']")))
        matches = _extract_matches_from_html(driver.page_source)
    finally:
        driver.quit()

    # Deduplicar por id
    unique = {it["id_partido"]: it for it in matches if it["id_partido"]}
    return list(unique.values())


# FUNCIÓN PRINCIPAL:
def get_calendar_matches(league_name: str, league_id: int, season: int, use_selenium_if_empty: bool = True) -> List[Dict]:
    #""Función que decide si usar requests o selenium para obtener los partidos""
    nm = LEAGUE_NAMES.get(league_name, league_name.lower())
    #Construcción del URL de la API de calendario.aspx
    params_url = f"{BASE}/calendario.aspx"
    params = {"g": league_id, "t": season, "nm": nm}

    try:
        r = requests.get(params_url, params=params, timeout=15)
        r.raise_for_status()
        html = r.text
    except:
        html = None

    matches = _extract_matches_from_html(html) if html else []
    #Si no encuentra partidos en el html, utiliza Selenium
    if not matches and use_selenium_if_empty:
        friendly = f"{BASE}/calendario/{nm}/{league_id}/{season}"
        print(f"[INFO] No matches found via requests for {league_name} {season} → using Selenium on {friendly}")
        try:
            matches = _selenium_extract_matches(friendly)
        except Exception as e:
            print("[WARN] Selenium extraction failed:", e)
            matches = []

    return sorted(matches, key=lambda x: x["id_partido"] or 0) # Devuelve la lista ordenada de partidos por ID


# MAIN SCRIPT
if __name__ == "__main__":
    all_matches = [] #Lista donde se van a almacenar los partidos de todas las ligas
    for league_name, league_id in LEAGUES.items(): #Bucle para recorrer cada una de las ligas y temporadas
        for season in SEASONS:
            print(f"Obteniendo partidos {league_name} temporada {season} ...")
            matches = get_calendar_matches(league_name, league_id, season)
            print(f"  Encontrados {len(matches)} partidos ")

            for m in matches: #Añadimos datos de liga y temporada a cada partido localizado
                m["liga_clave"] = league_name
                m["liga_id"] = league_id
                m["temporada"] = season
            all_matches.extend(matches)

    out_json = os.path.join(OUTDIR, "matches_all.json") #Guardado de datos en JSON
    out_csv = os.path.join(OUTDIR, "matches_all.csv") #Guardado de datos en CSV
    pd.DataFrame(all_matches).to_csv(out_csv, index=False, encoding="utf-8") #UTF-8 para que se reconozcan las tildes en el csv
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_matches, f, ensure_ascii=False, indent=2)

    print("Guardados:", out_json, out_csv)
