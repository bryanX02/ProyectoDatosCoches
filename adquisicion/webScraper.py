# AUTORES:
# BRYAN XAVIER QUILUMBA FARINANGO
# JESÚS MARÍA RODRÍGUEZ GARCÍA
# PABLO MANUEL RODRÍGUEZ SOSA

# Este programa es el encargado de extraer los datos (en crudo) de los vehículos.
# La fuente desde la que se extraen es la web de venta de coches de segunda mano 'AutoHero'
# Para la extracción empleamos las librerías de python:
#   - BeautifulSoup: Encargada de la búsqueda por clases html de los datos de los coches en la web
#   - Selenium: Encargada de recorrer e interactuar con la web realizando scrolls y dando clícks
# Para la ejecución del programa puede ser necesario: pip install selenium

# Librerías necesarias
from selenium.webdriver.common.by import By
from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import requests
import time


# Función que devuelve un drive para el scraping con las configuraciones aplicadas
def web_driver():
    options = webdriver.ChromeOptions()
    options.add_argument("--no-sandbox")  # Deshabilita el limitador
    options.add_argument("--headless")  # Deshabilita la ventana y se realiza en un segundo plano
    options.add_argument("--disable-gpu")  # Deshabilita la GPU para evitar problemas en algunas webs
    options.add_argument("--window-size=1920, 2160")  # Tamaño de la ventana (en segundo plano)
    options.add_argument("--disable-dev-shm-usage")  # Deshabilita los archivos temporales
    return webdriver.Chrome(options=options)


# Función que realiza la interacción con la web (web scraper) realizando scrolls y aceptando las cookies
# A su vez se aplica el web scraping obteniendo los datos de los vehículos y devolviéndolos al final
def scraper(driver, num_scroll):
    # Se emplea la librería Selenium para realizar el scraper (hacer scroll y clíck)
    # Y para extraer la información de los coches se emplea BeautifulShop

    # Tiempo de espera entre scrolls, para que cargue bien la página
    scroll_pause_time = 1

    # Conjunto (sin repeticiones) que almacena los vehículos que se vayan encontrando
    vehiculos = set()

    # Al ingresar a la web saltan las cookies. Y estas no dejas hacer scroll
    try:
        # Se espera a que salga la ventana de aceptar cookies
        time.sleep(3)

        # Se busca el botón y se le da clíck
        btn = driver.find_element(By.CLASS_NAME, "button___2R6qU.size-sm___3TKQS.default___1FRAY")
        btn.click()  # Se aceptan las cookies

    except:
        # Si no salen puede ser debido a que el navegador ya tiene en su historial las cookies
        # También es posible que sea necesario aumentar time.sleep para que le dé tiempo a salir
        print("No saltaron las cookies")

    # Web Scraping (aumentar n para extraer más coches)
    n = num_scroll  # numero de scrolls
    i = 1
    screen_height = driver.execute_script("return window.screen.height;")

    while i <= n:

        # Se realiza el scroll para la siguiente página
        driver.execute_script(f"window.scrollTo(0, {screen_height * i})")
        time.sleep(scroll_pause_time)  # Tiempo de carga de la siguiente página

        scraperSoup = BeautifulSoup(driver.page_source, "html.parser")
        # Se sacan los coches de la página actua
        pagVehiculos = scraperSoup.find_all(
            "a", class_="link___2Maxt color-inherit___SyKXO decoration-none___1IENu"
        )
        # Se añaden al conjunto
        for vehiculo in pagVehiculos:
            vehiculos.add(vehiculo)
        i += 1

    # Saca captura de la ultima instancia
    # driver.save_screenshot(f'screen.png')
    # Se cierra el driver (el navegador que estaba en segundo plano)
    driver.quit()

    return vehiculos


# Funciones que interpretan el html para recopilar la información
def agregar_autohero(link):
    return "https://www.autohero.com" + link


def obtener_info_vehiculo_1(vehiculo):
    infoExtra = vehiculo.get("href")
    nombre_vehiculo = vehiculo.find("h2", class_="title___1TYYE").text.strip()
    precioFinan = vehiculo.find("div", class_="price___uwAkj").text.strip()
    modelo = vehiculo.find("h3", class_="subtitle___1ZA66").text.strip()
    link = agregar_autohero(infoExtra)

    # Realizar solicitud al enlace principal
    response_vehiculo = requests.get(link)
    html_vehiculo = response_vehiculo.text
    soup_vehiculo = BeautifulSoup(html_vehiculo, "html.parser")

    # Extraer información adicional del enlace
    descripcion_vehiculo = soup_vehiculo.find_all("div", class_="listItemText___uoGhx")
    specs = soup_vehiculo.find_all("div", class_="item___3Uae0")

    # Inicializar variables
    primera_matriculacion = (
        kilometraje
    ) = carburante = transmision = potencia = traccion = None
    precio2 = soup_vehiculo.find("div", class_="vehiclePriceContainer___jwSn_")
    if precio2:
        precio = precio2.find("p", class_="vehiclePrice___1uUmJ").text.strip()
    else:
        precio = precioFinan

    # Iterar sobre la información extraída
    for elemento in descripcion_vehiculo:
        titulo = elemento.find("h5", class_="listItemTitle___2CQBv").text.strip()
        valor = elemento.find("span", class_="listItemValue___1IWSE").text.strip()

        # Asignar valores a las variables correspondientes
        if titulo == "Primera matriculación":
            primera_matriculacion = valor
        elif titulo == "Kilometraje":
            kilometraje = valor
        elif titulo == "Carburante":
            carburante = valor
        elif titulo == "Transmisión":
            transmision = valor
        elif titulo == "Potencia":
            potencia = valor
        elif titulo == "Tracción":
            traccion = valor

    # Iterar sobre la información extra
    for elemento in specs:
        try:
            titulo = elemento.find("div", class_="itemTitle___3GH8k").text.strip()
        except:
            continue
        try:
            valor = elemento.find("div", class_="body___2uId6").text.strip()
        except:
            # Si no se encuentra el elemento 'body___2uId6', se asigna None o se continua con el siguiente elemento
            valor = np.nan

        # Asignar valores a las variables correspondientes
        if titulo == "Cilindrada":
            cilindrada = valor
        elif titulo == "Coche accidentado y reparado":
            coche_accidentado = valor

    # Imprimir o almacenar la información según sea necesario
    # print(f"Nombre del vehículo: {nombre_vehiculo}")
    # print(f"Precio: {precio}")
    # print(f"Modelo: {modelo}")
    # print(f"Enlace Autohero: {link}")
    # print(f"Primera matriculación: {primera_matriculacion}")
    # print(f"Kilometraje: {kilometraje}")
    # print(f"Carburante: {carburante}")
    # print(f"Transmisión: {transmision}")
    # print(f"Potencia: {potencia}")
    # print(f"Tracción: {traccion}")
    # print("\n")

    # Crear un DataFrame con una sola fila
    data = {
        "Modelo": nombre_vehiculo,
        "Precio(€)": precio,
        "Adicional": modelo,
        #'Enlace Autohero': link,
        "Primera matriculación": primera_matriculacion,
        "Kilometraje(Km)": kilometraje,
        "Carburante": carburante,
        "Transmisión": transmision,
        "Potencia(Cv)": potencia,
        "Tracción": traccion,
        "Cilindrada(Cc)": cilindrada,
        "Accidentado": coche_accidentado,
    }
    df = pd.DataFrame([data])
    # Añadir la fecha del dato recopilado
    return df


# Ejecución
def main():
    # Fuente de los datos
    url = "https://www.autohero.com/es/search/"

    # Driver (navegador) que se empleará
    driver = web_driver()
    driver.get(url)

    # Llamada a la función que aplica el web scraping junto al scraper para obtener los vehículos
    vehiculos = scraper(driver, 150)
    print(f"Se han encontrado {len(vehiculos)} vehiculos:")

    # Los datos de los vehiculos en html se pasan a un dataframe, todavía en crudos
    df_resultado = pd.DataFrame()
    for vehiculo in vehiculos:
        try:
            df_resultado = pd.concat(
                [df_resultado, obtener_info_vehiculo_1(vehiculo)], ignore_index=True
            )
        except:
            print(
                f"Proximamente: {vehiculo.get('href')}"
            )  # solucionado problema comillas

    print(df_resultado)

    # El data frame se exporta a un archivo json
    df_resultado.to_json("datosCrudos.json", "index")


if __name__ == "__main__":
    main()
