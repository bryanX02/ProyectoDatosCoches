# Instalaciones necesarias

#pip install selenium

# Librerias necesarias
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
import numpy as np
import time
import pandas as pd
from selenium.webdriver.common.by import By

# Driver para el scraper (cambiar dependiendo del equipo)
def web_driver():
    options = webdriver.ChromeOptions()
    options.add_argument("--verbose")
    options.add_argument('--no-sandbox')
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument("--window-size=1920, 2160")
    options.add_argument('--disable-dev-shm-usage')
    #proxy = "12.345.67.890:1234"
    #options.add_argument("--proxy-server=%s" % proxy)
    return webdriver.Chrome(options=options)

### Scraping
def scraper(driver, num_scroll):
    # Se emplea la librería Sellenium para realizar el scraper (hacer scroll y click)
    # Y para extraer la información de los coches se emple BeautifulShop

    # Tiempo de espera entre scrolls, para que carge bien la pagina
    scroll_pause_time = 1

    # Contadores y almacenadores
    numVehiculos = 0
    pagVehiculos = []
    totalVehiculos = set()

    # Al ingresar a la web saltan las cookies. Y estas no dejas hacer scroll
    try:
        # Se espera a que salgan (si no salen -> aumentar el tiempo)
        time.sleep(3)
        boton = driver.find_element(By.CLASS_NAME, "button___2R6qU.size-sm___3TKQS.default___1FRAY")
        boton.click()  # Se aceptan las cookies
    except:
        print("No saltaron las cookies")

    # Web Scraping (aumentar n para extraer más coches)
    n = num_scroll  # numero de scrolls
    i = 1

    screen_height = driver.execute_script("return window.screen.height;")
    while i <= n:

        # Se realiza el scroll para la siguiente página
        driver.execute_script(f"window.scrollTo(0, {screen_height * i})")
        time.sleep(scroll_pause_time)  # Tiempo de carga de la siguiente página
        scraperSoup = BeautifulSoup(driver.page_source, 'html.parser')
        # Se sacan los coches de la página actua
        pagVehiculos = scraperSoup.find_all('a', class_='link___2Maxt color-inherit___SyKXO decoration-none___1IENu')
        for vehiculo in pagVehiculos:
            totalVehiculos.add(vehiculo)
        i += 1

    # Saca captura de la ultima instancial
    driver.save_screenshot(f'screen.png')
    driver.quit()
    return totalVehiculos

### Fuente de los datos

### Funciones que interpretan el html para recopilar la información
def agregar_autohero(link):
    return "https://www.autohero.com" + link

def obtener_info_vehiculo_1(vehiculo):
    infoExtra = vehiculo.get('href')
    nombre_vehiculo = vehiculo.find('h2', class_='title___1TYYE').text.strip()
    precioFinan = vehiculo.find('div', class_='price___uwAkj').text.strip()
    modelo = vehiculo.find('h3', class_='subtitle___1ZA66').text.strip()
    link = agregar_autohero(infoExtra)

    # Realizar solicitud al enlace principal
    response_vehiculo = requests.get(link)
    html_vehiculo = response_vehiculo.text
    soup_vehiculo = BeautifulSoup(html_vehiculo, 'html.parser')

    # Extraer información adicional del enlace
    descripcion_vehiculo = soup_vehiculo.find_all('div', class_='listItemText___uoGhx')
    specs = soup_vehiculo.find_all('div', class_='item___3Uae0')

    # Inicializar variables
    primera_matriculacion = kilometraje = carburante = transmision = potencia = traccion = None
    precio2 = soup_vehiculo.find('div', class_='vehiclePriceContainer___jwSn_')
    if precio2:
        precio = precio2.find('p', class_='vehiclePrice___1uUmJ').text.strip()
    else:
        precio = precioFinan

    # Iterar sobre la información extraída
    for elemento in descripcion_vehiculo:
        titulo = elemento.find('h5', class_='listItemTitle___2CQBv').text.strip()
        valor = elemento.find('span', class_='listItemValue___1IWSE').text.strip()

        # Asignar valores a las variables correspondientes
        if titulo == 'Primera matriculación':
            primera_matriculacion = valor
        elif titulo == 'Kilometraje':
            kilometraje = valor
        elif titulo == 'Carburante':
            carburante = valor
        elif titulo == 'Transmisión':
            transmision = valor
        elif titulo == 'Potencia':
            potencia = valor
        elif titulo == 'Tracción':
            traccion = valor

    # Iterar sobre la información extra
    for elemento in specs:
        try:
            titulo = elemento.find('div', class_='itemTitle___3GH8k').text.strip()
        except:
            continue
        try:
            valor = elemento.find('div', class_='body___2uId6').text.strip()
        except:
            # Si no se encuentra el elemento 'body___2uId6', se asigna None o se continua con el siguiente elemento
            valor = np.nan

        # Asignar valores a las variables correspondientes
        if titulo == 'Cilindrada':
            cilindrada = valor
        elif titulo == 'Coche accidentado y reparado':
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
        'Nombre del vehículo': nombre_vehiculo,
        'Precio(€)': precio,
        'Modelo': modelo,
        # 'Enlace Autohero': link,
        'Primera matriculación': primera_matriculacion,
        'Kilometraje(Km)': kilometraje,
        'Carburante': carburante,
        'Transmisión': transmision,
        'Potencia(Cv)': potencia,
        'Tracción': traccion,
        'Cilindrada(Cc)': cilindrada,
        'Accidentado': coche_accidentado
    }
    df = pd.DataFrame([data])

    return df


# Ejecución
def main():
    #driver =  webdriver.Chrome()
    url = 'https://www.autohero.com/es/search/'
    driver = web_driver()
    driver.get(url)
    totalVehiculos = scraper(driver, 2)
    print(f"Se han encontrado {len(totalVehiculos)} vehiculos:")

    df_resultado = pd.DataFrame()
    for vehiculo in totalVehiculos:
        df_resultado = pd.concat([df_resultado, obtener_info_vehiculo_1(vehiculo)], ignore_index=True)

    print(df_resultado)
    df_resultado.to_json("datosCrudos.json", "index")

if __name__ == "__main__":
    main()


# df_resultado['Marca'] = df_resultado['Modelo'].str.split().str[0]
#
# # Eliminar la marca de la columna 'Modelo'
# df_resultado['Modelo'] = df_resultado.apply(lambda row: row['Modelo'].replace(row['Marca'], ''), axis=1)
# # Eliminar espacios adicionales al inicio de la columna 'Modelo'
# df_resultado['Modelo'] = df_resultado['Modelo'].str.strip()
#
# column_order = ['Marca'] + [col for col in df_resultado.columns if col != 'Marca']
#
# df_resultado['Precio(€)'] = df_resultado['Precio(€)'].replace({'€': ''}, regex=True)
# df_resultado['Precio(€)'] = df_resultado['Precio(€)'].replace({'\\.': ''}, regex=True).astype(int)
# #df_resultado.rename(columns={'Precio': 'Precio(€)'}, inplace=True)
#
# df_resultado['Primera matriculación'] = pd.to_datetime(df_resultado['Primera matriculación'], format='%d.%m.%Y', errors='coerce')
#
# df_resultado['Kilometraje(Km)'] = df_resultado['Kilometraje(Km)'].replace({'km': ''}, regex=True)
# df_resultado['Kilometraje(Km)'] = df_resultado['Kilometraje(Km)'].replace({'\\.': ''}, regex=True).astype(int)
# #df_resultado.rename(columns={'Kilometraje': 'Kilometraje(KM)'}, inplace=True)
#
# df_resultado['Transmisión'] = df_resultado['Transmisión'].replace({'Cambio tipo': ''}, regex=True)
# df_resultado['Transmisión'] = df_resultado['Transmisión'].replace({'automático': 'automatico'}, regex=True)
#
# df_resultado['Potencia(Cv)'] = df_resultado['Potencia(Cv)'].str.extract('(\d+)', expand=False).astype(int)
# #df_resultado.rename(columns={'Potencia': 'Potencia(CV)'}, inplace=True)
#
# df_resultado['Tracción'] = df_resultado['Tracción'].replace({'Tracción ': ''}, regex=True)
# df_resultado['Tracción'] = df_resultado['Tracción'].replace({'total (4x4)': 'total'}, regex=True)
#
#
#
# # Utiliza expresiones regulares para extraer el valor numérico
# df_resultado['Cilindrada(Cc)'] = df_resultado['Cilindrada(Cc)'].str.extract('(\d+)', expand=False).astype(int)
#
# df_resultado = df_resultado[column_order]


# Utiliza expresiones regulares para extraer el valor numérico
#df_resultado['Cilindrada'] = df_resultado['Cilindrada'].str.extract('(\d+)', expand=False).astype(int)

#df_resultado = df_resultado[column_order]

#current_date = datetime.now()
#df_resultado['Edad(Años)'] = (current_date - df_resultado['Primera matriculación']).dt.days // 365

#print(df_resultado.dtypes)

#display(df_resultado)


