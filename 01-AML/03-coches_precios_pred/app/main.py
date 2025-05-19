from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import httpx
import os
import pyodbc
import decimal
from dotenv import load_dotenv

# Cargar variables de entorno desde el archivo .env
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Configuración de la conexión a Azure SQL Database
AZURE_SQL_SERVER = os.getenv("AZURE_SQL_SERVER")
AZURE_SQL_DATABASE = os.getenv("AZURE_SQL_DATABASE")
AZURE_SQL_USERNAME = os.getenv("AZURE_SQL_USERNAME")
AZURE_SQL_PASSWORD = os.getenv("AZURE_SQL_PASSWORD")
AZURE_SQL_DRIVER = os.getenv("AZURE_SQL_DRIVER")

def get_coches():
    conn_str = (
        f"DRIVER={AZURE_SQL_DRIVER};"
        f"SERVER={AZURE_SQL_SERVER};"
        f"DATABASE={AZURE_SQL_DATABASE};"
        f"UID={AZURE_SQL_USERNAME};"
        f"PWD={AZURE_SQL_PASSWORD};"
        "Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;"
    )
    with pyodbc.connect(conn_str) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT TOP (1000) [marca]
                ,[modelo]
                ,[version]
                ,[startYear]
                ,[endYear]
                ,[cilindrada]
                ,[cv]
                ,[id_carroceria]
                ,[pf]
                ,[puertas]
                ,[id_combustible]
                ,[matriculacion]
                ,[precio_compra]
                ,[periodoDescripcion]
                ,[Anno]
            FROM [dbo].[ML$coches_tarifa_compra]
        """)
        columns = [column[0] for column in cursor.description]
        coches = [dict(zip(columns, row)) for row in cursor.fetchall()]
    return coches

# Usar get_coches() en vez de la lista COCHES
def get_marcas():
    return sorted(set(c["marca"] for c in get_coches()))

def get_modelos():
    return sorted(set(c["modelo"] for c in get_coches()))

def get_versiones():
    return sorted(set(c["version"] for c in get_coches()))

def get_periodos():
    return sorted(set(c["periodoDescripcion"] for c in get_coches()))

def get_combustibles():
    return sorted(set(c["id_combustible"] for c in get_coches()))

def get_start_years():
    return sorted(set(str(c["startYear"]) for c in get_coches() if c["startYear"] is not None))

def get_end_years():
    return sorted(set(str(c["endYear"]) for c in get_coches() if c["endYear"] is not None))

def get_cilindradas():
    return sorted(set(str(c["cilindrada"]) for c in get_coches() if c["cilindrada"] is not None))

def get_cvs():
    return sorted(set(str(c["cv"]) for c in get_coches() if c["cv"] is not None))

def get_id_carrocerias():
    return sorted(set(str(c["id_carroceria"]) for c in get_coches() if c["id_carroceria"] is not None))

def get_pfs():
    return sorted(set(str(c["pf"]) for c in get_coches() if c["pf"] is not None))

def get_puertas():
    return sorted(set(str(c["puertas"]) for c in get_coches() if c["puertas"] is not None))

def get_matriculaciones():
    return sorted(set(str(c["matriculacion"]) for c in get_coches() if c["matriculacion"] is not None))

def get_annos():
    return sorted(set(str(c["Anno"]) for c in get_coches() if c["Anno"] is not None))

# Endpoint de Azure ML (reemplaza con tu URL real)
AZURE_ML_ENDPOINT = os.getenv("AZURE_ML_ENDPOINT")
AZURE_ML_KEY = os.getenv("AZURE_ML_KEY")

@app.get("/", response_class=HTMLResponse)
async def form_get(request: Request):
    marcas = get_marcas()
    modelos = get_modelos()
    versiones = get_versiones()
    periodos = get_periodos()
    combustibles = get_combustibles()
    start_years = get_start_years()
    end_years = get_end_years()
    cilindradas = get_cilindradas()
    cvs = get_cvs()
    id_carrocerias = get_id_carrocerias()
    pfs = get_pfs()
    puertas = get_puertas()
    matriculaciones = get_matriculaciones()
    annos = get_annos()
    return templates.TemplateResponse("form.html", {
        "request": request,
        "marcas": marcas,
        "modelos": modelos,
        "versiones": versiones,
        "periodos": periodos,
        "combustibles": combustibles,
        "start_years": start_years,
        "end_years": end_years,
        "cilindradas": cilindradas,
        "cvs": cvs,
        "id_carrocerias": id_carrocerias,
        "pfs": pfs,
        "puertas": puertas,
        "matriculaciones": matriculaciones,
        "annos": annos,
        "prediccion": None
    })

@app.post("/predecir", response_class=HTMLResponse)
async def predecir(request: Request,
                  marca: str = Form(...),
                  modelo: str = Form(...),
                  version: str = Form(...),
                  startYear: str = Form(...),
                  endYear: str = Form(...),
                  cilindrada: str = Form(...),
                  cv: str = Form(...),
                  id_carroceria: str = Form(...),
                  pf: str = Form(...),
                  puertas: str = Form(...),
                  id_combustible: str = Form(...),
                  matriculacion: str = Form(...),
                  periodoDescripcion: str = Form(...),
                  Anno: str = Form(...)):
    def parse_int(val):
        try:
            return int(float(val)) if val not in (None, "") else None
        except Exception:
            return None
    columnas = [
        "marca", "modelo", "version", "startYear", "endYear", "cilindrada", "cv", "id_carroceria", "pf", "puertas", "id_combustible", "matriculacion", "periodoDescripcion", "Anno"
    ]
    fila = [
        marca, modelo, version,
        parse_int(startYear),
        parse_int(endYear),
        parse_int(cilindrada),
        parse_int(cv),
        id_carroceria, parse_int(pf),
        parse_int(puertas),
        id_combustible,
        parse_int(matriculacion),
        periodoDescripcion, parse_int(Anno)
    ]
    payload = {
        "input_data": {
            "columns": columnas,
            "index": [0],
            "data": [fila]
        }
    }
    headers = {"Content-Type": "application/json"}
    if AZURE_ML_KEY:
        headers["Authorization"] = f"Bearer {AZURE_ML_KEY}"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(AZURE_ML_ENDPOINT, json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()
            # Ajuste: extraer el valor de la predicción según el formato de respuesta del endpoint
            if isinstance(result, dict) and "result" in result:
                prediccion = result["result"][0] if isinstance(result["result"], list) else result["result"]
            elif isinstance(result, list):
                prediccion = result[0]
            else:
                prediccion = result.get("precio_compra", result)
    except Exception as e:
        prediccion = f"Error: {e}"

    marcas = get_marcas()
    modelos = get_modelos()
    versiones = get_versiones()
    periodos = get_periodos()
    combustibles = get_combustibles()
    start_years = get_start_years()
    end_years = get_end_years()
    cilindradas = get_cilindradas()
    cvs = get_cvs()
    id_carrocerias = get_id_carrocerias()
    pfs = get_pfs()
    puertas = get_puertas()
    matriculaciones = get_matriculaciones()
    annos = get_annos()
    return templates.TemplateResponse("form.html", {"request": request, "marcas": marcas, "modelos": modelos, "versiones": versiones, "periodos": periodos, "combustibles": combustibles, "start_years": start_years, "end_years": end_years, "cilindradas": cilindradas, "cvs": cvs, "id_carrocerias": id_carrocerias, "pfs": pfs, "puertas": puertas, "matriculaciones": matriculaciones, "annos": annos, "prediccion": prediccion})
