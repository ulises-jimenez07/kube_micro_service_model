# FastAPI conversion of flask_elector.py
import asyncio
import json
import time
import aiohttp
import logging
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import socket

# Set up logging
FORMAT = "[%(asctime)-15s][%(levelname)-8s]%(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Starting Elector Service")

# Constants
DURACION_POR_LLAMADA = 5
DURACION_TOTAL = 10

app = FastAPI(title="Elector API")


class IrisData(BaseModel):
    s_l: float
    s_w: float
    p_l: float
    p_w: float


async def do_request(
    url: str, data: Dict[str, Any], timeout_peticion: float, session: aiohttp.ClientSession
):
    logger.info(f"[do_request][{url}] Calling web")
    try:
        async with session.post(url, timeout=timeout_peticion, json=data) as resp:
            respuesta = await resp.text()
            logger.info(f"[do_request][{url}] Returns result:{respuesta}")
            return [respuesta, url]  # esta respuesta se devulve cuando ya ha terminado la petición
    except asyncio.TimeoutError as ex:
        logger.warning(f"[do_request][{url}]Timeout captured:{ex}")
        return None
    except Exception as ex:
        logger.error(f"[do_request][{url}]Exception:{ex}")
        return None


async def esperar_respuestas(modelos):
    resultados = []
    tiempo_inicial = time.time()
    for completado in asyncio.as_completed(modelos):
        respuesta = await completado
        print(respuesta)
        resultados.append(respuesta)
        duracion = time.time() - tiempo_inicial
        if duracion > DURACION_TOTAL:
            logger.error("Se ha sobrepasado el tiempo de espera de respuestas")
            break
    return resultados


# Determine if we're running in containers or locally
def is_container_env():
    try:
        # Try to resolve the canary hostname to see if we're in a container environment
        socket.gethostbyname("canary")
        return True
    except socket.gaierror:
        return False


# Get the appropriate URLs based on environment
def get_service_urls():
    if is_container_env():
        return {"canary": "http://canary:5001", "model": "http://model:5000"}
    else:
        return {"canary": "http://localhost:5001", "model": "http://localhost:5000"}


async def llamar_a_modelos(session, data):
    modelos_llamados = []

    # Get appropriate URLs based on environment
    urls = get_service_urls()

    logger.info(f"Running with URLs: canary={urls['canary']}, model={urls['model']}")

    modelos_llamados.append(
        do_request(f"{urls['canary']}/predict", data, DURACION_POR_LLAMADA, session)
    )
    modelos_llamados.append(
        do_request(f"{urls['model']}/predict", data, DURACION_POR_LLAMADA, session)
    )
    return modelos_llamados


def trata_resultados(resultados):
    """Elije qué llamada responde"""
    respuesta = "Sin resultado de modelos"
    urls = get_service_urls()
    model_url = f"{urls['model']}/predict"

    logger.info(f"Checking for response from {model_url}")

    for resultado in resultados:
        logger.info(f"Processing result from {resultado[1] if resultado else 'None'}")
        if resultado is not None:
            if resultado[1] == model_url:
                respuesta = resultado[0]

    # If we didn't get a response from the main model, use the first valid response
    if respuesta == "Sin resultado de modelos":
        for resultado in resultados:
            if resultado is not None:
                respuesta = resultado[0]
                break

    return respuesta


async def get_datos(data):
    resultados_tratados = []
    async with aiohttp.ClientSession() as session:
        logger.info("Llamando a los modelos")
        modelos_llamados = await llamar_a_modelos(session, data)

        logger.info("Esperando a los modelos")
        resultados = await esperar_respuestas(modelos_llamados)

        logger.info("Trata resultados")
        resultados_tratados = trata_resultados(resultados)

    return resultados_tratados


@app.post("/predict")
async def predict(data: IrisData):
    try:
        data_dict = data.dict()
        respuesta = await get_datos(data_dict)
        return json.loads(respuesta)
    except Exception as ex:
        logger.error(f"Error in predict endpoint: {str(ex)}")
        raise HTTPException(status_code=400, detail=str(ex))


if __name__ == "__main__":
    import uvicorn

    logger.info("Elector starting")
    uvicorn.run(app, host="0.0.0.0", port=5002)
