# FastAPI version of flask_elector.py
import asyncio
import json
import time
import aiohttp
import logging
import os
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Dict, Any, List, Optional

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


# Determine if we're running in Docker
def is_docker_env():
    # In Docker Compose, check if environment shows we're in Docker
    # This is a simple check - in Docker, the hostname is usually the container ID
    return os.path.exists("/.dockerenv")


# Get the appropriate URLs based on environment
def get_service_urls():
    if is_docker_env():
        # In Docker Compose, use the service names
        return {"canary": "http://canary:5001", "model": "http://model:5000"}
    else:
        # For local development
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

    # First, check if we have any responses at all
    valid_responses = [r for r in resultados if r is not None]
    if not valid_responses:
        logger.warning("No valid responses received from any model")
        return respuesta

    # Look for the main model response
    for resultado in resultados:
        if resultado is not None:
            logger.info(f"Processing result from {resultado[1]}")
            if resultado[1] == model_url:
                respuesta = resultado[0]
                logger.info("Using main model response")
                break

    # If we didn't get a response from the main model, use the first valid response
    if respuesta == "Sin resultado de modelos" and valid_responses:
        respuesta = valid_responses[0][0]
        logger.info(f"Falling back to response from {valid_responses[0][1]}")

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
        # Handle the case where we got no valid responses
        if respuesta == "Sin resultado de modelos":
            raise HTTPException(status_code=503, detail="No models available for prediction")
        return json.loads(respuesta)
    except json.JSONDecodeError as ex:
        logger.error(f"Error decoding JSON response: {str(ex)}")
        raise HTTPException(status_code=500, detail="Invalid response from models")
    except Exception as ex:
        logger.error(f"Error in predict endpoint: {str(ex)}")
        raise HTTPException(status_code=400, detail=str(ex))


@app.get("/health")
async def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    logger.info("Elector starting")
    uvicorn.run(app, host="0.0.0.0", port=5002)
