from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import onnxruntime as ort
import numpy as np
import base64
import io
from PIL import Image

app = FastAPI(title="Landslide Detection API - La Convención")

# Habilitar CORS para permitir peticiones desde tu app estática en Hugging Face
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar los modelos ONNX desde la misma carpeta
# onnxruntime automáticamente detectará los archivos .data asociados
print("Cargando modelos ONNX...")
unet_session = ort.InferenceSession("unet_landslide.onnx", providers=["CPUExecutionProvider"])
deeplab_session = ort.InferenceSession("deeplabv3_landslide.onnx", providers=["CPUExecutionProvider"])
print("Modelos cargados correctamente.")

class ROIRequest(BaseModel):
    lat: float
    lon: float
    date: str

@app.get("/")
def read_root():
    return {"message": "API de Detección de Deslizamientos Activa"}

@app.post("/predict")
def predict_landslide(request: ROIRequest):
    try:
        # AQUI SE INTEGRARÁ 'cubo' PARA DESCARGAR SATÉLITE REAL.
        # Por ahora, simularemos la matriz (1, 6, 256, 256) que enviaría cubo.
        dummy_tensor = np.random.randn(1, 6, 256, 256).astype(np.float32)
        
        # Obtener el nombre de la capa de entrada del ONNX
        input_name = unet_session.get_inputs()[0].name
        
        # Realizar la inferencia con U-Net
        prediction = unet_session.run(None, {input_name: dummy_tensor})[0]
        
        # Aplicar Sigmoide y binarizar (simulando detección)
        sigmoid_pred = 1 / (1 + np.exp(-prediction))
        mask = (sigmoid_pred > 0.5).astype(np.uint8).squeeze()
        
        # Convertir la matriz numpy (0 y 1) a una imagen de mapa de bits (0 y 255)
        # y pintarla en rojo transparente para el mapa
        mask_visual = np.zeros((256, 256, 4), dtype=np.uint8)
        mask_visual[mask == 1] = [255, 0, 0, 150] # RGBA: Rojo con transparencia
        
        # Codificar en Base64
        img = Image.fromarray(mask_visual, mode="RGBA")
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        # Simulamos los límites geográficos del parche extraído
        bbox = [[request.lat - 0.02, request.lon - 0.02], [request.lat + 0.02, request.lon + 0.02]]
        
        return {
            "status": "success", 
            "mask_base64": img_base64, 
            "bounds": bbox,
            "model": "U-Net"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}g