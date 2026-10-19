from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import onnxruntime as ort
import numpy as np
import base64
import io
from PIL import Image
import cubo
import xarray as xr

app = FastAPI(title="Landslide Detection API - La Convención")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Cargando modelos ONNX...")
unet_session = ort.InferenceSession("unet_landslide.onnx", providers=["CPUExecutionProvider"])
deeplab_session = ort.InferenceSession("deeplabv3_landslide.onnx", providers=["CPUExecutionProvider"])
print("Modelos cargados correctamente.")

class ROIRequest(BaseModel):
    lat: float
    lon: float
    date: str

def fetch_real_satellite_tensor(lat: float, lon: float, date_str: str, edge: int = 256):
    """Descarga Sentinel-2 y DEM, calcula la pendiente y concatena en 6 canales."""
    
    # 1. Extraer Sentinel-2 (B01, B02, B03, B04)
    # cubo devuelve dimensiones (time, band, y, x)
    s2 = cubo.create(
        lat=lat,
        lon=lon,
        collection="sentinel-2-l2a",
        bands=["B01", "B02", "B03", "B04"],
        start_date=date_str,
        end_date=date_str,
        edge_size=edge,
        resolution=10
    )
    # Normalización estándar para imágenes ópticas (0 a 1)
    s2_np = s2.values / 10000.0 
    
    # 2. Extraer DEM (Usamos cop-dem-glo-30 como proveedor STAC estándar de elevación)
    dem = cubo.create(
        lat=lat,
        lon=lon,
        collection="cop-dem-glo-30",
        bands=["data"], # Elevación pura
        edge_size=edge,
        resolution=10
    )
    dem_np = dem.values # Dimensiones (1, 1, 256, 256)
    
    # 3. Calcular la Pendiente (Slope) a partir del DEM usando gradientes espaciales
    # Eliminamos las dimensiones sobrantes temporalmente para calcular el gradiente
    dem_2d = dem_np.squeeze() 
    dy, dx = np.gradient(dem_2d)
    slope_2d = np.sqrt(dx**2 + dy**2)
    # Restaurar dimensiones para la concatenación (1, 1, 256, 256)
    slope_np = np.expand_dims(np.expand_dims(slope_2d, axis=0), axis=0) 
    
    # 4. Concatenar S2(4), Slope(1), DEM(1) -> Tensor Final (1, 6, 256, 256)
    tensor_input = np.concatenate([s2_np, slope_np, dem_np], axis=1).astype(np.float32)
    
    # 5. Extraer coordenadas límite (Bounding Box) reales para proyectar en Leaflet
    bbox = [
        [float(s2.y.min()), float(s2.x.min())],
        [float(s2.y.max()), float(s2.x.max())]
    ]
    
    return tensor_input, bbox

@app.get("/")
def read_root():
    return {"message": "API de Detección de Deslizamientos Activa"}

@app.post("/predict")
def predict_landslide(request: ROIRequest):
    try:
        # Llamada a la función real de extracción
        tensor_input, bbox = fetch_real_satellite_tensor(
            lat=request.lat, 
            lon=request.lon, 
            date_str=request.date
        )
        
        input_name = unet_session.get_inputs()[0].name
        
        # Inferencia con U-Net (tensor de 6 canales reales)
        prediction = unet_session.run(None, {input_name: tensor_input})[0]
        
        # Sigmoide y binarización
        sigmoid_pred = 1 / (1 + np.exp(-prediction))
        mask = (sigmoid_pred > 0.5).astype(np.uint8).squeeze()
        
        # Capa roja transparente para zonas de peligro
        mask_visual = np.zeros((256, 256, 4), dtype=np.uint8)
        mask_visual[mask == 1] = [255, 0, 0, 150] 
        
        # Conversión a Base64
        img = Image.fromarray(mask_visual, mode="RGBA")
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return {
            "status": "success", 
            "mask_base64": img_base64, 
            "bounds": bbox,
            "model": "U-Net"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
