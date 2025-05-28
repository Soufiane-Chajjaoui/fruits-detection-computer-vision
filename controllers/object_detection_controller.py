from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from typing import List, Dict, Any, Union, Optional
from pydantic import BaseModel, Field
import os
import shutil
from ultralytics import YOLO
import cv2
import numpy as np
import uuid
from datetime import datetime

# Modèles de données pour la documentation
class ImageDetectionResponse(BaseModel):
    """
    Réponse de l'API de détection d'objets dans une image.
    
    Attributes:
        filename: Nom du fichier original
        content_type: Type MIME du fichier
        saved_at: Chemin où le fichier original est sauvegardé
        result_image: Chemin vers l'image avec les annotations
        detections: Liste des objets détectés avec leurs propriétés
        detection_count: Nombre total d'objets détectés
    """
    filename: str = Field(description="Nom du fichier original")
    content_type: str = Field(description="Type MIME du fichier")
    saved_at: str = Field(description="Chemin où le fichier original est sauvegardé")
    result_image: str = Field(description="Chemin vers l'image avec les annotations")
    detections: List[Dict[str, Any]] = Field(description="Liste des objets détectés")
    detection_count: int = Field(description="Nombre total d'objets détectés")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "filename": "example.jpg",
                "content_type": "image/jpeg",
                "saved_at": "uploads/example.jpg",
                "result_image": "uploads/result_example.jpg",
                "detections": [
                    {
                        "class": "person",
                        "confidence": 0.95,
                        "bbox": [10.5, 20.3, 200.1, 300.7]
                    },
                    {
                        "class": "car",
                        "confidence": 0.87,
                        "bbox": [150.2, 230.5, 350.8, 450.3]
                    }
                ],
                "detection_count": 2
            }
        }
    }

router = APIRouter(tags=["object-detection"])

# Créer les dossiers nécessaires
os.makedirs("uploads", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Charger le modèle YOLO
model = YOLO("model-detection.pt")  # Utilise le modèle YOLOv8 nano préentraîné

@router.post(
    "/detect-image/", 
    response_model=ImageDetectionResponse,
    summary="Détecter des objets dans une image",
    description="""
    Téléchargez une image et détectez les objets présents en utilisant YOLOv8.
    
    L'API retourne:
    - Les informations sur le fichier téléchargé
    - Le chemin vers l'image résultante avec les annotations
    - La liste des objets détectés avec leur classe, confiance et position
    - Le nombre total d'objets détectés
    """
)
async def detect_image(file: UploadFile = File(...)):
    """
    Détecte les objets dans une image téléchargée en utilisant YOLOv8.
    
    Args:
        file: Le fichier image à analyser
        
    Returns:
        Un objet ImageDetectionResponse contenant les résultats de la détection
        
    Raises:
        HTTPException: Si le fichier n'est pas une image
    """
    # Vérifier que le fichier est une image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Le fichier doit être une image")
    
    # Chemin où le fichier sera sauvegardé
    file_path = os.path.join("uploads", file.filename)
    
    # Sauvegarder le fichier
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Exécuter la détection avec YOLO
    results = model(file_path)
    
    # Extraire les résultats
    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = box.conf[0].item()
            class_id = int(box.cls[0].item())
            class_name = result.names[class_id]
            
            detections.append({
                "class": class_name,
                "confidence": round(confidence, 3),
                "bbox": [round(x, 2) for x in [x1, y1, x2, y2]]
            })
    
    # Sauvegarder l'image avec les annotations
    result_image_path = os.path.join("results", f"result_{file.filename}")
    result = results[0].plot()
    cv2.imwrite(result_image_path, result)
    
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "saved_at": file_path,
        "result_image": result_image_path,
        "detections": detections,
        "detection_count": len(detections)
    }
