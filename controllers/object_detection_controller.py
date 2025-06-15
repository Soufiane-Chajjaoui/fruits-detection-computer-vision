from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from typing import List, Dict, Any, Union, Optional
from pydantic import BaseModel, Field
import os
import shutil
from ultralytics import YOLO
import cv2
import numpy as np
import uuid
from datetime import datetime
import base64
import json
import asyncio

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

class VideoDetectionResponse(BaseModel):
    """
    Réponse de l'API de détection d'objets dans une vidéo.
    
    Attributes:
        task_id: Identifiant unique de la tâche de traitement
        filename: Nom du fichier original
        content_type: Type MIME du fichier
        saved_at: Chemin où le fichier original est sauvegardé
        status: État du traitement (en attente, en cours, terminé, erreur)
        result_video: Chemin vers la vidéo avec les annotations (disponible une fois terminé)
    """
    task_id: str = Field(description="Identifiant unique de la tâche")
    filename: str = Field(description="Nom du fichier original")
    content_type: str = Field(description="Type MIME du fichier")
    saved_at: str = Field(description="Chemin où le fichier original est sauvegardé")
    status: str = Field(description="État du traitement")
    result_video: Optional[str] = Field(None, description="Chemin vers la vidéo avec les annotations")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "task_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
                "filename": "example.mp4",
                "content_type": "video/mp4",
                "saved_at": "uploads/example.mp4",
                "status": "processing",
                "result_video": None
            }
        }
    }

class RealTimeDetectionResponse(BaseModel):
    """
    Réponse pour la détection en temps réel.
    
    Attributes:
        frame_id: Identifiant de l'image
        detections: Liste des objets détectés
        detection_count: Nombre d'objets détectés
        annotated_frame: Image annotée encodée en base64
    """
    frame_id: int = Field(description="Identifiant de l'image")
    detections: List[Dict[str, Any]] = Field(description="Liste des objets détectés")
    detection_count: int = Field(description="Nombre d'objets détectés")
    annotated_frame: str = Field(description="Image annotée encodée en base64")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "frame_id": 42,
                "detections": [
                    {
                        "class": "person",
                        "confidence": 0.95,
                        "bbox": [10.5, 20.3, 200.1, 300.7]
                    }
                ],
                "detection_count": 1,
                "annotated_frame": "base64_encoded_image_data..."
            }
        }
    }

# Dictionnaire pour stocker l'état des tâches
video_tasks = {}

router = APIRouter(tags=["object-detection"])

# Créer les dossiers nécessaires
os.makedirs("uploads", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Charger le modèle YOLO
model = YOLO("yolov8n.pt")  # Utilise le modèle YOLOv8 nano préentraîné

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

def process_video(task_id: str, video_path: str):
    """
    Traite une vidéo avec YOLO pour détecter les objets.
    Cette fonction s'exécute en arrière-plan.
    
    Args:
        task_id: Identifiant de la tâche
        video_path: Chemin vers la vidéo à traiter
    """
    try:
        # Mettre à jour le statut
        video_tasks[task_id]["status"] = "processing"
        
        # Préparer le chemin de sortie
        filename = os.path.basename(video_path)
        output_path = os.path.join("results", f"result_{filename}")
        
        # Ouvrir la vidéo source
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("Impossible d'ouvrir la vidéo")
        
        # Obtenir les propriétés de la vidéo
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Créer l'objet VideoWriter pour la sortie
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Traiter chaque image
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Détecter les objets avec YOLO
            results = model(frame)
            
            # Dessiner les résultats sur l'image
            annotated_frame = results[0].plot()
            
            # Écrire l'image dans la vidéo de sortie
            out.write(annotated_frame)
            
            # Mettre à jour le compteur
            frame_count += 1
            
            # Mettre à jour le statut avec la progression
            progress = int((frame_count / total_frames) * 100)
            video_tasks[task_id]["progress"] = progress
        
        # Libérer les ressources
        cap.release()
        out.release()
        
        # Mettre à jour le statut final
        video_tasks[task_id]["status"] = "completed"
        video_tasks[task_id]["result_video"] = output_path
        
    except Exception as e:
        # En cas d'erreur, mettre à jour le statut
        video_tasks[task_id]["status"] = "error"
        video_tasks[task_id]["error"] = str(e)

@router.post(
    "/detect-video/", 
    response_model=VideoDetectionResponse,
    summary="Détecter des objets dans une vidéo",
    description="""
    Téléchargez une vidéo et détectez les objets présents en utilisant YOLOv8.
    
    Le traitement se fait en arrière-plan et peut prendre du temps selon la longueur de la vidéo.
    Utilisez l'endpoint GET /video-status/{task_id} pour vérifier l'état du traitement.
    
    L'API retourne immédiatement:
    - Un identifiant de tâche unique
    - Les informations sur le fichier téléchargé
    - Le statut initial du traitement
    """
)
async def detect_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Détecte les objets dans une vidéo téléchargée en utilisant YOLOv8.
    Le traitement se fait en arrière-plan.
    
    Args:
        background_tasks: Gestionnaire de tâches en arrière-plan
        file: Le fichier vidéo à analyser
        
    Returns:
        Un objet VideoDetectionResponse contenant l'ID de la tâche et le statut initial
        
    Raises:
        HTTPException: Si le fichier n'est pas une vidéo
    """
    # Vérifier que le fichier est une vidéo
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Le fichier doit être une vidéo")
    
    # Générer un ID unique pour cette tâche
    task_id = str(uuid.uuid4())
    
    # Chemin où le fichier sera sauvegardé
    file_path = os.path.join("uploads", file.filename)
    
    # Sauvegarder le fichier
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Créer une entrée pour cette tâche
    video_tasks[task_id] = {
        "task_id": task_id,
        "filename": file.filename,
        "content_type": file.content_type,
        "saved_at": file_path,
        "status": "pending",
        "created_at": datetime.now().isoformat(),
        "progress": 0,
        "result_video": None
    }
    
    # Lancer le traitement en arrière-plan
    background_tasks.add_task(process_video, task_id, file_path)
    
    # Retourner immédiatement la réponse avec l'ID de la tâche
    return {
        "task_id": task_id,
        "filename": file.filename,
        "content_type": file.content_type,
        "saved_at": file_path,
        "status": "pending",
        "result_video": None
    }

@router.get(
    "/video-status/{task_id}",
    summary="Vérifier l'état d'une tâche de détection vidéo",
    description="Récupère l'état actuel d'une tâche de détection d'objets dans une vidéo."
)
async def get_video_status(task_id: str):
    """
    Récupère l'état actuel d'une tâche de traitement vidéo.
    
    Args:
        task_id: L'identifiant unique de la tâche
        
    Returns:
        Les informations sur la tâche, y compris son état actuel
        
    Raises:
        HTTPException: Si la tâche n'existe pas
    """
    if task_id not in video_tasks:
        raise HTTPException(status_code=404, detail="Tâche non trouvée")
    
    return video_tasks[task_id]

@router.websocket("/ws/detect-realtime/")
async def detect_realtime(websocket: WebSocket):
    """
    Établit une connexion WebSocket pour la détection d'objets en temps réel.
    
    Le client envoie des images encodées en base64, et le serveur répond avec
    les détections et l'image annotée.
    """
    await websocket.accept()
    
    try:
        frame_id = 0
        while True:
            # Recevoir l'image du client
            data = await websocket.receive_text()
            
            try:
                # Décoder les données JSON
                json_data = json.loads(data)
                
                # Extraire l'image encodée en base64
                if "image" not in json_data:
                    await websocket.send_json({"error": "Aucune image trouvée dans les données"})
                    continue
                
                # Décoder l'image base64
                image_data = base64.b64decode(json_data["image"])
                nparr = np.frombuffer(image_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    await websocket.send_json({"error": "Impossible de décoder l'image"})
                    continue
                
                # Exécuter la détection avec YOLO
                results = model(frame)
                
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
                
                # Dessiner les résultats sur l'image
                annotated_frame = results[0].plot()
                
                # Encoder l'image annotée en base64
                _, buffer = cv2.imencode('.jpg', annotated_frame)
                annotated_frame_b64 = base64.b64encode(buffer).decode('utf-8')
                
                # Préparer la réponse
                response = {
                    "frame_id": frame_id,
                    "detections": detections,
                    "detection_count": len(detections),
                    "annotated_frame": annotated_frame_b64
                }
                
                # Envoyer la réponse au client
                await websocket.send_json(response)
                
                # Incrémenter l'ID de l'image
                frame_id += 1
                
            except json.JSONDecodeError:
                await websocket.send_json({"error": "Format JSON invalide"})
            except Exception as e:
                await websocket.send_json({"error": str(e)})
                
    except WebSocketDisconnect:
        print("Client déconnecté")
