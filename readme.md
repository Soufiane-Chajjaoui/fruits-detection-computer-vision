# 🚀 Projet de Détection d'Objets avec YOLOv8

![YOLOv8 Detection Example](assets/demo.gif) *(Optionnel: Ajoutez une capture d'écran ou GIF)*

## 📝 Description
Ce projet utilise **YOLOv8** pour détecter des objets en temps réel à partir :
- D'une webcam
- D'une vidéo
- D'images statiques

**Cas d'utilisation** : Surveillance, analyse sportive, robotique, etc.

## 📦 Installation

### 1. Cloner le dépôt
```bash
git clone https://github.com/Soufiane-Chajjaoui/fruits-detection-computer-vision.git
cd fruits-detection-computer-vision
```

### 2. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 3. Lancer l'application
```bash
uvicorn main:app --reload
```
L'API sera accessible à l'adresse `http://localhost:8000`.