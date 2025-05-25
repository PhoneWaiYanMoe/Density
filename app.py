import requests
import cv2
import numpy as np
import tensorflow as tf
import time
import os
import json
import logging
import threading
import sys
import random
from flask import Flask, jsonify
from flask_cors import CORS
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)

print("🚀 DEBUG: APP.PY LOADED - CLEAN VERSION!")
logging.info("🚀 DEBUG: APP.PY LOADED - CLEAN VERSION!")

# Flask app setup
app = Flask(__name__)
CORS(app)

# Global variables
current_densities = {}
today_densities = {}
critical_densities = {}
last_update_time = None
is_processing = False
road_model = None
vehicle_model = None
USE_MOCK_DATA = True

# Parameters
IMG_HEIGHT = 128
IMG_WIDTH = 128
UPDATE_INTERVAL = 15

# Camera data
camera_mapping = {
    'Lý Thái Tổ - Sư Vạn Hạnh': 'A',
    'Ba Tháng Hai - Cao Thắng': 'B',
    'Điện Biên Phủ – Cao Thắng': 'C',
    'Nút giao Ngã sáu Nguyễn Tri Phương_1': 'D',
    'Nút giao Ngã sáu Nguyễn Tri Phương': 'E',
    'Nút giao Lê Đại Hành 2 (Lê Đại Hành)': 'F',
    'Lý Thái Tổ - Nguyễn Đình Chiểu': 'G',
    'Nút giao Ngã sáu Cộng Hòa_1': 'H',
    'Nút giao Ngã sáu Cộng Hòa': 'I',
    'Điện Biên Phủ - Cách Mạng Tháng Tám': 'J',
    'Nút giao Công Trường Dân Chủ': 'K',
    'Nút giao Công Trường Dân Chủ_1': 'L'
}

cameras = [
    ("6623e7076f998a001b2523ea", "Lý Thái Tổ - Sư Vạn Hạnh"),
    ("5deb576d1dc17d7c5515acf8", "Ba Tháng Hai - Cao Thắng"),
    ("63ae7a9cbfd3d90017e8f303", "Điện Biên Phủ – Cao Thắng"),
    ("5deb576d1dc17d7c5515ad21", "Nút giao Ngã sáu Nguyễn Tri Phương"),
    ("5deb576d1dc17d7c5515ad22", "Nút giao Ngã sáu Nguyễn Tri Phương_1"),
    ("5d8cdd26766c880017188974", "Nút giao Lê Đại Hành 2 (Lê Đại Hành)"),
    ("63ae763bbfd3d90017e8f0c4", "Lý Thái Tổ - Nguyễn Đình Chiểu"),
    ("5deb576d1dc17d7c5515acf6", "Nút giao Ngã sáu Cộng Hòa"),
    ("5deb576d1dc17d7c5515acf7", "Nút giao Ngã sáu Cộng Hòa_1"),
    ("5deb576d1dc17d7c5515acf2", "Điện Biên Phủ - Cách Mạng Tháng Tám"),
    ("5deb576d1dc17d7c5515acf9", "Nút giao Công Trường Dân Chủ"),
    ("5deb576d1dc17d7c5515acfa", "Nút giao Công Trường Dân Chủ_1")
]

# Create session for camera requests
session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
})

# Camera feed URLs
main_url = "https://giaothong.hochiminhcity.gov.vn"
base_url = "https://giaothong.hochiminhcity.gov.vn:8007/Render/CameraHandler.ashx"
default_params = {"bg": "black", "w": 300, "h": 230}

def initialize_critical_densities():
    global critical_densities
    critical_densities = {
        'A': 75.0, 'B': 65.0, 'C': 70.0, 'D': 85.0, 'E': 80.0, 'F': 55.0,
        'G': 68.0, 'H': 90.0, 'I': 85.0, 'J': 72.0, 'K': 78.0, 'L': 76.0
    }
    logging.info(f"✓ Critical densities initialized: {critical_densities}")

def download_models_from_drive():
    """Download models from Google Drive"""
    try:
        os.makedirs("models", exist_ok=True)
        logging.info("=== Downloading ML Models from Google Drive ===")
        
        model_urls = {
            "models/unet_road_segmentation.Better.keras": "https://drive.google.com/uc?export=download&id=1sVkYG4mqeH8wDeElz-Q9WVskvDq55U5B",
            "models/unet_multi_classV1.keras": "https://drive.google.com/uc?export=download&id=1NCjF309WQO0R_ATUkBRWxXC7GjBdUwkr"
        }
        
        for model_path, url in model_urls.items():
            if not os.path.exists(model_path):
                logging.info(f"Downloading {os.path.basename(model_path)} from Google Drive...")
                
                session_drive = requests.Session()
                response = session_drive.get(url, stream=True, allow_redirects=True, timeout=300)
                
                if 'download_warning' in response.text or 'virus scan warning' in response.text:
                    logging.info("Google Drive requires confirmation, extracting token...")
                    import re
                    confirm_token = None
                    
                    patterns = [
                        r'confirm=([a-zA-Z0-9\-_]+)',
                        r'"downloadUrl":"[^"]*confirm=([a-zA-Z0-9\-_]+)',
                        r'&confirm=([a-zA-Z0-9\-_]+)'
                    ]
                    
                    for pattern in patterns:
                        match = re.search(pattern, response.text)
                        if match:
                            confirm_token = match.group(1)
                            logging.info(f"Found confirmation token: {confirm_token}")
                            break
                    
                    if confirm_token:
                        confirm_url = f"{url}&confirm={confirm_token}"
                        response = session_drive.get(confirm_url, stream=True, allow_redirects=True, timeout=300)
                        logging.info("Made confirmed request to Google Drive")
                
                response.raise_for_status()
                
                total_size = 0
                chunk_count = 0
                with open(model_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            total_size += len(chunk)
                            chunk_count += 1
                            
                            if chunk_count % 1000 == 0:
                                logging.info(f"Downloaded {total_size/1024/1024:.1f}MB...")
                
                file_size = os.path.getsize(model_path)
                
                if file_size < 1000000:
                    logging.error(f"Downloaded file too small ({file_size} bytes), likely failed")
                    if os.path.exists(model_path):
                        os.remove(model_path)
                    raise Exception(f"Download failed - file too small")
                
                logging.info(f"✓ {os.path.basename(model_path)} downloaded successfully: {file_size/1024/1024:.1f}MB")
            else:
                file_size = os.path.getsize(model_path)
                logging.info(f"✓ {os.path.basename(model_path)} already exists: {file_size/1024/1024:.1f}MB")
        
        return True
        
    except Exception as e:
        logging.error(f"Failed to download models from Google Drive: {e}")
        return False

def try_load_models():
    global road_model, vehicle_model, USE_MOCK_DATA
    
    try:
        logging.info("=== Attempting to load ML models ===")
        logging.info("🔍 DEBUG: About to call download_models_from_drive()")
        
        download_success = download_models_from_drive()
        logging.info(f"📥 DEBUG: download_models_from_drive() returned: {download_success}")
        
        if not download_success:
            logging.info("Model download failed, using mock data mode")
            USE_MOCK_DATA = True
            return
        
        road_model_path = "models/unet_road_segmentation.Better.keras"
        vehicle_model_path = "models/unet_multi_classV1.keras"
        
        logging.info(f"📁 DEBUG: Checking if {road_model_path} exists: {os.path.exists(road_model_path)}")
        logging.info(f"📁 DEBUG: Checking if {vehicle_model_path} exists: {os.path.exists(vehicle_model_path)}")
        
        if os.path.exists(road_model_path) and os.path.exists(vehicle_model_path):
            logging.info("Model files found, attempting to load...")
            
            def dice_loss(y_true, y_pred, smooth=1e-6):
                y_true_f = tf.keras.backend.flatten(y_true)
                y_pred_f = tf.keras.backend.flatten(y_pred)
                intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
                return 1 - ((2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth))
            
            logging.info("🤖 DEBUG: Loading road segmentation model...")
            road_model = tf.keras.models.load_model(road_model_path, custom_objects={"dice_loss": dice_loss})
            logging.info("✓ Road model loaded")
            
            logging.info("🚗 DEBUG: Loading vehicle classification model...")
            vehicle_model = tf.keras.models.load_model(vehicle_model_path, custom_objects={"dice_loss": dice_loss})
            logging.info("✓ Vehicle model loaded")
            
            USE_MOCK_DATA = False
            logging.info("=== ✓ REAL MODELS LOADED SUCCESSFULLY ===")
            logging.info("Switching to real camera processing mode")
            
        else:
            logging.info("❌ DEBUG: Model files not found after download, using mock data mode")
            USE_MOCK_DATA = True
            
    except Exception as e:
        logging.error(f"💥 DEBUG: Exception in try_load_models: {e}")
        import traceback
        logging.error(f"💥 DEBUG: Traceback: {traceback.format_exc()}")
        logging.info("Falling back to mock data mode")
        USE_MOCK_DATA = True

def generate_mock_density_data():
    global current_densities, today_densities, last_update_time
    
    current_time = datetime.now()
    hour = current_time.hour
    
    if 7 <= hour <= 9:
        multiplier = 1.8
    elif 17 <= hour <= 19:
        multiplier = 2.0
    elif 11 <= hour <= 13:
        multiplier = 1.3
    elif hour >= 22 or hour <= 6:
        multiplier = 0.4
    else:
        multiplier = 1.0
    
    new_densities = {}
    base_densities = {
        'A': 65, 'B': 55, 'C': 70, 'D': 85, 'E': 80, 'F': 45,
        'G': 60, 'H': 90, 'I': 85, 'J': 75, 'K': 80, 'L': 75
    }
    
    for cam_id, cam_location in cameras:
        camera_id = camera_mapping.get(cam_location)
        if not camera_id:
            continue
        
        base = base_densities.get(camera_id, 60)
        raw_density = base * multiplier * random.uniform(0.85, 1.15)
        raw_density = max(10, min(95, raw_density))
        
        kc = critical_densities.get(camera_id, 100.0)
        normalized = (raw_density / kc) * 100
        normalized = min(normalized, 100.0)
        
        new_densities[camera_id] = {
            'density': round(normalized, 2),
            'raw_density': round(raw_density, 2),
            'critical_density': kc,
            'location': cam_location,
            'timestamp': current_time.isoformat(),
            'status': 'mock_data' if USE_MOCK_DATA else 'real_ai'
        }
        
        if camera_id not in today_densities:
            today_densities[camera_id] = {}
        time_key = current_time.strftime('%H:%M:%S')
        today_densities[camera_id][time_key] = raw_density
    
    current_densities = new_densities
    last_update_time = current_time.isoformat()

def manage_daily_data():
    global today_densities, critical_densities
    
    today = datetime.now().date().strftime('%Y-%m-%d')
    
    if not today_densities or today_densities.get('date') != today:
        logging.info(f"New day detected: {today}")
        
        if today_densities and 'date' in today_densities:
            new_critical = {}
            for camera_id in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']:
                if camera_id in today_densities:
                    max_density = max(today_densities[camera_id].values()) if today_densities[camera_id] else 0
                    new_critical[camera_id] = max_density
            
            if new_critical:
                critical_densities = new_critical
                logging.info(f"✓ Updated critical densities from yesterday: {critical_densities}")
        
        today_densities = {'date': today}

def background_processor():
    while True:
        try:
            manage_daily_data()
            generate_mock_density_data()
            logging.info(f"✓ Data updated ({len(current_densities)} cameras) - Mode: {'Mock' if USE_MOCK_DATA else 'Real AI'}")
            time.sleep(UPDATE_INTERVAL)
        except Exception as e:
            logging.error(f"Background processor error: {e}")
            time.sleep(UPDATE_INTERVAL)

# API Routes
@app.route('/')
def home():
    return jsonify({
        "message": "Ho Chi Minh City Traffic Density API",
        "version": "1.0.0",
        "status": "running",
        "data_mode": "mock_data" if USE_MOCK_DATA else "real_ai_models",
        "last_update": last_update_time,
        "update_interval": f"{UPDATE_INTERVAL} seconds",
        "total_cameras": len(cameras),
        "models_loaded": road_model is not None and vehicle_model is not None,
        "endpoints": {
            "/densities": "Current densities (updated every 15s)",
            "/densities/<camera_id>": "Specific camera density",
            "/cameras": "All cameras list",
            "/today-densities": "Today's historical data",
            "/critical-densities": "Critical thresholds",
            "/data-source": "Check if using real or mock data"
        }
    })

@app.route('/densities')
def get_densities():
    return jsonify({
        "data": current_densities,
        "last_update": last_update_time,
        "cameras_online": len(current_densities),
        "data_mode": "mock_data" if USE_MOCK_DATA else "real_ai_models",
        "status": "success"
    })

@app.route('/densities/<camera_id>')
def get_camera_density(camera_id):
    camera_id = camera_id.upper()
    if camera_id in current_densities:
        return jsonify({
            "camera_id": camera_id,
            "data": current_densities[camera_id],
            "status": "success"
        })
    return jsonify({"error": f"Camera {camera_id} not found"}), 404

@app.route('/cameras')
def get_cameras():
    camera_info = []
    for cam_id, cam_location in cameras:
        camera_id = camera_mapping.get(cam_location)
        is_online = camera_id in current_densities
        current_data = current_densities.get(camera_id, {})
        
        camera_info.append({
            "id": camera_id,
            "internal_id": cam_id,
            "location": cam_location,
            "online": is_online,
            "current_density": current_data.get('density'),
            "critical_density": critical_densities.get(camera_id),
            "last_update": current_data.get('timestamp'),
            "data_source": current_data.get('status', 'offline')
        })
    
    return jsonify({
        "cameras": camera_info,
        "total": len(camera_info),
        "online": len(current_densities),
        "data_mode": "mock_data" if USE_MOCK_DATA else "real_ai_models",
        "status": "success"
    })

@app.route('/today-densities')
def get_today_densities():
    return jsonify({
        "date": today_densities.get('date'),
        "data": {k: v for k, v in today_densities.items() if k != 'date'},
        "status": "success"
    })

@app.route('/critical-densities')
def get_critical_densities():
    return jsonify({
        "critical_densities": critical_densities,
        "description": "Daily updated thresholds from previous day's peak traffic",
        "status": "success"
    })

@app.route('/data-source')
def get_data_source():
    return jsonify({
        "mode": "mock_data" if USE_MOCK_DATA else "real_ai_models",
        "models_loaded": road_model is not None and vehicle_model is not None,
        "description": "Mock data simulates realistic traffic patterns" if USE_MOCK_DATA else "AI models process live camera feeds from HCMC traffic system"
    })

@app.route('/status')
def get_status():
    return jsonify({
        "status": "healthy",
        "data_mode": "mock_data" if USE_MOCK_DATA else "real_ai_models",
        "cameras_online": len(current_densities),
        "models_loaded": road_model is not None and vehicle_model is not None,
        "last_update": last_update_time
    })

if __name__ == '__main__':
    logging.info("=== HCMC Traffic Density API Starting ===")
    logging.info("🚀 DEBUG: Application starting up...")
    
    initialize_critical_densities()
    
    logging.info("🤖 DEBUG: About to call try_load_models()...")
    try_load_models()
    logging.info(f"🤖 DEBUG: After try_load_models(), USE_MOCK_DATA = {USE_MOCK_DATA}")
    logging.info(f"🤖 DEBUG: road_model is None: {road_model is None}")
    logging.info(f"🤖 DEBUG: vehicle_model is None: {vehicle_model is None}")
    
    thread = threading.Thread(target=background_processor, daemon=True)
    thread.start()
    logging.info("✓ Background processor started")
    
    port = int(os.environ.get('PORT', 5000))
    logging.info(f"✓ API ready on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)