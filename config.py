# Paths to models
COCO_MODEL_PATH = './models/yolov8n.pt'
LICENSE_PLATE_MODEL_PATH = './models/license-plate-detector.pt'

# Path to video
VIDEO_PATH = './data/sample.mp4'

# Vehicle class IDs in COCO format
VEHICLES = [2, 3, 5, 7]  # Car, motorbike, bus, truck

# License data path
LICENSE_DATA_PATH = "./license_data.json"

# Unauthorized plates paths
UNAUTHORIZED_PATH = "./unauthorized"
UNAUTHORIZED_DATA_PATH = "./unauthorized/unauthorized_plates.json"

# Display window size (set default values if None)
DISPLAY_WINDOW_WIDTH = None # 1280  
DISPLAY_WINDOW_HEIGHT = None # 720  

# Ensure the unauthorized folder exists
import os
if not os.path.exists(UNAUTHORIZED_PATH):
    os.makedirs(UNAUTHORIZED_PATH)
