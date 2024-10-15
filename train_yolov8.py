import os

# Set the environment variable to avoid OpenMP errors
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Now, proceed with your YOLOv8 code
from ultralytics import YOLO

# Load a pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')

# Train the model
model.train(data=r'C:/Users/jbull/OneDrive - Fayetteville State University/CSC490 SENIOR PROJECT/Lightning.yaml', 
            epochs=50, 
            imgsz=640)


