from ultralytics import YOLO

# Create a new YOLOv8n-OBB model from scratch
model = YOLO('yolov8n.pt', task='detect')

# Train the model on the DOTAv2 dataset
results = model.train(data='retina_yolo.yaml', epochs=100, imgsz=640, device="cuda:0")

# metrics = model.val()  # run validation
# Plot results
# results.plot()

# Save trained model
# model.save('trained_model.pt')
path = model.export()
print(f"Model exported to {path}")

