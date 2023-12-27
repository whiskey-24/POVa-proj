from ultralytics import YOLO

# Create a new YOLOv8n-OBB model from scratch
model = YOLO('yolov8n-obb.yaml', task='detect')

# Train the model on the DOTAv2 dataset
results = model.train(data='drone_1.yaml', epochs=100, imgsz=640, device="cuda")

metrics = model.val()  # run validation
# Plot results
# results.plot()

# Save trained model
# model.save('trained_model.pt')
path = model.export()
print(f"Model exported to {path}")

