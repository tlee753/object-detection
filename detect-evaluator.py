from imageai.Detection.Custom import DetectionModelTrainer

model = "satboat-yolo/models/final.h5"

trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory="satboat")
trainer.evaluateModel(
    model_path= model,
    json_path="satboat-yolo/json/detection_config.json",
    iou_threshold=0.5,
    object_threshold=0.001,
    nms_threshold=0.5
)
