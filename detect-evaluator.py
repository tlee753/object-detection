from imageai.Detection.Custom import DetectionModelTrainer

model = "data/models"

trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory="data")
trainer.evaluateModel(
    model_path= model,
    json_path="data/json/detection_config.json",
    iou_threshold=0.5,
    object_threshold=0.3,
    nms_threshold=0.5
)
