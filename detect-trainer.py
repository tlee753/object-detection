from imageai.Detection.Custom import DetectionModelTrainer

trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory="satboat")
trainer.setTrainConfig(
    object_names_array=["satboat"],
    batch_size=8,
    num_experiments=100,
    train_from_pretrained_model="resnet.h5"
)
trainer.trainModel()
