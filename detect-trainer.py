from imageai.Detection.Custom import DetectionModelTrainer

trainer = DetectionModelTrainer()
trainer.setModelTypeAsRetinaNet()
trainer.setDataDirectory(data_directory="data")
trainer.setTrainConfig(
    object_names_array=["boat"],
    batch_size=32,
    num_experiments=100,
    train_from_pretrained_model="resnet.h5"
)
trainer.trainModel()
