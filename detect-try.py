from imageai.Detection.Custom import CustomObjectDetection

model = "detection_model-ex-098--loss-0011.529.h5"

detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath("data/models/" + model)
detector.setJsonPath("data/json/detection_config.json")
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image="boats-2/24.jpg", output_image_path="24-d.jpg")
for detection in detections:
    print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])
