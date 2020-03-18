from imageai.Detection.Custom import CustomObjectDetection

detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath("data/models/detection_model-ex-095--loss-0018.137.h5") 
detector.setJsonPath("data/json/detection_config.json")
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image="boats/19.jpg", output_image_path="19-detected.jpg")

for detection in detections:
    print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])

