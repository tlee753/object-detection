from imageai.Detection.Custom import CustomObjectDetection

detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath("data/models/final.h5") 
detector.setJsonPath("data/json/detection_config.json")
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image="boats-2/24.jpg", output_image_path="result.jpg")

for detection in detections:
    print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])

