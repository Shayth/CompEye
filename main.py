from imageai.Detection import ObjectDetection
import os

ex_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(os.path.join(ex_path, "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()
list = detector.detectCustomObjectsFromImage(
    input_image=os.path.join(ex_path, "Image.jpg"),
    output_image_path=os.path.join(ex_path, "New_Image.jpg")
)