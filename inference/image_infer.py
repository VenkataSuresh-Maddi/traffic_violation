import cv2
from ultralytics import YOLO
from utils.draw_boxes import draw_boxes

model = YOLO("models/best.pt")

image_path = input("Enter image path: ")
image = cv2.imread(image_path)

if image is None:
	print("❌ Could not read image. Check the file path and try again.")
else:
	results = model(image)
	output = draw_boxes(image, results)

	cv2.imshow("Helmet Detection", output)
	cv2.imwrite("outputs/images/result.jpg", output)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	print("✅ Image saved to outputs/images/")
