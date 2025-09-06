import cv2
import os
import urllib.request

# Model URLs
prototxt_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
model_url = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

prototxt_file = "deploy.prototxt"
model_file = "res10_300x300_ssd_iter_140000.caffemodel"

# Download if missing
if not os.path.exists(prototxt_file):
    print("Downloading deploy.prototxt...")
    urllib.request.urlretrieve(prototxt_url, prototxt_file)

if not os.path.exists(model_file):
    print("Downloading res10_300x300_ssd_iter_140000.caffemodel...")
    urllib.request.urlretrieve(model_url, model_file)

print("Model files ready!")

# Load the DNN model
net = cv2.dnn.readNetFromCaffe(prototxt_file, model_file)

# Image input
image_path = input("Enter the path of your image: ").strip()
img = cv2.imread(image_path)
if img is None:
    print("Error: Image not found!")
    exit()

(h, w) = img.shape[:2]

# Detect faces
blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
                             (300, 300), (104.0, 177.0, 123.0))
net.setInput(blob)
detections = net.forward()

count = 0
for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > 0.5:
        count += 1
        box = detections[0, 0, i, 3:7] * [w, h, w, h]
        (x1, y1, x2, y2) = box.astype("int")
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

print(f"Faces detected: {count}")
cv2.imwrite("face_detected_dnn.png", img)
print("Saved as face_detected_dnn.png")
cv2.imshow("Detected Faces (DNN)", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
