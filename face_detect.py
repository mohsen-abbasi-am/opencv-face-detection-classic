import cv2
import sys

cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

image_path = "sample.jpg"
image = cv2.imread(image_path)

if image is None:
    print(f"Error: Image '{image_path}' not found.")
    sys.exit()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags=cv2.CASCADE_SCALE_IMAGE
)

print(f"[INFO] {len(faces)} face(s) detected.")

for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)


cv2.imshow("Detected Faces", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
