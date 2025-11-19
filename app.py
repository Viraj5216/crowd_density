import cv2
import gradio as gr

def count_people(image):
    # Load a lightweight pre-trained face detector from OpenCV
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces (proxy for crowd density)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw boxes around each detected face
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    count = len(faces)
    return image, f"Estimated people count: {count}"

# Create Gradio interface
demo = gr.Interface(
    fn=count_people,
    inputs=gr.Image(type="numpy"),
    outputs=[gr.Image(type="numpy"), gr.Textbox()],
    title="Simple Crowd Density Estimator",
    description="Upload a scene image and get estimated crowd density using face detection."
)

if __name__ == "__main__":
    demo.launch()

