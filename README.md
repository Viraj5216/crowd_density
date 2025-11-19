Crowd Density Detection (YOLOv8 + Gradio UI)

This project detects people in images and estimates crowd density using YOLOv8 and heatmap visualization.
It also provides a Gradio interface for easy interaction.

🚀 Features

Detects people using YOLOv8n

Draws bounding boxes around detected persons

Generates a density heatmap

Shows total count

Fully working CLI + Gradio Web UI

📁 Project Structure
crowd_density/
│
├── app.py               # CLI version
├── gradio_yolo.py       # Gradio UI version
├── sample_images/
│   └── small crowd.jpg  # Sample input
├── yolov8n.pt           # YOLO model
└── requirements.txt

🛠️ Installation

Clone the repository:

git clone https://github.com/Viraj5216/crowd_density
cd crowd_density


Install dependencies:

pip install -r requirements.txt

▶️ Run (CLI Mode)
python app.py sample_images/"small crowd.jpg" --out results


Outputs:

Annotated image

Heatmap

Text stats

💻 Run (Gradio UI)
python gradio_yolo.py


This launches the local Gradio interface on your machine.

🧠 Model Used

YOLOv8n (Ultralytics)

Pretrained on COCO dataset

Lightweight & fast for small crowd images

📝 Notes

This project runs locally, not online.

Anyone with Python + dependencies can run it easily.
