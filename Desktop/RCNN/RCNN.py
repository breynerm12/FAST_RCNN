import cv2
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torchvision.transforms as T
from PIL import Image
import torch


# Carga del modelo preentrenado
frcnn_model = fasterrcnn_resnet50_fpn(pretrained=True)
#frcnn_model = fasterrcnn_resnet50_fpn(weights="FasterRCNN_ResNet50_FPN_Weights.DEFAULT")


frcnn_model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
frcnn_model = frcnn_model.to(device)

# Lista de nombres de categorías
CATEGORY_NAMES = [
    '_background_', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Función para obtener predicciones del modelo
def get_prediction(img_path, threshold=0.7):
    img = Image.open(img_path)
    img = T.ToTensor()(img).to(device)
    with torch.no_grad():
        pred = frcnn_model([img])

    pred_boxes = pred[0]['boxes']
    pred_labels = pred[0]['labels']
    pred_scores = pred[0]['scores']

    # Filtra las detecciones por umbral
    keep = pred_scores > threshold
    pred_boxes = pred_boxes[keep]
    pred_labels = pred_labels[keep]
    pred_scores = pred_scores[keep]

    return pred_boxes, pred_labels, pred_scores

# Captura de la cámara en tiempo real y detección de objetos
cap = cv2.VideoCapture(0)  # Selecciona la cámara
while True:
    ret, frame = cap.read()

    # Guarda la última captura como imagen
    cv2.imwrite('temp_frame.jpg', frame)

    # Llama a la función para obtener predicciones
    pred_boxes, pred_labels, pred_scores = get_prediction('temp_frame.jpg', threshold=0.7)

    # Visualiza las detecciones en el video en tiempo real
    for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
        if score >= 0.5:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{CATEGORY_NAMES[label]}: {score:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Presiona Esc para salir
        break

cap.release()
cv2.destroyAllWindows()