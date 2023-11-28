import torch
import torchvision.transforms as T
import cv2 as cv
from torchvision.models.detection import fasterrcnn_resnet50_fpn

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

def load_faster_rcnn_model():
    """
    Carga un modelo Faster R-CNN preentrenado.
    """
    model = fasterrcnn_resnet50_fpn(
        pretrained=True, progress=True, num_classes=91,
        pretrained_backbone=True, trainable_backbone_layers=3, max_size=360
    )
    model.eval()

    # Mover el modelo a la GPU si está disponible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model

    

def predict_objects(model, frame, thresh=0.8):
    """
    Realiza la detección de objetos en un frame utilizando el modelo Faster R-CNN.

    Args:
        model: Modelo Faster R-CNN cargado.
        frame: Imagen de entrada en formato BGR.
        thresh: Umbral de confianza.

    Returns:
        boxes: Bounding boxes de objetos detectados.
        labels: Etiquetas de los objetos detectados.
        conf: Confianza de los objetos detectados.
    """
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize(380),
        T.ToTensor(),
    ])

    t_image = transform(frame).unsqueeze(0)
    output = model(t_image)

    boxes = output[0]["boxes"].detach()
    labels = output[0]["labels"].detach()
    conf = output[0]["scores"].detach()

    keep = conf > thresh
    boxes = boxes[keep]
    labels = labels[keep]
    conf = conf[keep]

    img_w = frame.shape[1] / t_image.size()[3]
    img_h = frame.shape[0] / t_image.size()[2]
    boxes = boxes * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)

    return boxes, labels, conf

def main():
    cap = cv.VideoCapture(0, cv.CAP_DSHOW)

    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")
        return

    model = load_faster_rcnn_model()

    fourcc = cv.VideoWriter_fourcc(*'XVID')
    result = cv.VideoWriter('Output.avi', fourcc, 20.0, (640, 480))

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: No se pudo capturar el frame.")
            break

        boxes, labels, conf = predict_objects(model, frame, thresh=0.9)

        for box, label, score in zip(boxes, labels, conf):
            if score >= 0.8:
                x1, y1, x2, y2 = map(int, box)
                cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv.putText(frame, f"{CATEGORY_NAMES[label]}: {score:.2f}", (x1, y1 - 10),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv.imshow('Object Detection', frame)
        result.write(frame)

        if cv.waitKey(1) & 0xFF == ord("e"):
            break

    result.release()
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
