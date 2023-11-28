import torchvision
import torch
import torchvision.transforms as T
import cv2 as cv
import numpy as np  # Agrega esta línea para importar NumPy


def faster_rcnn_model() -> torch.nn.Module:

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=True, progress=True, num_classes=91,
        pretrained_backbone=True, trainable_backbone_layers=3, max_size=360
    )
    model.eval()
    return model


def frcnn_predict(model: torch.nn.Module, frame: np.ndarray, thresh: float = 0.8):
    """yyyy
    Predict with Faster R-CNN.

    Args:
        model (torch.nn.Module): Faster R-CNN model.
        frame (np.ndarray): OpenCV image in BGR format.
        thresh (float, optional): Confidence threshold. Defaults to 0.8.

    Returns:
        torch.Tensor: Predicted bounding boxes.
        torch.Tensor: Predicted labels.
        torch.Tensor: Predicted class confidences.
    """
    # Preprocess image
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize(380),
        T.ToTensor(),
    ])

    t_image = transform(frame).unsqueeze(0)

    # Predict
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

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cap = cv.VideoCapture(0)  # Select the camera
    model = faster_rcnn_model()
    model = model.to(device)  # Mover el modelo a la GPU (si está disponible)

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Unable to capture frame.")
            break

        frame = cv.resize(frame, (640, 480))
        boxes, labels, conf = frcnn_predict(model, frame, thresh=0.9)

        for box, label, score in zip(boxes, labels, conf):
            if score >= 0.8:
                x1, y1, x2, y2 = map(int, box)
                cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv.putText(frame, f"{CATEGORY_NAMES[label]}: {score:.2f}", (x1, y1 - 10),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv.imshow('Object Detection', frame)

        if cv.waitKey(1) & 0xFF == 27:  # Press Esc to exit
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
