import cv2

# Abre la cámara
cap = cv2.VideoCapture(0)

# Obtiene la resolución de la cámara
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

print(f"La resolución de la cámara es {width}x{height}")

# Cierra la cámara
cap.release()
