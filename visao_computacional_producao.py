import cv2
import numpy as np
import pandas as pd
from datetime import datetime

# Configurações iniciais
video_path = 'linha_producao.mp4'  # Substitua por 0 se quiser usar webcam
cap = cv2.VideoCapture(video_path)

# Linha para contagem de objetos
line_position = 300
offset = 6  # margem de erro

# Armazenamento de contagem e posições
object_count = 0
detections = []

# Dados registrados
dados_registrados = []

# Subtrator de fundo
fgbg = cv2.createBackgroundSubtractorMOG2()

def detect_center(x, y, w, h):
    """ Calcula o centro do contorno detectado """
    cx = int(x + w / 2)
    cy = int(y + h / 2)
    return cx, cy

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Pré-processamento
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 5)
    fgmask = fgbg.apply(blur)

    # Remoção de ruído
    dilated = cv2.dilate(fgmask, np.ones((5,5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

    # Contornos
    contours, _ = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Desenha linha de contagem
    cv2.line(frame, (0, line_position), (frame.shape[1], line_position), (0, 255, 0), 2)

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)

        if cv2.contourArea(contour) > 2000:
            center = detect_center(x, y, w, h)
            detections.append(center)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.circle(frame, center, 4, (0, 0, 255), -1)

 import time

ultima_contagem = 0

for (x, y) in detections:
    if (line_position - offset) < y < (line_position + offset):
        if time.time() - ultima_contagem > 1:  # mínimo 1 segundo entre contagens
            object_count += 1
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"Objeto #{object_count} detectado às {timestamp}")
            dados_registrados.append({"Objeto": object_count, "Timestamp": timestamp})
            ultima_contagem = time.time()
        detections.remove((x, y))

    # Exibição (opcional)
    cv2.putText(frame, f"Contagem: {object_count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Monitoramento da Planta", frame)

    # Tecla 'q' para sair
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Libera recursos
cap.release()
cv2.destroyAllWindows()

# Salva os dados em CSV
df = pd.DataFrame(dados_registrados)
df.to_csv('dados_producao.csv', index=False)

print("Execução finalizada. Dados salvos em 'dados_producao.csv'.")
