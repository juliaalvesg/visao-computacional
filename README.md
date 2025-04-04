# visao-computacional
# Projeto de Integração TA e TI: Visão Computacional

## Objetivo

A proposta inicial deste projeto é iniciar a integração entre a Tecnologia da Automação (TA) e a Tecnologia da Informação (TI), utilizando visão computacional como forma de capturar e transformar dados diretamente da planta industrial em informações gerenciais úteis.

## Problema Detectado

Hoje existe uma lacuna entre o que é gerado na planta industrial (TA) e o que chega até os sistemas de controle e análise (TI), o que ocasiona:

- Perda de dados relevantes
- Atraso na tomada de decisão
- Dificuldade de rastreabilidade de falhas

## Solução Proposta (Fase 1 - Visão Computacional)

Criamos um sistema de visão computacional que:

1. Captura vídeos/imagens da linha de produção.
2. Utiliza o OpenCV para:
   - Processar os frames
   - Detectar objetos que passam por um ponto de controle
   - Contabilizar esses objetos
3. Registra os dados em um arquivo `.csv` ou em uma base de dados para uso posterior por sistemas gerenciais.

## Tecnologias Utilizadas

- **Python**
- **OpenCV**
- **Pandas**
- **NumPy**

## Código Fonte (Resumo)

```python
import cv2
import pandas as pd
from datetime import datetime

cap = cv2.VideoCapture('linha_producao.mp4')
object_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Conversão para cinza e detecção de bordas
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Contagem simples por contornos (pode ser melhorado com filtros)
    for contour in contours:
        if cv2.contourArea(contour) > 1000:
            object_count += 1
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f'Objeto detectado às {timestamp}')

cap.release()
cv2.destroyAllWindows()

# Salvando os dados
df = pd.DataFrame({
    "quantidade": [object_count],
    "data_hora": [datetime.now()]
})
df.to_csv('dados_producao.csv', index=False)
