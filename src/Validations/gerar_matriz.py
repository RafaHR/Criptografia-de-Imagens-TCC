import numpy as np
import cv2
import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.rng_quantico import gerar_matriz_qrng_real

def salvar_matrizes_qrng_real(qtd=43, tamanho=400):

    for i in range(40, qtd + 1):
        print(f"\nGerando matriz QRNG simulado {i}/{qtd}...")
        start_time = time.time()

        matriz = gerar_matriz_qrng_real(tamanho=tamanho)

        caminho_img = f"src/matrices/real/real_matriz_qrng_{i:03d}.png"
        cv2.imwrite(caminho_img, matriz)

        end_time = time.time()
        print(f"Salvo {caminho_img} em {end_time - start_time:.2f}s")

if __name__ == "__main__":
    salvar_matrizes_qrng_real()
 