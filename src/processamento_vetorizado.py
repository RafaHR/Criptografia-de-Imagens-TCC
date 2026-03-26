import numpy as np
import cv2
import time
import matplotlib.pyplot as plt

CAMINHO_IMAGEM = "/home/rafael-brain/GitHub/Estudo/TCC/src/Apresentacao_TCC/img_entrada/f961be76-1d3b-4b26-bc53-bca89b0f0e18.jpg"
CAMINHO_QRNG   = "/home/rafael-brain/GitHub/Estudo/TCC/src/matrices/real/matriz_qrng_001.png"

TAMANHO_CORTE = 400  # tamanho do quadrado central (400x400)


##Funções auxiliares vetorizadas
def aplicar_xor_com_qrng_vectorizado(imagem_processada, matriz_qrng):
    """
    Aplica XOR entre a imagem e a matriz QRNG usando vetorização NumPy.
    Funciona tanto para imagens em tons de cinza quanto RGB.
    """
    if imagem_processada.shape[:2] != matriz_qrng.shape:
        raise ValueError("A imagem e a matriz QRNG devem ter o mesmo tamanho")

    # Garante mesmo tipo
    imagem = imagem_processada.astype(np.uint8)
    qrng = matriz_qrng.astype(np.uint8)

    # Imagem RGB: shape (H, W, 3)
    if imagem.ndim == 3 and imagem.shape[2] == 3:
        # Expande a matriz QRNG para (H, W, 1) e deixa o NumPy broadcastar
        qrng_expandida = qrng[..., None]  # equivalente a np.expand_dims(qrng, axis=2)
        xor_resultado = np.bitwise_xor(imagem, qrng_expandida)
    else:
        # Imagem 2D (tons de cinza)
        xor_resultado = np.bitwise_xor(imagem, qrng)

    return xor_resultado



def logistic_map_vectorizado(size, r=3.999, x0=0.98892455322743):
    """
    Gera a sequência do mapa logístico.
    Ainda é sequencial (depende de x[n-1]), mas usa array pré-alocado em NumPy.
    """
    x = x0
    seq = np.empty(size, dtype=np.float64)
    for i in range(size):
        x = r * x * (1 - x)
        seq[i] = x
    return seq

def aplicar_lm_vectorizado(imagem):
    """
    Aplica o mapa logístico para permutar os pixels da imagem de forma vetorizada.
    Funciona para imagens 2D (cinza) e 3D (RGB).
    """
    h, w = imagem.shape[:2]
    total = h * w

    chaos_seq = logistic_map_vectorizado(total)
    indices = np.argsort(chaos_seq)

    if imagem.ndim == 3:
        # Imagem RGB: (H, W, C) -> (H*W, C)
        C = imagem.shape[2]
        img_flat = imagem.reshape(-1, C)          # (total, C)
        img_perm = img_flat[indices, :]           # reordena linhas via indices
        imagem_caotica = img_perm.reshape(h, w, C)
    else:
        # Imagem em tons de cinza: (H, W) -> (H*W,)
        img_flat = imagem.reshape(-1)             # (total,)
        img_perm = img_flat[indices]              # reordena via indices
        imagem_caotica = img_perm.reshape(h, w)

    return imagem_caotica, chaos_seq


##Função não vetorizada
def aplicar_xor_com_qrng_nao_vectorizado(imagem_processada, matriz_qrng):
    """
    Versão não vetorizada: usa laços explícitos em pixels e canais.
    Útil para comparação de desempenho com a versão vetorizada.
    """
    if imagem_processada.shape[:2] != matriz_qrng.shape:
        raise ValueError("A imagem e a matriz QRNG devem ter o mesmo tamanho")

    imagem = imagem_processada.astype(np.uint8)
    qrng = matriz_qrng.astype(np.uint8)

    altura, largura = qrng.shape

    if imagem.ndim == 3 and imagem.shape[2] == 3:
        # RGB
        xor_resultado = np.empty_like(imagem, dtype=np.uint8)
        for i in range(altura):
            for j in range(largura):
                for c in range(3):
                    xor_resultado[i, j, c] = imagem[i, j, c] ^ qrng[i, j]
    else:
        # Tons de cinza
        xor_resultado = np.empty_like(imagem, dtype=np.uint8)
        for i in range(altura):
            for j in range(largura):
                xor_resultado[i, j] = imagem[i, j] ^ qrng[i, j]

    return xor_resultado

def logistic_map_nao_vectorizado(size, r=3.999, x0=0.98892455322743):
    """
    Versão não vetorizada do mapa logístico: usa lista + append.
    """
    x = x0
    seq = []
    for _ in range(size):
        x = r * x * (1 - x)
        seq.append(x)
    return np.array(seq, dtype=np.float64)

def aplicar_lm_nao_vectorizado(imagem):
    """
    Aplica o mapa logístico permutando a imagem usando laços explícitos
    (sem flatten/reshape + indexação vetorizada).
    """
    h, w = imagem.shape[:2]
    total = h * w

    chaos_seq = logistic_map_nao_vectorizado(total)

    # Gera os índices ordenados "na mão" (sem np.argsort)
    indices = list(range(total))
    indices.sort(key=lambda i: chaos_seq[i])

    # Aloca saída com mesmo shape da imagem original
    imagem_caotica = np.empty_like(imagem)

    if imagem.ndim == 3:
        C = imagem.shape[2]
        for novo_idx, antigo_idx in enumerate(indices):
            # Coordernadas (linha, coluna) na imagem permutada
            new_i = novo_idx // w
            new_j = novo_idx % w
            # Coordenadas (linha, coluna) na imagem original
            old_i = antigo_idx // w
            old_j = antigo_idx % w

            for c in range(C):
                imagem_caotica[new_i, new_j, c] = imagem[old_i, old_j, c]

    else:
        for novo_idx, antigo_idx in enumerate(indices):
            new_i = novo_idx // w
            new_j = novo_idx % w
            old_i = antigo_idx // w
            old_j = antigo_idx % w

            imagem_caotica[new_i, new_j] = imagem[old_i, old_j]

    return imagem_caotica, chaos_seq

##Funções auxiliares de imagem

def cortar_centro(imagem: np.ndarray, tamanho: int) -> np.ndarray | None:
    """
    Recorta um quadrado central da imagem com o tamanho especificado.

    Returns:
        np.ndarray: A imagem cortada.
        None: Se a imagem for menor que o tamanho do corte.
    """
    if imagem is None:
        print("Erro: Nenhuma imagem fornecida para cortar.")
        return None

    h, w = imagem.shape[:2]
    if h < tamanho or w < tamanho:
        print(f"Erro: Imagem ({w}x{h}) muito pequena para cortar um quadrado de {tamanho}x{tamanho}.")
        return None

    centro_x, centro_y = w // 2, h // 2
    metade_tamanho = tamanho // 2
    
    # Calcula as coordenadas do corte
    y1 = centro_y - metade_tamanho
    y2 = centro_y + metade_tamanho
    x1 = centro_x - metade_tamanho
    x2 = centro_x + metade_tamanho

    return imagem[y1:y2, x1:x2]

def mostrar_imagem(imagem_input: np.ndarray, titulo: str) -> None:
    """
    Mostra uma única imagem usando Matplotlib.
    Converte de BGR (OpenCV) para RGB (Matplotlib).
    """
    if imagem_input is not None:
        # Converte a imagem do padrão BGR do OpenCV para o RGB do Matplotlib
        imagem_rgb = cv2.cvtColor(imagem_input, cv2.COLOR_BGR2RGB)
        plt.imshow(imagem_rgb)
        plt.title(titulo)
        plt.axis("off")
        plt.show()
    else:
        print("Erro: Tentativa de mostrar uma imagem que é nula (None).")

def mostrar_lado_a_lado(img1: np.ndarray, img2: np.ndarray, titulo1: str = "Imagem 1", titulo2: str = "Imagem 2") -> None:
    """Mostra duas imagens lado a lado para comparação."""
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Imagem 1
    axs[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    axs[0].set_title(titulo1)
    axs[0].axis('off')

    # Imagem 2
    axs[1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    axs[1].set_title(titulo2)
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()



def main():
    # --------- Carregar imagens a partir dos caminhos ---------
    imagem = cv2.imread(CAMINHO_IMAGEM, cv2.IMREAD_COLOR)
    if imagem is None:
        print(f"Erro ao carregar imagem normal: {CAMINHO_IMAGEM}")
        return

    matriz_qrng = cv2.imread(CAMINHO_QRNG, cv2.IMREAD_GRAYSCALE)
    if matriz_qrng is None:
        print(f"Erro ao carregar imagem QRNG: {CAMINHO_QRNG}")
        return

    print("Tamanho da imagem original:", imagem.shape)
    print("Tamanho da QRNG original:", matriz_qrng.shape)

    tempo_start = time.time()


    # ---- Cortar centro das duas imagens para garantir mesmo tamanho ----
    tamanho_crop = TAMANHO_CORTE  # mesmo valor que você já usa
    imagem_processada = cortar_centro(imagem, tamanho_crop)
    matriz_qrng_processada = cortar_centro(matriz_qrng, tamanho_crop)

    if imagem_processada is None or matriz_qrng_processada is None:
        print("Erro ao cortar imagens, abortando.")
        return

    print("Tamanho da imagem após corte:", imagem_processada.shape)
    print("Tamanho da QRNG após corte:", matriz_qrng_processada.shape)

    mostrar_imagem(imagem_processada, "Imagem de tamanho 400x400")

    tempo_start = time.time()

    # =========================
    # XOR vetorizado vs não vetorizado
    # =========================
    t0 = time.time()
    imagem_xor_vec = aplicar_xor_com_qrng_vectorizado(imagem_processada, matriz_qrng_processada)
    t1 = time.time()

    imagem_xor_nao = aplicar_xor_com_qrng_nao_vectorizado(imagem_processada, matriz_qrng_processada)
    t2 = time.time()

    print(f"Tempo XOR vetorizado:     {t1 - t0:.4f} s")
    print(f"Tempo XOR não vetorizado: {t2 - t1:.4f} s\n")

    # =========================
    # Logistic map vetorizado vs não vetorizado
    # =========================
    imagem_caotica_vec, chaos_seq_vec = aplicar_lm_vectorizado(imagem_xor_vec)
    t3 = time.time()

    imagem_caotica_nao, chaos_seq_nao = aplicar_lm_nao_vectorizado(imagem_xor_nao)
    t4 = time.time()

    print(f"Tempo LM vetorizado:     {t3 - t2:.4f} s")
    print(f"Tempo LM não vetorizado: {t4 - t3:.4f} s\n")

    tempo_end = time.time()
    print(f"Pipeline completo concluído em {tempo_end - tempo_start:.2f} segundos.\n")

    # =========================
    # Visualização
    # =========================
    mostrar_lado_a_lado(imagem_xor_vec, imagem_caotica_vec,
                        "XOR vetorizado", "LM vetorizado")
    mostrar_lado_a_lado(imagem_xor_nao, imagem_caotica_nao,
                        "XOR não vetorizado", "LM não vetorizado")

if __name__ == "__main__":
    main()
