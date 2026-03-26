import numpy as np


# ======== Aplica XOR ========
def aplicar_xor_com_qrng(imagem_processada, matriz_qrng):
    if imagem_processada.shape[:2] != matriz_qrng.shape:
        raise ValueError("A imagem e a matriz QRNG devem ter o mesmo tamanho")
    if imagem_processada.ndim == 3 and imagem_processada.shape[2] == 3:
        xor_resultado = np.empty_like(imagem_processada, dtype=np.uint8)
        for canal in range(3):
            xor_resultado[:, :, canal] = np.bitwise_xor(imagem_processada[:, :, canal], matriz_qrng)
    else:
        xor_resultado = np.bitwise_xor(imagem_processada, matriz_qrng)
    return xor_resultado

# ======== Mapa Caótico ========
def logistic_map(size):
    r = 3.999
    x0 = 0.98892455322743
    x = x0
    seq = []
    for _ in range(size):
        x = r * x * (1 - x)
        seq.append(x)
    return np.array(seq)

def aplicar_lm(imagem):
    h, w = imagem.shape[:2]
    total = h * w
    chaos_seq = logistic_map(total)
    indices = np.argsort(chaos_seq)
    if imagem.ndim == 3:
        canais = []
        for c in range(imagem.shape[2]):
            canais.append(imagem[:, :, c].flatten()[indices].reshape(h, w))
        imagem_caotica = np.stack(canais, axis=2)
    else:
        imagem_caotica = imagem.flatten()[indices].reshape(h, w)
    return imagem_caotica, chaos_seq


# def npcr(img1: np.ndarray, img2: np.ndarray) -> float:
#     if img1.shape[:2] != img2.shape[:2]:
#         raise ValueError("NPCR: imagens com tamanhos diferentes.")
#     H, W = img1.shape[:2]
#     if img1.ndim == 3 and img2.ndim == 3:
#         diff = np.any(img1 != img2, axis=2).astype(np.uint8)
#     else:
#         diff = (img1 != img2).astype(np.uint8)
#     return diff.sum() * 100.0 / (H * W)

# ======== Difusão com feedback (mod 256) ========
def _diffuse_mod256(img: np.ndarray, ks2d: np.ndarray) -> np.ndarray:
    """
    Difusão simples: varredura raster com feedback.
    - img: uint8 HxW ou HxWx3
    - ks2d: uint8 HxW (keystream/matriz)
    Retorna uint8 com forte difusão.
    """
    if img.shape[:2] != ks2d.shape:
        raise ValueError("difusão: ks2d deve ter o mesmo HxW da imagem.")
    h, w = img.shape[:2]
    out = img.astype(np.uint16).copy()
    ks = ks2d.astype(np.uint16)

    if img.ndim == 2:
        prev = 0
        for y in range(h):
            for x in range(w):
                out[y, x] = (out[y, x] + ks[y, x] + prev) % 256
                prev = out[y, x]
        prev = 0
        for y in range(h-1, -1, -1):
            for x in range(w-1, -1, -1):
                out[y, x] = (out[y, x] + ks[y, x] + prev) % 256
                prev = out[y, x]
    else:
        C = img.shape[2]
        prev = np.zeros(C, dtype=np.uint16)
        for y in range(h):
            for x in range(w):
                out[y, x] = (out[y, x] + ks[y, x] + prev) % 256
                prev = out[y, x]
        prev = np.zeros(C, dtype=np.uint16)
        for y in range(h-1, -1, -1):
            for x in range(w-1, -1, -1):
                out[y, x] = (out[y, x] + ks[y, x] + prev) % 256
                prev = out[y, x]
    return out.astype(np.uint8)


def aplicar_confusao_difusao(imagem: np.ndarray, matriz_qrng: np.ndarray):
    """
    Mantém sua permutação caótica (confusão) e adiciona difusão com feedback.
    - Usa mesma sequência/índices para qualquer imagem do par (NPCR).
    - Usa matriz_qrng como keystream de difusão (uint8 HxW).
    """
    h, w = imagem.shape[:2]
    total = h * w

    # 1) Confusão (permutação) – exatamente sua lógica
    chaos_seq = logistic_map(total)
    indices = np.argsort(chaos_seq)
    if imagem.ndim == 3:
        canais = [imagem[:, :, c].flatten()[indices].reshape(h, w) for c in range(imagem.shape[2])]
        permutada = np.stack(canais, axis=2)
    else:
        permutada = imagem.flatten()[indices].reshape(h, w)
        
    ks = matriz_qrng
    if ks.dtype != np.uint8:
        ks = np.clip(ks, 0, 255).astype(np.uint8)

    cifrada = _diffuse_mod256(permutada, ks)
    return cifrada, chaos_seq, indices


# ======== Desfazer Mapa ========
def desfazer_lm(imagem_caotica, indices):
    h, w = imagem_caotica.shape[:2]
    reverse_indices = np.argsort(indices)
    if imagem_caotica.ndim == 3:
        canais = []
        for c in range(imagem_caotica.shape[2]):
            canais.append(imagem_caotica[:, :, c].flatten()[reverse_indices].reshape(h, w))
        return np.stack(canais, axis=2)
    else:
        return imagem_caotica.flatten()[reverse_indices].reshape(h, w)

def recuperar_imagem(imagem_codificada, resultado):
    chaos_seq = np.frombuffer(resultado["chaos_seq"], dtype=np.float64)
    indices = np.argsort(chaos_seq)
    matriz_qsimul = resultado["matriz_qsimul"]

    imagem_desarra = desfazer_lm(imagem_codificada, indices)
    if imagem_desarra.ndim == 3:
        imagem_recuperada = np.empty_like(imagem_desarra)
        for canal in range(3):
            imagem_recuperada[:, :, canal] = np.bitwise_xor(imagem_desarra[:, :, canal], matriz_qsimul)
    else:
        imagem_recuperada = np.bitwise_xor(imagem_desarra, matriz_qsimul)
    return imagem_recuperada
