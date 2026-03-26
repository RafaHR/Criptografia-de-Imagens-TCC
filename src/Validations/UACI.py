# upload_preprocess_e_matriz_uaci.py
from __future__ import annotations
from pathlib import Path
from tkinter import Tk, filedialog
import numpy as np
import cv2, sys, random
import matplotlib.pyplot as plt

# === resolver caminho para TCC/src e importar seus módulos ===
SRC_DIR = Path(__file__).resolve().parents[1]  # .../TCC/src
sys.path.insert(0, str(SRC_DIR))
from modules.crypt_image import (
    aplicar_xor_com_qrng,
    aplicar_confusao_difusao,
)

TARGET_SIZE = 400  # recorte central 400x400

# ---------- upload ----------
def pick_file(title: str, patterns="*.png;*.jpg;*.jpeg;*.bmp;*.tiff;*.npy") -> Path | None:
    root = Tk(); root.withdraw()
    fpath = filedialog.askopenfilename(
        title=title,
        filetypes=[
            ("Arquivos suportados", patterns),
            ("Imagens", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff"),
            ("NumPy array", "*.npy"),
            ("Todos", "*.*"),
        ],
    )
    root.update(); root.destroy()
    return Path(fpath) if fpath else None

# ---------- IO / pré-processamento ----------
def load_image_color(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Não foi possível abrir a imagem.")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def preprocess_image_center(img_rgb: np.ndarray, target: int = TARGET_SIZE) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    s = min(target, h, w)
    y0 = (h - s) // 2
    x0 = (w - s) // 2
    return img_rgb[y0:y0+s, x0:x0+s]

def load_matrix(path: Path) -> np.ndarray:
    ext = path.suffix.lower()
    if ext == ".npy":
        arr = np.load(str(path), allow_pickle=False).squeeze()
        if not isinstance(arr, np.ndarray):
            raise ValueError("O arquivo .npy não contém um ndarray.")
        return arr
    else:
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError("Não foi possível abrir a matriz/figura.")
        return gray

# ---------- utilidades ----------
def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8 and arr.min() >= 0 and arr.max() <= 255:
        return arr
    arr = arr.astype(np.float32)
    mn, mx = float(np.min(arr)), float(np.max(arr))
    if mx == mn:
        return np.zeros_like(arr, dtype=np.uint8)
    scaled = (arr - mn) / (mx - mn) * 255.0
    return np.clip(scaled, 0, 255).astype(np.uint8)

def shapes_hw(arr: np.ndarray) -> tuple[int, int]:
    return arr.shape[:2]

def perturbar_pixel_aleatorio(img: np.ndarray) -> np.ndarray:
    p = img.copy()
    y = np.random.randint(0, img.shape[0])
    x = np.random.randint(0, img.shape[1])

    if img.ndim == 3:
        # Substitui o pixel por um novo valor aleatório ≠ do original (todos os canais)
        novo = np.random.randint(0, 256, size=(img.shape[2],), dtype=np.uint8)
        while np.all(novo == p[y, x]):
            novo = np.random.randint(0, 256, size=(img.shape[2],), dtype=np.uint8)
        p[y, x] = novo
    else:
        novo = np.random.randint(0, 256, dtype=np.uint8)
        while int(novo) == int(p[y, x]):
            novo = np.random.randint(0, 256, dtype=np.uint8)
        p[y, x] = novo
    return p

def uaci(a: np.ndarray, b: np.ndarray) -> float:
    """UACI em %, SEM normalização por imagem."""
    if a.shape[:2] != b.shape[:2]:
        raise ValueError("UACI: tamanhos diferentes.")
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    if b.dtype != np.uint8:
        b = np.clip(b, 0, 255).astype(np.uint8)
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
    if a.ndim == 3:
        diff = diff.mean(axis=2)
    return float(diff.mean()) * 100.0 / 255.0

# ---------- main ----------
def main():
    try:
        print("Selecione a IMAGEM para pré-processar")
        p_img = pick_file("Selecione a IMAGEM", patterns="*.png;*.jpg;*.jpeg;*.bmp;*.tiff")
        if not p_img:
            print("Erro: nenhuma imagem selecionada.")
            return

        print("Selecione a MATRIZ (.npy ou imagem)")
        p_mat = pick_file("Selecione a MATRIZ (.npy ou imagem)")
        if not p_mat:
            print("Erro: nenhuma matriz selecionada.")
            return

        img_rgb = load_image_color(p_img)
        img_proc = preprocess_image_center(img_rgb, TARGET_SIZE)
        H, W = shapes_hw(img_proc)

        mat = load_matrix(p_mat)
        if mat.ndim == 3:
            mat = normalize_to_uint8(mat)[..., 0]
        if shapes_hw(mat) != (H, W):
            print(f"Erro: a matriz tem tamanho {shapes_hw(mat)} e a imagem {H,W}.")
            return
        if mat.dtype != np.uint8:
            mat = normalize_to_uint8(mat)

        n = int(input("Quantas repetições deseja para o teste de UACI? "))

        resultados = []
        for i in range(1, n+1):
            img_pert = perturbar_pixel_aleatorio(img_proc)

            # pipeline: XOR -> confusão+difusão (mesma matriz dentro da execução)
            xor_a = aplicar_xor_com_qrng(img_proc, mat)
            xor_b = aplicar_xor_com_qrng(img_pert, mat)

            cdf_a, _, _ = aplicar_confusao_difusao(xor_a, mat)
            cdf_b, _, _ = aplicar_confusao_difusao(xor_b, mat)

            valor = uaci(cdf_a, cdf_b)
            resultados.append(valor)
            print(f"{i} -- UACI = {valor:.4f}%")

        media  = float(np.mean(resultados))
        minimo = float(np.min(resultados))
        maximo = float(np.max(resultados))
        print(f"\nMédia final UACI em {n} execuções: {media:.4f}%")
        print(f"Valor mínimo: {minimo:.4f}%")
        print(f"Valor máximo: {maximo:.4f}%")

        # --------- GRÁFICO DE LINHAS ---------
        plt.figure(figsize=(10,5))
        plt.plot(resultados, marker="o", markersize=3, linestyle="-", color="purple")
        plt.xlabel("Execução")
        plt.ylabel("UACI (%)")
        plt.title(f"UACI ao longo de {n} interações com QRNG Simulado\nMédia={media:.4f}% | Min={minimo:.4f}% | Max={maximo:.4f}%")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Erro: {e}")

if __name__ == "__main__":
    main()
