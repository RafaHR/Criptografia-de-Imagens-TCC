# upload_preprocess_e_matriz_entropy_batch.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys

# === resolver caminho para TCC/src e importar seus módulos ===
SRC_DIR = Path(__file__).resolve().parents[1]  # .../TCC/src
sys.path.insert(0, str(SRC_DIR))
from modules.crypt_image import aplicar_xor_com_qrng, aplicar_confusao_difusao

# --------- CONFIG (hard-code) ---------
IMAGE_PATH      = Path("/home/rafael-brain/GitHub/Estudo/TCC/src/inputs_images/f961be76-1d3b-4b26-bc53-bca89b0f0e18.jpg")     # << ajuste aqui
MATRICES_FOLDER = Path("/home/rafael-brain/GitHub/Estudo/TCC/src/matrices/real") # << ajuste aqui (png/jpg/bmp/npy)
N_MATRIZES      = 20                                     # 0 = usar todas
TARGET_SIZE     = 400                                      # corte central 400x400
# -------------------------------------

# ---------- IO / pré-processamento ----------
def load_image_color(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Não foi possível abrir a imagem: {path}")
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
            raise ValueError(f"O .npy não contém ndarray válido: {path}")
        return arr
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Não foi possível abrir a matriz/figura: {path}")
    return gray

def to_u8(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8 and arr.min() >= 0 and arr.max() <= 255:
        return arr
    arr = arr.astype(np.float32)
    mn, mx = float(arr.min()), float(arr.max())
    if mx == mn:
        return np.zeros_like(arr, dtype=np.uint8)
    return np.clip((arr - mn) / (mx - mn) * 255.0, 0, 255).astype(np.uint8)

# ---------- Entropia (Shannon, bits/byte) ----------
def entropy_bits(img: np.ndarray) -> float:
    """Entropia de Shannon em bits/byte (0..8)."""
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    data = img.reshape(-1)  # inclui todos os canais
    hist = np.bincount(data, minlength=256).astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 0.0
    p = hist / total
    p = p[p > 0.0]
    return float(-np.sum(p * np.log2(p)))

def main():
    try:
        # valida entradas
        if not IMAGE_PATH.exists():
            print(f"Erro: IMAGE_PATH não existe: {IMAGE_PATH}")
            return
        if not MATRICES_FOLDER.exists() or not MATRICES_FOLDER.is_dir():
            print(f"Erro: MATRICES_FOLDER inválida: {MATRICES_FOLDER}")
            return

        # carrega e pré-processa imagem
        img_rgb = load_image_color(IMAGE_PATH)
        img = preprocess_image_center(img_rgb, TARGET_SIZE)
        H, W = img.shape[:2]

        # lista arquivos de matriz
        paths = sorted([
            p for p in MATRICES_FOLDER.iterdir()
            if p.suffix.lower() in (".npy", ".png", ".jpg", ".jpeg", ".bmp", ".tiff")
        ])
        if not paths:
            print("Nenhuma matriz encontrada na pasta.")
            return
        if N_MATRIZES > 0:
            paths = paths[:N_MATRIZES]

        # processa cada matriz
        entropias = []
        nomes = []
        usados = 0
        for p in paths:
            try:
                mat = load_matrix(p)
                if mat.ndim == 3:
                    mat = to_u8(mat)[..., 0]
                mat = to_u8(mat)

                if mat.shape[:2] != (H, W):
                    print(f"[IGNORADA] {p.name}: tamanho {mat.shape[:2]} != imagem {H,W}")
                    continue

                # cifra (sem perturbar a imagem — aqui vamos comparar apenas o efeito da matriz/“chave”)
                xor_img = aplicar_xor_com_qrng(img, mat)
                cifrada, _, _ = aplicar_confusao_difusao(xor_img, mat)

                Hc = entropy_bits(cifrada)
                entropias.append(Hc)
                nomes.append(p.name)
                usados += 1
                print(f"{usados:02d} - {p.name}: Entropia = {Hc:.4f} bits")
            except Exception as e:
                print(f"[ERRO] {p.name}: {e}")

        if usados == 0:
            print("Nenhuma matriz válida processada.")
            return

        entropias = np.array(entropias, dtype=np.float64)
        media = float(entropias.mean())
        vmin  = float(entropias.min())
        vmax  = float(entropias.max())
        print(f"\nResumo ({usados} matrizes): média={media:.4f} bits | min={vmin:.4f} | max={vmax:.4f}")

        # ---------- Gráfico de linhas (matriz vs entropia) ----------
        plt.figure(figsize=(10,5))
        plt.plot(entropias, marker="o", markersize=3, linestyle="-")
        plt.xticks(ticks=np.arange(usados), labels=[f"{i+1}" for i in range(usados)], rotation=0)
        plt.xlabel("Índice da matriz (N)")
        plt.ylabel("Entropia (bits)")
        plt.title(f"Entropia por matriz de QRNG Real\nMédia={media:.4f} | Min={vmin:.4f} | Max={vmax:.4f}")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Erro: {e}")

if __name__ == "__main__":
    main()
