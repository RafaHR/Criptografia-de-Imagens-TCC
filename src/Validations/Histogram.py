# hist_compare_pre_vs_cipher.py
from __future__ import annotations
from pathlib import Path
from tkinter import Tk, filedialog
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys

# --- caminho para .../TCC/src e import dos módulos ---
SRC_DIR = Path(__file__).resolve().parents[1]  # .../TCC/src
sys.path.insert(0, str(SRC_DIR))
from modules.crypt_image import aplicar_xor_com_qrng, aplicar_confusao_difusao

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
def load_image_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Não foi possível abrir a imagem.")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def center_crop(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = min(size, h, w)
    y0 = (h - s) // 2
    x0 = (w - s) // 2
    return img[y0:y0+s, x0:x0+s]

def load_matrix(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        arr = np.load(str(path), allow_pickle=False).squeeze()
        if not isinstance(arr, np.ndarray):
            raise ValueError("O arquivo .npy não contém um ndarray.")
        return arr
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if g is None:
        raise ValueError("Não foi possível abrir a matriz/figura.")
    return g

def to_u8(a: np.ndarray) -> np.ndarray:
    if a.dtype == np.uint8 and a.min() >= 0 and a.max() <= 255:
        return a
    a = a.astype(np.float32)
    mn, mx = float(a.min()), float(a.max())
    if mx == mn:
        return np.zeros_like(a, dtype=np.uint8)
    return np.clip((a - mn) / (mx - mn) * 255.0, 0, 255).astype(np.uint8)

# ---------- histogramas ----------
def plot_histograms(img_pre: np.ndarray, img_cipher: np.ndarray, title_pre: str, title_cipher: str):
    # Fig: 2 imagens (topo) + 3 histogramas (RGB) ou 1 hist (gray)
    if img_pre.ndim == 3:
        fig = plt.figure(figsize=(12, 8))
        gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.2])
        ax_img_pre   = fig.add_subplot(gs[0, 0])
        ax_img_ciph  = fig.add_subplot(gs[0, 1])
        ax_legend    = fig.add_subplot(gs[0, 2])  # espaço para legenda
        ax_r = fig.add_subplot(gs[1, 0])
        ax_g = fig.add_subplot(gs[1, 1])
        ax_b = fig.add_subplot(gs[1, 2])

        # imagens
        ax_img_pre.imshow(img_pre); ax_img_pre.set_title(title_pre); ax_img_pre.axis("off")
        ax_img_ciph.imshow(img_cipher); ax_img_ciph.set_title(title_cipher); ax_img_ciph.axis("off")
        ax_legend.axis("off")

        # histogramas por canal
        colors = ['r', 'g', 'b']
        ch_names = ['R', 'G', 'B']
        axes = [ax_r, ax_g, ax_b]
        for c, ax, name in zip(range(3), axes, ch_names):
            h_pre, _  = np.histogram(img_pre[..., c].ravel(), bins=256, range=(0, 256), density=True)
            h_ciph, _ = np.histogram(img_cipher[..., c].ravel(), bins=256, range=(0, 256), density=True)
            ax.plot(h_pre,  label=f'Pré - {name}')
            ax.plot(h_ciph, label=f'Cifrada - {name}')
            ax.set_title(f'Histograma {name}')
            ax.set_xlim(0, 255)
            ax.legend(loc='upper right', fontsize=8)
        plt.tight_layout()
        plt.show()
    else:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        ax_img_pre, ax_img_ciph, ax_hist_pre, ax_hist_ciph = axes.ravel()

        ax_img_pre.imshow(img_pre, cmap='gray'); ax_img_pre.set_title(title_pre); ax_img_pre.axis("off")
        ax_img_ciph.imshow(img_cipher, cmap='gray'); ax_img_ciph.set_title(title_cipher); ax_img_ciph.axis("off")

        h_pre, _  = np.histogram(img_pre.ravel(), bins=256, range=(0, 256), density=True)
        h_ciph, _ = np.histogram(img_cipher.ravel(), bins=256, range=(0, 256), density=True)
        ax_hist_pre.plot(h_pre);  ax_hist_pre.set_title('Histograma (pré)')
        ax_hist_ciph.plot(h_ciph); ax_hist_ciph.set_title('Histograma (cifrada)')
        for ax in (ax_hist_pre, ax_hist_ciph):
            ax.set_xlim(0, 255)
            ax.set_xlabel('Intensidade'); ax.set_ylabel('Densidade')
        plt.tight_layout()
        plt.show()

# ---------- main ----------
def main():
    try:
        print("Selecione a IMAGEM (será recortada 400x400)")
        p_img = pick_file("Selecione a IMAGEM", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff")
        if not p_img:
            print("Erro: nenhuma imagem selecionada.")
            return

        print("Selecione a MATRIZ (.npy ou imagem)")
        p_mat = pick_file("Selecione a MATRIZ (.npy ou imagem)")
        if not p_mat:
            print("Erro: nenhuma matriz selecionada.")
            return

        # Pré-processamento
        img = center_crop(load_image_rgb(p_img), TARGET_SIZE)      # RGB
        mat = load_matrix(p_mat)                                   # 2D
        if mat.ndim == 3:
            mat = to_u8(mat)[..., 0]
        if mat.shape[:2] != img.shape[:2]:
            print(f"Erro: tamanhos diferentes — matriz {mat.shape[:2]} vs imagem {img.shape[:2]}.")
            return
        if mat.dtype != np.uint8:
            mat = to_u8(mat)

        # Pipeline: XOR -> confusão+difusão
        xor_img = aplicar_xor_com_qrng(img, mat)
        cipher, _, _ = aplicar_confusao_difusao(xor_img, mat)

        # Plotar histogramas (pré vs cifrada)
        plot_histograms(img, cipher, "Imagem pré-processada", "Imagem cifrada")

    except Exception as e:
        print(f"Erro: {e}")

if __name__ == "__main__":
    main()
