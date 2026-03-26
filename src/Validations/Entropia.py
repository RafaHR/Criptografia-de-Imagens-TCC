# upload_preprocess_e_matriz_entropy.py
from __future__ import annotations
from pathlib import Path
from tkinter import Tk, filedialog
import numpy as np
import cv2, sys
import matplotlib.pyplot as plt

# === resolver caminho para TCC/src e importar seus módulos ===
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
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError("Não foi possível abrir a matriz/figura.")
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
    """Entropia de Shannon em bits/byte, sem converter para cinza e sem reescalar."""
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

# ---------- main ----------
def main():
    try:
        print("Selecione a IMAGEM para pré-processar")
        p_img = pick_file("Selecione a IMAGEM", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff")
        if not p_img:
            print("Erro: nenhuma imagem selecionada.")
            return

        print("Selecione a MATRIZ (.npy ou imagem)")
        p_mat = pick_file("Selecione a MATRIZ (.npy ou imagem)")
        if not p_mat:
            print("Erro: nenhuma matriz selecionada.")
            return

        # imagem e corte central
        img = preprocess_image_center(load_image_color(p_img), TARGET_SIZE)

        # matriz
        mat = load_matrix(p_mat)
        if mat.ndim == 3:
            mat = to_u8(mat)[..., 0]
        if mat.shape[:2] != img.shape[:2]:
            print("Erro: matriz e imagem com tamanhos diferentes.")
            return
        mat = to_u8(mat)

        # N execuções com PERTURBAÇÃO de 1 pixel na imagem
        n = int(input("Quantas repetições deseja para o teste de Entropia? "))

        vals = []
        for i in range(1, n + 1):
            img_pert = img.copy()
            y = np.random.randint(0, img.shape[0]); x = np.random.randint(0, img.shape[1])
            if img.ndim == 3:
                img_pert[y, x] = np.random.randint(0, 256, size=img.shape[2], dtype=np.uint8)
            else:
                img_pert[y, x] = np.random.randint(0, 256, dtype=np.uint8)

            # >>> usa a IMAGEM PERTURBADA <<<
            xor_img = aplicar_xor_com_qrng(img_pert, mat)
            cifrada, _, _ = aplicar_confusao_difusao(xor_img, mat)

            H = entropy_bits(cifrada)
            vals.append(H)
            print(f"{i:02d} -- Entropia = {H:.4f} bits")

        vals = np.array(vals, dtype=np.float64)
        mean_v = float(vals.mean())
        min_v  = float(vals.min())
        max_v  = float(vals.max())
        print(f"\nMédia da Entropia ({n} execuções): {mean_v:.4f} bits")
        print(f"Valor mínimo: {min_v:.4f} bits")
        print(f"Valor máximo: {max_v:.4f} bits")

        # --------- GRÁFICO DE LINHAS ---------
        plt.figure(figsize=(10,5))
        plt.plot(vals, marker="o", markersize=3, linestyle="-")
        plt.xlabel("Execução")
        plt.ylabel("Entropia (bits)")
        # plt.ylim(7.8, 8.01)  # opcional: desscomente para fixar a escala
        plt.title(f"Entropia ao longo de {n} interações em QRNG Simulado\nMédia={mean_v:.4f} | Min={min_v:.4f} | Max={max_v:.4f}")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Erro: {e}")

if __name__ == "__main__":
    main()
