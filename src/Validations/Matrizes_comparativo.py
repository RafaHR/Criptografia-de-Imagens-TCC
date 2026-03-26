# compare_matrices_pearson.py
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
from itertools import combinations
import seaborn as sns

# --------- CONFIG (hard-code) ---------
PASTA_REAL = Path("/home/rafael-brain/GitHub/Estudo/TCC/src/matrices/real")        # << ajuste
PASTA_SIMULADO = Path("/home/rafael-brain/GitHub/Estudo/TCC/src/matrices/simuladas")# << ajuste
N_MATRIZES = 50      # 0 = usar todas as disponíveis em cada pasta
TODOS_VS_TODOS_CRUZADO = True  # True = Real (todos) vs Sim (todos); False = pareado por índice
# -------------------------------------

def load_matrix(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        arr = np.load(str(path), allow_pickle=False).squeeze()
    else:
        arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise ValueError(f"Erro ao carregar {path}")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr

def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError(f"Tamanhos diferentes: {a.shape} vs {b.shape}")
    x = a.ravel().astype(np.float64)
    y = b.ravel().astype(np.float64)
    x -= x.mean(); y -= y.mean()
    sx = x.std(); sy = y.std()
    if sx == 0 or sy == 0:
        return 0.0
    return float(np.dot(x, y) / (len(x) * sx * sy))

def carregar_pasta(folder: Path, limite: int):
    if not folder.exists() or not folder.is_dir():
        raise RuntimeError(f"Pasta inválida: {folder}")
    paths = sorted([p for p in folder.iterdir()
                    if p.suffix.lower() in (".npy",".png",".jpg",".jpeg",".bmp",".tiff")])
    if not paths:
        raise RuntimeError(f"Nenhum arquivo encontrado em {folder}")
    if limite > 0:
        paths = paths[:limite]
    mats, nomes = [], []
    for p in paths:
        try:
            m = load_matrix(p)
            mats.append(m); nomes.append(p.name)
        except Exception as e:
            print(f"[IGNORANDO] {p.name}: {e}")
    return mats, nomes

def pares_pearson(mats):
    vals = []
    for i, j in combinations(range(len(mats)), 2):
        if mats[i].shape != mats[j].shape:
            # pula pares incompatíveis
            continue
        vals.append(pearson_corr(mats[i], mats[j]))
    return np.array(vals, dtype=np.float64)

def cruzado_pearson(mats_real, mats_sim, todos_vs_todos=True):
    vals = []
    if todos_vs_todos:
        for i in range(len(mats_real)):
            for j in range(len(mats_sim)):
                if mats_real[i].shape != mats_sim[j].shape: 
                    continue
                vals.append(pearson_corr(mats_real[i], mats_sim[j]))
    else:
        n = min(len(mats_real), len(mats_sim))
        for i in range(n):
            if mats_real[i].shape != mats_sim[i].shape: 
                continue
            vals.append(pearson_corr(mats_real[i], mats_sim[i]))
    return np.array(vals, dtype=np.float64)

def resumo(nome, arr):
    if arr.size == 0:
        print(f"{nome}: sem valores válidos.")
        return
    print(f"{nome}: n={arr.size} | média={arr.mean():.6f} | std={arr.std():.6f} | min={arr.min():.6f} | max={arr.max():.6f}")

def heatmap_correlacao(mats, titulo, limite):
    if limite <= 0: 
        return
    n = min(len(mats), limite)
    if n < 2: 
        return
    X = np.vstack([mats[i].ravel().astype(np.float64) for i in range(n)])
    # normaliza cada vetor
    X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-12)
    C = np.corrcoef(X)
    plt.figure(figsize=(5.2,4.2))
    im = plt.imshow(C, vmin=-0.02, vmax=0.02, cmap="coolwarm")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title(titulo)
    plt.tight_layout()
    plt.show()

def main():
    try:
        mats_real, nomes_real = carregar_pasta(PASTA_REAL, N_MATRIZES)
        mats_sim,  nomes_sim  = carregar_pasta(PASTA_SIMULADO, N_MATRIZES)

        # garante tamanhos compatíveis baseados na primeira matriz de cada conjunto
        if not mats_real or not mats_sim:
            print("Conjuntos vazios após leitura.")
            return
        HWR = mats_real[0].shape
        HWS = mats_sim[0].shape
        # filtra qualquer matriz que não case com as dimensões do seu próprio conjunto
        mats_real = [m for m in mats_real if m.shape == HWR]
        mats_sim  = [m for m in mats_sim  if m.shape == HWS]

        # 1) Pearson dentro de cada conjunto
        r_intra = pares_pearson(mats_real)
        s_intra = pares_pearson(mats_sim)

        # 2) Pearson cruzado (pareado por índice, ou todos-vs-todos)
        rs_cross = cruzado_pearson(mats_real, mats_sim, TODOS_VS_TODOS_CRUZADO)

        # Resumos
        print("\n=== Correlações de Pearson (par-a-par) ===")
        resumo("Dentro do Real", r_intra)
        resumo("Dentro do Simulado", s_intra)
        resumo("Cruzado Real × Simulado", rs_cross)

        # Histogramas das distribuições
        plt.figure(figsize=(15,4))
        plt.subplot(1,3,1)
        if r_intra.size:
            plt.hist(r_intra, bins=30, color="tab:blue", alpha=0.85)
        sns.histplot(r_intra, bins=30, kde=True, color="tab:blue")
        plt.title("Pearson — QRNG Real")
        plt.xlabel("Coeficiente r"); plt.ylabel("Frequência")

        plt.subplot(1,3,2)
        if s_intra.size:
            plt.hist(s_intra, bins=30, color="tab:orange", alpha=0.85)
        sns.histplot(s_intra, bins=30, kde=True, color="tab:orange")
        plt.title("Pearson - QRNG Simulado")
        plt.xlabel("Coeficiente r"); plt.ylabel("Frequência")

        plt.subplot(1,3,3)
        if rs_cross.size:
            plt.hist(rs_cross, bins=30, color="tab:green", alpha=0.85)
        sns.histplot(rs_cross, bins=30, kde=True, color="tab:green")
        plt.title("Pearson - Comparação entre real e simulado")
        plt.xlabel("Coeficiente r"); plt.ylabel("Frequência")

        plt.tight_layout()
        plt.show()



    except Exception as e:
        print(f"Erro: {e}")

if __name__ == "__main__":
    main()
