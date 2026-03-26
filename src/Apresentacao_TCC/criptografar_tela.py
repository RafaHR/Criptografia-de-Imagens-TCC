import json
import os
import sys
import time
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2

# Ajuste de paths para módulos
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append('TCC/src/modules')

from modules.crypt_image import aplicar_lm, aplicar_xor_com_qrng, recuperar_imagem
from modules.kyber import decrypt_config, encrypt_config
from modules.rng_simulado import gerar_matriz_qrng
from modules.utils_image import (
    captura_img,
    cortar_centro,
    hash_imagem,
    mostrar_imagem,
    mostrar_lado_a_lado,  # pode nem usar mais, mas deixei
)

imagem_circuito = "/home/rafael-brain/GitHub/Estudo/TCC/src/Apresentacao_TCC/img_auxiliares/circuito_qrng.png"
imagem_mapa = "/home/rafael-brain/GitHub/Estudo/TCC/src/Apresentacao_TCC/img_auxiliares/mapa_logistico.png"


def criptografar_imagem(imagem, pasta_saida_img=None, caminho_json=None):
    """
    Recebe uma imagem (np.array BGR do OpenCV) e executa o pipeline.
    Retorna:
        caminho_img_saida, caminho_json_saida
    """
    if imagem is None:
        raise ValueError("Imagem recebida é None.")

    print("Tamanho da imagem:", imagem.shape)

    tempo_start = time.time()

    # 1) Corte central 400x400
    imagem_processada = cortar_centro(imagem, 400)
    if imagem_processada is None:
        raise RuntimeError("Erro ao cortar imagem para 400x400.")

    # Visualização opcional: original x cortada
    mostrar_lado_a_lado(imagem, imagem_processada, "Imagem original", "Imagem 400x400")

    # 2) Hash da imagem original (cortada)
    hash_original = hash_imagem(imagem_processada)
    mostrar_imagem(imagem_processada, f"Hash original: {hash_original}")
    print("Hash da imagem original:", hash_original, "\n")

    # 3) Matriz QRNG simulada
    matriz_qsimul = gerar_matriz_qrng()
    print("Matriz QRNG simulada gerada.\n")
    circuito = cv2.imread(imagem_circuito)
    if circuito is None:
        print(f"[AVISO] Não foi possível carregar a imagem do circuito em: {imagem_circuito}")
    else:
        # Se precisar, você pode redimensionar o circuito para bater com a matriz ou com 400x400
        # Exemplo: circuito = cv2.resize(circuito, (400, 400))
        mostrar_lado_a_lado(matriz_qsimul, circuito, "Matriz QRNG Simulada", "Circuito QRNG")


    # 4) XOR imagem + matriz
    imagem_xor = aplicar_xor_com_qrng(imagem_processada, matriz_qsimul)
    mostrar_lado_a_lado(imagem_processada, imagem_xor, "Imagem 400x400", "Imagem após XOR")

    # 5) Mapa caótico
    imagem_caotica_simulada, chaos_seq = aplicar_lm(imagem_xor)

    mapa = cv2.imread(imagem_mapa)
    if mapa is None:
        print(f"[AVISO] Não foi possível carregar a imagem do circuito em: {imagem_mapa}")
    else:
        # Se precisar, você pode redimensionar o circuito para bater com a matriz ou com 400x400
        # Exemplo: circuito = cv2.resize(circuito, (400, 400))
        mostrar_lado_a_lado(imagem_xor, mapa, "Imagem após XOR", "Mapa Logistico")

    mostrar_lado_a_lado(
        imagem_xor,
        imagem_caotica_simulada,
        "Após aplicar o XOR",
        "Após aplicar o mapa caótico"
    )

    # 6) Onde salvar a imagem criptografada
    if pasta_saida_img is None:
        pasta_saida_img = "/home/rafael-brain/GitHub/Estudo/TCC/src/Apresentacao_TCC/img_criptografada"

    os.makedirs(pasta_saida_img, exist_ok=True)
    caminho_img_saida = os.path.join(pasta_saida_img, "caos_img.png")
    cv2.imwrite(caminho_img_saida, imagem_caotica_simulada)

    # 7) Criptografia dos parâmetros / pacote
    #    Agora encrypt_config recebe também hash_original
    pacote = encrypt_config(chaos_seq, matriz_qsimul, hash_original)

    # 8) Salvar pacote criptografado
    if caminho_json is None:
        caminho_json = os.path.join(
            pasta_saida_img,
            "pacote_criptografado.json"
        )

    with open(caminho_json, "w") as f:
        json.dump(pacote, f)

    tempo_end = time.time()
    print(f"Criptografia concluída em {tempo_end - tempo_start:.2f} segundos.")
    print(f"Imagem criptografada salva em: {caminho_img_saida}")
    print(f"Pacote criptografado salvo em: {caminho_json}")

    return caminho_img_saida, caminho_json


def descriptografar_imagem(caminho_img_caos, caminho_json):
    """
    Recebe:
        caminho_img_caos: caminho da imagem caótica (caos_img.png)
        caminho_json    : caminho do pacote JSON criptografado

    Retorna:
        imagem_descriptografada (np.array),
        hash_ok (bool),
        hash_original_str (str),
        hash_descriptografada_str (str)
    """
    if not os.path.isfile(caminho_json):
        raise FileNotFoundError(f"JSON não encontrado: {caminho_json}")

    if not os.path.isfile(caminho_img_caos):
        raise FileNotFoundError(f"Imagem caótica não encontrada: {caminho_img_caos}")

    with open(caminho_json, "r") as f:
        pacote = json.load(f)

    time_start = time.time()

    # 1) Recupera chaos_seq, matriz_qsimul, hash_original
    resultado = decrypt_config(pacote)
    hash_original_str = str(resultado.get("hash_original", ""))

    # 2) Carrega imagem caótica
    imagem_caotica = cv2.imread(caminho_img_caos)
    if imagem_caotica is None:
        raise RuntimeError(f"Falha ao carregar a imagem caótica: {caminho_img_caos}")

    # 3) Recupera imagem original
    imagem_descriptografada = recuperar_imagem(imagem_caotica, resultado)

    # 4) Mostrar para inspeção
    mostrar_imagem(imagem_descriptografada, "Imagem descriptografada")

    # 5) Verificar hash
    hash_descriptografada = hash_imagem(imagem_descriptografada)
    hash_descriptografada_str = str(hash_descriptografada)

    hash_ok = (hash_descriptografada_str == hash_original_str)

    if hash_ok:
        print("HASH válida: imagem íntegra (hash descriptografada == hash original).")
    else:
        print("HASH inválida: possível corrupção ou alteração da imagem.")
        print(f"Hash original         : {hash_original_str}")
        print(f"Hash descriptografada : {hash_descriptografada_str}")

    tempo_end = time.time()
    print(f"Descriptografia concluída em {tempo_end - time_start:.2f} segundos.")

    return imagem_descriptografada, hash_ok, hash_original_str, hash_descriptografada_str


# =====================================================================
# GUI
# =====================================================================
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Criptografia de Imagens")
        # janela principal maior
        self.root.geometry("500x300")

        # Fonte maior para os botões
        self.btn_font = ("Arial", 12, "bold")

        # Botão Criptografar
        btn_crip = tk.Button(
            root,
            text="Criptografar",
            width=25,
            height=3,
            font=self.btn_font,
            command=self.janela_criptografar
        )
        btn_crip.pack(pady=15)

        # Botão Descriptografar
        btn_decrip = tk.Button(
            root,
            text="Descriptografar",
            width=25,
            height=3,
            font=self.btn_font,
            command=self.janela_descriptografar
        )
        btn_decrip.pack(pady=15)

        # Variáveis para armazenar caminhos
        self.caminho_imagem_crip = None
        self.caminho_imagem_decrip = None
        self.caminho_json_decrip = None

    # ===================== JANELA CRIPTOGRAFAR =====================
    def janela_criptografar(self):
        # esconde a janela principal
        self.root.withdraw()

        janela = tk.Toplevel(self.root)
        janela.title("Criptografar")
        # janela de criptografia maior
        janela.geometry("600x250")

        tk.Label(
            janela,
            text="Escolha uma opção para criptografar:",
            font=("Arial", 13, "bold")
        ).pack(pady=15)

        btn_foto = tk.Button(
            janela,
            text="Tirar foto",
            width=22,
            height=3,
            font=self.btn_font,
            command=self.tirar_foto
        )
        btn_foto.pack(pady=10)

        btn_sel_arquivo = tk.Button(
            janela,
            text="Selecionar arquivo de imagem",
            width=30,
            height=3,
            font=self.btn_font,
            command=self.selecionar_imagem_criptografar
        )
        btn_sel_arquivo.pack(pady=10)

        # se fechar a janela, volta pra principal
        def ao_fechar():
            janela.destroy()
            self.root.deiconify()  # mostra a principal de novo

        janela.protocol("WM_DELETE_WINDOW", ao_fechar)

    def tirar_foto(self):
        imagem = captura_img()
        if imagem is None:
            messagebox.showwarning("Erro", "Nenhuma imagem capturada.")
            return

        try:
            caminho_img, caminho_json = criptografar_imagem(imagem)
        except Exception as e:
            messagebox.showerror("Erro na criptografia", str(e))
            return

        self.caminho_imagem_crip = caminho_img
        messagebox.showinfo(
            "Criptografia concluída",
            f"Imagem criptografada salva em:\n{caminho_img}\n\n"
            f"Pacote JSON salvo em:\n{caminho_json}"
        )

    def selecionar_imagem_criptografar(self):
        caminho = filedialog.askopenfilename(
            title="Selecione a imagem para criptografar",
            filetypes=(("Imagens", "*.png;*.jpg;*.jpeg;*.bmp"),
                       ("Todos os arquivos", "*.*"))
        )
        if not caminho:
            return

        imagem = cv2.imread(caminho)
        if imagem is None:
            messagebox.showerror("Erro", "Falha ao carregar a imagem selecionada.")
            return

        try:
            caminho_img, caminho_json = criptografar_imagem(imagem)
        except Exception as e:
            messagebox.showerror("Erro na criptografia", str(e))
            return

        self.caminho_imagem_crip = caminho_img
        messagebox.showinfo(
            "Criptografia concluída",
            f"Imagem criptografada salva em:\n{caminho_img}\n\n"
            f"Pacote JSON salvo em:\n{caminho_json}"
        )

    # ===================== JANELA DESCRIPTOGRAFAR =====================
    def janela_descriptografar(self):
        # esconde a janela principal
        self.root.withdraw()

        janela = tk.Toplevel(self.root)
        janela.title("Descriptografar")
        # janela de descriptografia maior
        janela.geometry("650x280")

        tk.Label(
            janela,
            text="Selecione os arquivos para descriptografar:",
            font=("Arial", 13, "bold")
        ).pack(pady=15)

        # ---------- Seção IMAGEM ----------
        frame_img = tk.Frame(janela)
        frame_img.pack(fill="x", padx=15, pady=8)

        tk.Label(frame_img, text="Imagem:", font=("Arial", 11)).pack(side="left")

        self.entry_img_decrip = tk.Entry(frame_img, font=("Arial", 11))
        self.entry_img_decrip.pack(side="left", fill="x", expand=True, padx=8)

        btn_img = tk.Button(
            frame_img,
            text="Selecionar...",
            width=15,
            height=2,
            font=self.btn_font,
            command=self.selecionar_imagem_descriptografar
        )
        btn_img.pack(side="left")

        # ---------- Seção JSON ----------
        frame_json = tk.Frame(janela)
        frame_json.pack(fill="x", padx=15, pady=8)

        tk.Label(frame_json, text="JSON:", font=("Arial", 11)).pack(side="left")

        self.entry_json_decrip = tk.Entry(frame_json, font=("Arial", 11))
        self.entry_json_decrip.pack(side="left", fill="x", expand=True, padx=8)

        btn_json = tk.Button(
            frame_json,
            text="Selecionar...",
            width=15,
            height=2,
            font=self.btn_font,
            command=self.selecionar_json_descriptografar
        )
        btn_json.pack(side="left")

        btn_descrip = tk.Button(
            janela,
            text="Descriptografar",
            width=25,
            height=3,
            font=self.btn_font,
            command=self.executar_descriptografia
        )
        btn_descrip.pack(pady=20)

        def ao_fechar():
            janela.destroy()
            self.root.deiconify()  # mostra a principal

        janela.protocol("WM_DELETE_WINDOW", ao_fechar)

    def selecionar_imagem_descriptografar(self):
        caminho = filedialog.askopenfilename(
            title="Selecione a imagem criptografada",
            filetypes=(("Imagens", "*.png;*.jpg;*.jpeg;*.bmp"),
                       ("Todos os arquivos", "*.*"))
        )
        if caminho:
            self.caminho_imagem_decrip = caminho
            self.entry_img_decrip.delete(0, tk.END)
            self.entry_img_decrip.insert(0, caminho)

    def selecionar_json_descriptografar(self):
        caminho = filedialog.askopenfilename(
            title="Selecione o arquivo JSON",
            filetypes=(("Arquivos JSON", "*.json"),
                       ("Todos os arquivos", "*.*"))
        )
        if caminho:
            self.caminho_json_decrip = caminho
            self.entry_json_decrip.delete(0, tk.END)
            self.entry_json_decrip.insert(0, caminho)

    def executar_descriptografia(self):
        img = self.caminho_imagem_decrip
        js = self.caminho_json_decrip

        if not img or not js:
            messagebox.showwarning(
                "Faltando arquivo",
                "Selecione a imagem e o JSON antes de descriptografar."
            )
            return

        try:
            _, hash_ok, hash_original, hash_desc = descriptografar_imagem(img, js)
        except Exception as e:
            messagebox.showerror("Erro na descriptografia", str(e))
            return

        if hash_ok:
            messagebox.showinfo(
                "HASH",
                "HASH válida: imagem íntegra.\n\n"
                f"Hash original        : {hash_original}\n"
                f"Hash descriptografada: {hash_desc}"
            )
        else:
            messagebox.showwarning(
                "HASH",
                "HASH inválida: possível corrupção/alteração da imagem.\n\n"
                f"Hash original        : {hash_original}\n"
                f"Hash descriptografada: {hash_desc}"
            )

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
