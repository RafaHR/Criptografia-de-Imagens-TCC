import base64
import json

import numpy as np
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes


# ================= Mock Kyber512 ==================
class MockKyber512:
    _last_ciphertext = b''
    _last_shared_secret = b''

    @staticmethod
    def generate_keypair():
        return get_random_bytes(800), get_random_bytes(2400)  # tamanho aproximado

    @staticmethod
    def encrypt(public_key):
        shared_secret = get_random_bytes(32)  # 256 bits
        ciphertext = get_random_bytes(768)    # tamanho simulado
        MockKyber512._last_ciphertext = ciphertext
        MockKyber512._last_shared_secret = shared_secret
        return ciphertext, shared_secret
     
    @staticmethod
    def decrypt(ciphertext, secret_key):
        if ciphertext == MockKyber512._last_ciphertext:
            return MockKyber512._last_shared_secret
        else:
            raise ValueError("Invalid ciphertext (no match in mock)")


kyber512 = MockKyber512()

# ===================================================

def encrypt_config(chaos_seq, matriz_qsimul, hash_original):
    """
    Recebe:
        chaos_seq      : np.array (float64, por exemplo)
        matriz_qsimul  : np.array uint8 (400x400)
        hash_original  : hash da imagem original (string ou int)

    Retorna:
        pacote criptografado (dict serializável em JSON)
    """
    dados = {
        "chaos_seq": base64.b64encode(chaos_seq).decode("utf-8"),
        "matriz_qsimul": base64.b64encode(matriz_qsimul.astype(np.uint8)).decode("utf-8"),
        "hash_original": str(hash_original)
    }
    dados_bytes = json.dumps(dados).encode("utf-8")
    
    # Kyber keypair + encrypt (simulado)
    public_key, secret_key = kyber512.generate_keypair()
    ciphertext, shared_secret = kyber512.encrypt(public_key)

    # AES-GCM
    aes_key = shared_secret[:32]
    nonce = get_random_bytes(12)
    cipher = AES.new(aes_key, AES.MODE_GCM, nonce=nonce)
    encrypted_data, tag = cipher.encrypt_and_digest(dados_bytes)

    return {
        "ciphertext_kyber": base64.b64encode(ciphertext).decode(),
        "nonce": base64.b64encode(nonce).decode(),
        "tag": base64.b64encode(tag).decode(),
        "encrypted_data": base64.b64encode(encrypted_data).decode(),
        "public_key": base64.b64encode(public_key).decode(),
        "secret_key": base64.b64encode(secret_key).decode(),       
        "shared_secret": base64.b64encode(shared_secret).decode()
    }


def decrypt_config(encrypted_package):
    """
    Recebe pacote criptografado e retorna dicionário com:
        chaos_seq      : bytes (você converte depois se quiser)
        matriz_qsimul  : np.array uint8 (400x400)
        hash_original  : string
    """
    nonce = base64.b64decode(encrypted_package["nonce"])
    tag = base64.b64decode(encrypted_package["tag"])
    encrypted_data = base64.b64decode(encrypted_package["encrypted_data"])

    #shared_secret salvo no próprio pacote
    shared_secret = base64.b64decode(encrypted_package["shared_secret"])

    aes_key = shared_secret[:32]
    cipher = AES.new(aes_key, AES.MODE_GCM, nonce=nonce)
    decrypted_data = cipher.decrypt_and_verify(encrypted_data, tag)

    dados = json.loads(decrypted_data)

    # Reconstruir chaos_seq e matriz
    dados["chaos_seq"] = base64.b64decode(dados["chaos_seq"])
    matriz_qsimul_bytes = base64.b64decode(dados["matriz_qsimul"])
    dados["matriz_qsimul"] = np.frombuffer(matriz_qsimul_bytes, dtype=np.uint8).reshape((400, 400))

    return dados
