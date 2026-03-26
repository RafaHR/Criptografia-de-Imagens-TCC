import time
import os
from dotenv import load_dotenv


import numpy as np
from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler

load_dotenv()

token= os.getenv("token")
instance = os.getenv("instance")
def gerar_matriz_qrng_real(tamanho=16, job_id=None):
    total_numeros = tamanho * tamanho
    num_bits = 8

    service = QiskitRuntimeService(channel='ibm_quantum_platform', instance=instance, token=token)
    backend = service.least_busy(simulator=False, operational=True)
    print(f"Usando backend real: {backend.name}")

    if job_id:
        # Usar resultado de job já executado
        print(f"Buscando resultado do job {job_id}...")
        job = service.job(job_id)
        job_result = job.result()
        pub_result = job_result[0]
        counts = pub_result.data.meas.get_counts()

        todos_bitstrings = []

        for bitstring, freq in counts.items():
            todos_bitstrings.extend([bitstring] * freq)

    else:
        # Executar o circuito
        qc = QuantumCircuit(num_bits)
        qc.h(range(num_bits))
        qc.measure_all()

        pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
        optimized_circuit = pm.run(qc)

        sampler = Sampler(mode=backend)
        sampler.options.default_shots = total_numeros

        print("Executando no Sampler...")
        start = time.time()

        job = sampler.run([optimized_circuit])
        pub_result = job.result()[0]

        counts = pub_result.data.meas.get_counts()

        todos_bitstrings = []

        for bitstring, freq in counts.items():
            todos_bitstrings.extend([bitstring] * freq)

        end = time.time()
        print(f"Matriz gerada em {end - start:.2f} segundos.")

    np.random.shuffle(todos_bitstrings)
    valores = np.array([int(b, 2) for b in todos_bitstrings], dtype=np.uint8)
    matriz = valores.reshape((tamanho, tamanho))

    return matriz

# Exemplo de uso:
# matriz = gerar_matriz_qrng_real(tamanho=16)  # executa na IBM
# matriz = gerar_matriz_qrng_real(tamanho=16, job_id='d2ceu435v10c73c02epg')  # usa job já rodado
# print(matriz)
