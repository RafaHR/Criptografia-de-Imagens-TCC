# Criptografia de Imagens com QRNG e Mapa Caótico

Este projeto implementa um sistema de criptografia de imagens utilizando técnicas avançadas de criptografia simétrica e assimétrica, combinadas com geradores de números aleatórios quânticos (QRNG) e mapas caóticos para garantir a segurança e integridade das imagens criptografadas.

## 📋 Visão Geral

O sistema de criptografia de imagens implementado neste projeto utiliza uma combinação de técnicas criptográficas modernas para proteger dados visuais:

1. **Geradores de Números Aleatórios Quânticos (QRNG)**: Utiliza circuitos quânticos para gerar sequências aleatórias de alta qualidade
2. **Mapas Caóticos**: Aplica o mapa logístico para permutar pixels da imagem e aumentar a entropia
3. **Criptografia Híbrida**: Combina criptografia simétrica (AES) com criptografia assimétrica (Kyber)
4. **Processamento Vetorizado**: Implementação otimizada para melhor desempenho

## 🏗️ Estrutura do Projeto

```
.
├── src/
│   ├── criptografar.py              # Script principal para criptografia
│   ├── descriptografar.py           # Script principal para descriptografia
│   ├── processamento_vetorizado.py  # Comparação de desempenho entre versões vetorizadas e não vetorizadas
│   ├── modules/                     # Módulos de funcionalidades
│   │   ├── crypt_image.py           # Funções de criptografia de imagem
│   │   ├── kyber.py                 # Implementação de criptografia Kyber (simulada)
│   │   ├── rng_simulado.py          # Geração de QRNG simulado
│   │   ├── rng_quantico.py          # Geração de QRNG real (IBM Quantum)
│   │   └── utils_image.py           # Funções utilitárias para manipulação de imagens
│   ├── inputs_images/               # Imagens de entrada para testes
│   ├── matrices/                    # Matrizes QRNG geradas
│   ├── Validations/                 # Resultados e testes de validação
│   └── Apresentacao_TCC/            # Arquivos para apresentação do TCC
├── requirements.txt                 # Dependências do projeto
└── README.md                        # Este arquivo
```

## 🔧 Funcionalidades Principais

### 1. Criptografia
- **Entrada de Imagem**: Suporte para upload de imagens ou captura via webcam
- **Pré-processamento**: Cortes centrais de imagens para tamanho fixo (400x400)
- **XOR com QRNG**: Aplicação de operação XOR entre a imagem e uma matriz QRNG
- **Mapa Caótico**: Permutação dos pixels usando o mapa logístico para aumentar a entropia
- **Criptografia Híbrida**: Utilização de Kyber para proteção das chaves e AES-GCM para criptografia dos dados

### 2. Descriptografia
- **Recuperação de Dados**: Leitura do pacote criptografado gerado durante a criptografia
- **Desfazimento do Mapa Caótico**: Reversão da permutação aplicada durante a criptografia
- **Descriptografia AES**: Uso da chave simétrica para descriptografar os dados
- **Validação de Integridade**: Comparação de hashes para verificar se a imagem foi corrompida

### 3. Processamento Vetorizado
- Comparação entre implementações vetorizadas e não vetorizadas para otimização de desempenho
- Implementações otimizadas com NumPy para melhor desempenho

## 🛠️ Dependências

O projeto requer as seguintes dependências (ver `requirements.txt`):

- `opencv-python`: Para manipulação de imagens
- `numpy`: Para operações numéricas
- `matplotlib`: Para visualização de imagens
- `qiskit`: Para geração de QRNG quântico
- `pycryptodome`: Para criptografia AES
- `tkinter`: Para interface gráfica de seleção de arquivos

## 🚀 Como Usar

### Criptografia
```bash
python src/criptografar.py
```

O script irá:
1. Solicitar a entrada da imagem (upload ou captura via webcam)
2. Processar a imagem (corte central)
3. Gerar uma matriz QRNG simulada
4. Aplicar XOR com a matriz QRNG
5. Aplicar o mapa caótico
6. Criptografar os dados usando Kyber + AES
7. Salvar o pacote criptografado

### Descriptografia
```bash
python src/descriptografar.py
```

O script irá:
1. Carregar o pacote criptografado
2. Descriptografar os dados
3. Recuperar a imagem original
4. Validar a integridade com hash

## 📊 Validação e Testes

O projeto inclui diversos scripts de validação:
- **Entropia**: Cálculo da entropia das imagens
- **NPCR**: Cálculo do número de pixels de mudança relativo
- **Histograma**: Análise de distribuição de pixels
- **Comparativos**: Comparação entre diferentes implementações

## 📚 Tecnologias Utilizadas

- **Python 3.x**
- **OpenCV**: Processamento de imagens
- **NumPy**: Operações matriciais
- **Qiskit**: Geração de QRNG quântico
- **PyCryptodome**: Criptografia simétrica
- **Matplotlib**: Visualização de resultados

## 📝 Notas Importantes

1. **QRNG Real vs Simulado**: O projeto suporta tanto geração de QRNG simulado quanto real via IBM Quantum
2. **Desempenho**: Implementações vetorizadas oferecem melhor desempenho
3. **Segurança**: A combinação de técnicas criptográficas proporciona alta segurança
4. **Extensibilidade**: Estrutura modular facilita a adição de novas técnicas

## 📞 Contato

Para mais informações sobre este projeto, entre em contato com o autor do TCC.