"""
Módulo de Configuração Global (Painel de Controle).

Este arquivo centraliza todos os parâmetros e configurações utilizados no
experimento de otimização. Alterar as variáveis neste arquivo permite
controlar o fluxo de execução, os modelos utilizados, os datasets e os
hiperparâmetros do Algoritmo Genético sem modificar o código principal.
"""

import os
from typing import Any, Dict

import numpy as np
import torch


# --- Parâmetros do Experimento ---
CSV_PATH: str = "data/12_2_3-Jose_Gustavo_2008.xlsx"
DATASET: str = "D3"            # Opções: 'D1', 'D2', 'D3' (genérico)

# --- Configuração do Modelo e Validação ---
MODEL_TO_OPTIMIZE: str = "RF"  # Opções: 'MLP', 'RF'
USE_WEIGHTS: bool = True
VALIDATION_METHOD: str = "Holdout"  # Opções: 'Holdout', 'K-fold'

# --- Configurações Técnicas ---
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
N_JOBS_GA: int = max(1, os.cpu_count() - 1)
K_FOLDS: int = 5
SEED: int = 42

# --- Parâmetros de Ponderação de Amostras (se USE_WEIGHTS = True) ---
WEIGHT_PARAMS: Dict[str, float] = {"thr": 50.0, "w_major": 1.0, "w_minor": 1.3}

# --- Parâmetros do Algoritmo Genético ---
GA_PARAMS: Dict[str, Any] = {
    "pop": 2,        # Tamanho da população
    "gens": 3,       # Número de gerações
    "elite": 0.1,    # Fração da elite
    "mut": 0.1,      # Probabilidade de mutação
    "cx": 0.9,       # Probabilidade de crossover
    "patience": 10,  # Gerações sem melhora para parada antecipada
    "repeats": 1,    # Repetições na avaliação de cada indivíduo
}


def set_seeds() -> None:
    """
    Define as sementes de aleatoriedade para NumPy e PyTorch para garantir
    a reprodutibilidade dos experimentos.
    """
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(os.cpu_count())