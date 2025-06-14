# config.py
# ==============================================================================
# PAINEL DE CONTROLE E CONFIGURAÇÕES GLOBAIS
# ==============================================================================
import os
import torch
import torch.nn as nn
import numpy as np

# --- Arquivo e Seleção de Features ---
CSV_PATH = 'data/12_2_3-Jose_Gustavo_2008.xlsx'
DATASET = 'D1'  # Opções: 'D1', 'D2', ou outro valor para usar o conjunto de features padrão

# --- Configuração Principal do Experimento ---
MODEL_TO_OPTIMIZE = 'MLP'; USE_WEIGHTS = False; VALIDATION_METHOD = 'Holdout'  # Cenário 1
# MODEL_TO_OPTIMIZE = 'MLP'; USE_WEIGHTS = False; VALIDATION_METHOD = 'K-fold'   # Cenário 2
# MODEL_TO_OPTIMIZE = 'MLP'; USE_WEIGHTS = True;  VALIDATION_METHOD = 'Holdout'  # Cenário 3
# MODEL_TO_OPTIMIZE = 'MLP'; USE_WEIGHTS = True;  VALIDATION_METHOD = 'K-fold'   # Cenário 4

# --- Configurações Técnicas ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_JOBS_GA = max(1, os.cpu_count() - 1)
K_FOLDS = 5
SEED = 42

# --- Parâmetros de Ponderação ---
WEIGHT_PARAMS = {
    'thr': 50.0,
    'w_major': 1.0,
    'w_minor': 1.5
}

# --- Parâmetros do Algoritmo Genético ---
GA_PARAMS = {
    'pop': 2,
    'gens': 5,
    'elite': 0.05,
    'mut': 0.05,
    'cx': 0.9,
    'patience': 5,
    'repeats': 2,
}

# --- Configurações de Reprodutibilidade ---
def set_seeds():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if DEVICE == 'cuda':
        torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(os.cpu_count())

# --- Mapeamento de Modelos ---
# Para facilitar a seleção dinâmica das classes de modelo
from models.mlp_space import MLPBlockSpace
from models.rf_space import RandomForestSpace

MODEL_CLASSES = {
    'MLP': MLPBlockSpace,
    'RF': RandomForestSpace
}