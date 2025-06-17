"""
Módulo de Avaliação de Modelos.

Contém a lógica para avaliar uma configuração de hiperparâmetros,
seja via Holdout ou K-Fold. Também define a classe 'GenericSpace', que
atua como uma ponte entre o otimizador genérico e os modelos específicos.
"""
from typing import Any, Dict, Type

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import config
from models.mlp_space import MLPBlockSpace
from models.rf_space import RandomForestSpace

# O dicionário de mapeamento de modelos agora vive aqui para evitar importação circular.
MODEL_CLASSES: Dict[str, Type] = {"MLP": MLPBlockSpace, "RF": RandomForestSpace}


def evaluate_gene_configuration(
    cfg: Dict[str, Any],
    rep: int,
    X_dev: NDArray,
    y_dev: NDArray,
    X_train_s: NDArray,
    y_train_s: NDArray,
    X_valid_s: NDArray,
    y_valid_s: NDArray,
) -> float:
    """
    Avalia a performance de uma configuração de hiperparâmetros (gene).

    Esta função lida com a lógica de validação Holdout ou K-Fold para
    calcular a pontuação de fitness de uma configuração.

    Args:
        cfg (Dict[str, Any]): Dicionário de hiperparâmetros decodificado.
        rep (int): O número da repetição (para controle de semente).
        X_dev (NDArray): Dados de desenvolvimento (para K-Fold).
        y_dev (NDArray): Alvo de desenvolvimento (para K-Fold).
        X_train_s (NDArray): Features de treino escalonadas (para Holdout).
        y_train_s (NDArray): Alvo de treino escalonado (para Holdout).
        X_valid_s (NDArray): Features de validação escalonadas (para Holdout).
        y_valid_s (NDArray): Alvo de validação escalonado (para Holdout).

    Returns:
        float: A pontuação de fitness (menor é melhor).
    """
    space_class = MODEL_CLASSES[config.MODEL_TO_OPTIMIZE]
    params = {"use_weights": config.USE_WEIGHTS, **config.WEIGHT_PARAMS}

    if config.VALIDATION_METHOD == "Holdout":
        space = space_class(X_train_s, y_train_s, X_valid_s, y_valid_s, **params)
        return space.evaluate(cfg, rep)

    elif config.VALIDATION_METHOD == "K-fold":
        kf = KFold(n_splits=config.K_FOLDS, shuffle=True, random_state=config.SEED)
        fold_scores = []
        for train_idx, val_idx in kf.split(X_dev, y_dev):
            X_train_f, X_val_f = X_dev[train_idx], X_dev[val_idx]
            y_train_f, y_val_f = y_dev[train_idx], y_dev[val_idx]

            scaler_x = StandardScaler().fit(X_train_f)
            X_train_sf, X_val_sf = scaler_x.transform(X_train_f), scaler_x.transform(X_val_f)

            scaler_y = StandardScaler().fit(y_train_f)
            y_train_sf, y_val_sf = scaler_y.transform(y_train_f), scaler_y.transform(y_val_f)

            space_fold = space_class(X_train_sf, y_train_sf, X_val_sf, y_val_sf, **params)
            fold_scores.append(space_fold.evaluate(cfg, rep))
        return np.mean(fold_scores)

    raise ValueError(f"Método de validação desconhecido: {config.VALIDATION_METHOD}")


class GenericSpace:
    """
    Atua como uma interface (Adapter) entre o `BaseGATuner` e as classes
    de modelo específicas.
    """

    def __init__(self, model_type: str, **kwargs: Any) -> None:
        """
        Inicializa o espaço genérico.

        Args:
            model_type (str): O tipo de modelo a ser usado ('MLP' ou 'RF').
            **kwargs: Argumentos de dados (X_dev, etc.) para a avaliação.
        """
        self.base_class: Type = MODEL_CLASSES[model_type]
        self.bounds: NDArray = self.base_class.bounds
        self.types: NDArray = self.base_class.types
        self.eval_kwargs: Dict[str, Any] = kwargs

    def decode(self, gene: NDArray) -> Dict[str, Any]:
        """Delega a decodificação do gene para a classe de modelo base."""
        return self.base_class.decode(gene)

    def evaluate(self, cfg: Dict[str, Any], rep: int) -> float:
        """Delega a avaliação da configuração para a função principal."""
        return evaluate_gene_configuration(cfg, rep, **self.eval_kwargs)