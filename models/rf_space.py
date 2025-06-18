"""
Define o espaço de busca e a lógica de avaliação para um modelo Random Forest.
"""
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

import config


class RandomForestSpace:
    """
    Encapsula os hiperparâmetros, treino e avaliação de um modelo Random
    Forest para ser usado pelo Algoritmo Genético.

    Esta classe adere à interface esperada pelo `BaseGATuner`, fornecendo
    os limites dos genes e os métodos `decode` e `evaluate`. A seleção de
    hiperparâmetros foi expandida para uma busca mais granular e completa.
    """

    # De 50 a 500 árvores, com passo de 50.
    n_estimators_opts: List[int] = list(range(50, 501, 50))

    # De 5 a 29 de profundidade, com passo de 3, mais a opção de árvore completa.
    max_depth_opts: List[Optional[int]] = list(range(5, 31, 3)) + [None]

    # Todos os valores inteiros de 2 a 20 para critério de divisão.
    min_samples_split_opts: List[int] = list(range(2, 21))

    # Todos os valores inteiros de 1 a 15 para o mínimo em uma folha.
    min_samples_leaf_opts: List[int] = list(range(1, 16))

    # Opção "sqrt" e uma variação de 10% a 100% do total de features.
    max_features_float_opts: List[float] = np.linspace(0.1, 1.0, 10).round(2).tolist()
    max_features_opts: List[Union[str, float]] = ["sqrt"] + max_features_float_opts

    bounds: NDArray = np.array(
        [
            [0, len(n_estimators_opts) - 1],
            [0, len(max_depth_opts) - 1],
            [0, len(min_samples_split_opts) - 1],
            [0, len(min_samples_leaf_opts) - 1],
            [0, len(max_features_opts) - 1],
        ]
    )
    types: NDArray = np.array(["int", "int", "int", "int", "int"])

    def __init__(
        self,
        Xtr: NDArray,
        ytr: NDArray,
        Xv: NDArray,
        yv: NDArray,
        use_weights: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Inicializa o espaço de busca com os dados de treino e validação.

        Args:
            Xtr (NDArray): Features de treino.
            ytr (NDArray): Alvo de treino.
            Xv (NDArray): Features de validação.
            yv (NDArray): Alvo de validação.
            use_weights (bool): Se verdadeiro, aplica pesos às amostras.
            **kwargs: Parâmetros de ponderação ('thr', 'w_minor', 'w_major').
        """
        self.Xtr, self.ytr = Xtr, ytr.ravel()
        self.Xv, self.yv = Xv, yv.ravel()
        self.seed0 = config.SEED
        if use_weights:
            self.w_tr = np.where(
                self.ytr > kwargs["thr"], kwargs["w_minor"], kwargs["w_major"]
            ).astype(np.float32)
        else:
            self.w_tr = np.ones_like(self.ytr)

    @staticmethod
    def decode(gene: NDArray) -> Dict[str, Any]:
        """
        Decodifica um gene (array de índices) em um dicionário de
        hiperparâmetros legíveis.

        Args:
            gene (NDArray): O indivíduo do AG a ser decodificado.

        Returns:
            Dict[str, Any]: Um dicionário com os hiperparâmetros nominais.
        """
        return {
            "n_estimators": RandomForestSpace.n_estimators_opts[int(gene[0])],
            "max_depth": RandomForestSpace.max_depth_opts[int(gene[1])],
            "min_samples_split": RandomForestSpace.min_samples_split_opts[int(gene[2])],
            "min_samples_leaf": RandomForestSpace.min_samples_leaf_opts[int(gene[3])],
            "max_features": RandomForestSpace.max_features_opts[int(gene[4])],
        }

    def _train_model(
        self, cfg: Dict[str, Any], rep: int
    ) -> Tuple[RandomForestRegressor, None, None]:
        """
        Executa o treino para uma configuração do modelo Random Forest.

        Args:
            cfg (Dict[str, Any]): Dicionário de hiperparâmetros.
            rep (int): O número da repetição (para controle de semente).

        Returns:
            Tuple[RandomForestRegressor, None, None]:
                - O modelo treinado.
                - None, None para manter a consistência com a interface do MLP.
        """
        model = RandomForestRegressor(**cfg, random_state=self.seed0 + rep, n_jobs=-1)
        model.fit(self.Xtr, self.ytr, sample_weight=self.w_tr)
        return model, None, None

    def evaluate(self, cfg: Dict[str, Any], rep: int) -> float:
        """
        Avalia a performance de uma configuração, retornando sua pontuação de
        fitness (R² negativo).

        Args:
            cfg (Dict[str, Any]): Dicionário de hiperparâmetros.
            rep (int): O número da repetição.

        Returns:
            float: A pontuação de fitness (menor é melhor).
        """
        model, _, _ = self._train_model(cfg, rep)
        r2 = r2_score(self.yv, model.predict(self.Xv))
        return -r2