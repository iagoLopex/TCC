"""
Define o espaço de busca e a lógica de avaliação para uma Rede Neural MLP.
"""
from typing import Any, Dict, List, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from numpy.typing import NDArray
from sklearn.metrics import r2_score

import config


class MLPBlockSpace:
    """
    Encapsula os hiperparâmetros, a construção, o treino e a avaliação de um
    modelo MLP para ser usado pelo Algoritmo Genético.

    Esta classe adere à interface esperada pelo `BaseGATuner`, fornecendo
    os limites dos genes e os métodos `decode` e `evaluate`.
    """

    n_layers: List[int] = list(range(1, 10))
    units: List[int] = list(range(1, 100))
    dropout: List[float] = [0.0, 0.05, 0.1, 0.15, 0.2]
    acts: List[Type[nn.Module]] = [nn.Tanh, nn.Sigmoid, nn.LeakyReLU]
    bss: List[int] = list(range(6, 65, 1))  # Batch sizes [6, 7, 8, ..., 65]
    lr_log10: List[float] = [
        1e-1, 1e-2, 1e-3, 1e-4,  # 0.1, 0.01, ...
        2e-1, 2e-2, 2e-3, 2e-4,  # 0.2, 0.02, ...
        5e-1, 5e-2, 5e-3, 5e-4,  # 0.5, 0.05, 0.005, 0.0005
    ]
    

    bounds: NDArray = np.array(
        [
            [0, len(n_layers) - 1], [0, len(units) - 1],
            [0, len(dropout) - 1], [0, len(acts) - 1],
            [0, len(bss) - 1], [0, len(lr_log10) - 1],
        ]
    )
    types: NDArray = np.array(["int", "int", "int", "int", "int", "int"])

    def __init__(
        self,
        Xtr: NDArray, ytr: NDArray, Xv: NDArray, yv: NDArray,
        use_weights: bool = False, **kwargs: Any
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
        self.Xt = torch.from_numpy(Xtr).to(config.DEVICE)
        self.yt = torch.from_numpy(ytr).to(config.DEVICE)
        self.Xv = torch.from_numpy(Xv).to(config.DEVICE)
        self.yv = torch.from_numpy(yv).to(config.DEVICE)
        self.in_dim = Xtr.shape[1]
        self.dev = config.DEVICE
        self.seed0 = config.SEED

        if use_weights:
            self.wt = self._create_weights(ytr, **kwargs)
            self.wv = self._create_weights(yv, **kwargs)
        else:
            self.wt = torch.ones_like(self.yt).to(self.dev)
            self.wv = torch.ones_like(self.yv).to(self.dev)

    def _create_weights(self, y_data: NDArray, **kwargs: Any) -> torch.Tensor:
        """Cria um tensor de pesos para um dado conjunto de alvos."""
        weights = np.where(
            y_data.flatten() > kwargs["thr"], kwargs["w_minor"], kwargs["w_major"]
        ).astype(np.float32)
        return torch.from_numpy(weights).to(self.dev)

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
            "n_layers": MLPBlockSpace.n_layers[int(gene[0])],
            "units": MLPBlockSpace.units[int(gene[1])],
            "drop": MLPBlockSpace.dropout[int(gene[2])],
            "act": MLPBlockSpace.acts[int(gene[3])],
            "batch": MLPBlockSpace.bss[int(gene[4])],
            "lr": MLPBlockSpace.lr_log10[int(gene[5])],
        }

    def _build(self, cfg: Dict[str, Any]) -> nn.Sequential:
        """
        Constrói a arquitetura do modelo MLP com base na configuração fornecida.

        Args:
            cfg (Dict[str, Any]): Dicionário de hiperparâmetros.

        Returns:
            nn.Sequential: O modelo PyTorch construído e pronto para treino.
        """
        layers = []
        in_features = self.in_dim
        drop_layer_idx = (
            np.random.randint(0, cfg["n_layers"]) if cfg["n_layers"] > 0 else -1
        )
        for i in range(cfg["n_layers"]):
            layers.extend([
                nn.Linear(in_features, cfg["units"]),
                nn.BatchNorm1d(cfg["units"]),
                cfg["act"](),
            ])
            if i == drop_layer_idx and cfg["drop"] > 0:
                layers.append(nn.Dropout(cfg["drop"]))
            in_features = cfg["units"]
        layers.append(nn.Linear(in_features, 1))
        return nn.Sequential(*layers).to(self.dev)

    def _train_model(
        self, cfg: Dict[str, Any], rep: int
    ) -> Tuple[nn.Module, List[float], List[float]]:
        """
        Executa o ciclo de treino e validação para uma configuração do modelo.

        Implementa um loop de treino com otimizador Adam, Mean Absolute Error
        (MAE) como função de perda e um critério de parada antecipada (early
        stopping) para evitar overfitting.

        Args:
            cfg (Dict[str, Any]): Dicionário de hiperparâmetros.
            rep (int): O número da repetição (para controle de semente).

        Returns:
            Tuple[nn.Module, List[float], List[float]]:
                - O modelo treinado.
                - O histórico de loss de treino por época.
                - O histórico de loss de validação por época.
        """
        torch.manual_seed(self.seed0 + rep)
        model = self._build(cfg)
        optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])
        dataset = torch.utils.data.TensorDataset(self.Xt, self.yt, self.wt.unsqueeze(1))
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=cfg["batch"], shuffle=True, drop_last=True
        )

        train_loss_history, val_loss_history = [], []
        wait_epochs, patience = 0, 5

        for _ in range(100):
            model.train()
            epoch_train_losses = []
            for xb, yb, wb in loader:
                optimizer.zero_grad()
                predictions = model(xb)
                loss = (torch.abs(predictions - yb) * wb).mean()
                loss.backward()
                optimizer.step()
                epoch_train_losses.append(loss.item())
            train_loss_history.append(np.mean(epoch_train_losses))

            model.eval()
            with torch.no_grad():
                val_preds = model(self.Xv)
                val_loss = (torch.abs(val_preds - self.yv) * self.wv.unsqueeze(1)).mean().item()
                val_loss_history.append(val_loss)

            if len(val_loss_history) > 1 and val_loss_history[-1] > val_loss_history[-2]:
                wait_epochs += 1
            else:
                wait_epochs = 0
            if wait_epochs >= patience:
                break
        return model, train_loss_history, val_loss_history

    def evaluate(self, cfg: Dict[str, Any], rep: int) -> float:
        """
        Avalia a performance de uma configuração, retornando sua pontuação de
        fitness.

        O fitness é o R² negativo, pois o AG foi projetado para minimizar a
        função objetivo.

        Args:
            cfg (Dict[str, Any]): Dicionário de hiperparâmetros.
            rep (int): O número da repetição.

        Returns:
            float: A pontuação de fitness (menor é melhor).
        """
        model, _, _ = self._train_model(cfg, rep)
        with torch.no_grad():
            y_pred_val = model(self.Xv)
            r2 = r2_score(self.yv.cpu().numpy(), y_pred_val.cpu().numpy())
        return -r2