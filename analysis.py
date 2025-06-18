"""
Módulo de Análise de Resultados e Visualização.

Este módulo contém uma suíte de funções para calcular métricas de performance,
comparar o desempenho de treino e teste, e gerar visualizações detalhadas
para a análise do modelo final.
"""
import logging
import os
from typing import Any, Dict, List, Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from numpy.typing import NDArray
from rich.console import Console
from rich.table import Table
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

import config


def calculate_regression_metrics(y_true: NDArray, y_pred: NDArray) -> Dict[str, float]:
    """Calcula um conjunto de métricas de regressão (R², MAE, RMSE, MAPE)."""
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    r2 = r2_score(y_true_flat, y_pred_flat)
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    rmse = np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))

    non_zero_mask = y_true_flat != 0
    mape = (
        np.mean(np.abs((y_true_flat[non_zero_mask] - y_pred_flat[non_zero_mask]) / y_true_flat[non_zero_mask])) * 100
        if np.any(non_zero_mask) else np.nan
    )
    return {"R²": r2, "MAE": mae, "RMSE": rmse, "MAPE (%)": mape}


def _get_predictions(
    model: Any, X_data_scaled: NDArray, scaler_y: StandardScaler
) -> NDArray:
    """Gera previsões e reverte a escala para os valores originais."""
    if config.MODEL_TO_OPTIMIZE == "MLP":
        model.eval()
        with torch.no_grad():
            tensor_X = torch.from_numpy(X_data_scaled).to(config.DEVICE)
            predictions_scaled = model(tensor_X)
        return scaler_y.inverse_transform(predictions_scaled.cpu().numpy())
    else:
        predictions_scaled = model.predict(X_data_scaled)
        return scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1))


def _save_plot(figure: plt.Figure, results_path: str, filename: str) -> None:
    """Salva uma figura do Matplotlib em um arquivo e a fecha."""
    filepath = os.path.join(results_path, filename)
    figure.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(figure)
    logging.info(f"Gráfico salvo em: {filepath}")


def _analyze_performance_by_bins(
    y_true: NDArray, y_pred: NDArray, console: Console
) -> None:
    """Analisa e exibe a performance do modelo por faixas de CBR."""
    df = pd.DataFrame({"Real": y_true.flatten(), "Predito": y_pred.flatten()})
    bins = [0, 15, 30, 50, 80, np.inf]
    labels = ["0-15", "16-30", "31-50", "51-80", ">80"]
    df["CBR_Faixa"] = pd.cut(df["Real"], bins=bins, labels=labels, right=True)

    console.print("\n[bold]--- Análise de Performance por Faixa de CBR (Teste) ---[/bold]")
    table = Table(title="Métricas por Faixa de CBR")
    table.add_column("Faixa CBR", style="cyan")
    table.add_column("Nº Amostras", style="magenta")
    table.add_column("R²", style="green")
    table.add_column("RMSE", style="yellow")

    for label in labels:
        subset = df[df["CBR_Faixa"] == label]
        if not subset.empty:
            metrics = calculate_regression_metrics(
                subset[["Real"]].values, subset[["Predito"]].values
            )
            table.add_row(
                label, str(len(subset)), f"{metrics['R²']:.3f}", f"{metrics['RMSE']:.2f}"
            )
    console.print(table)

def _plot_ga_convergence(fitness_history: List[float], results_path: str) -> None:
    """Gera e salva o gráfico de convergência do Algoritmo Genético."""
    fig = plt.figure(figsize=(10, 6))
    r2_history = [-f for f in fitness_history]
    plt.plot(range(1, len(r2_history) + 1), r2_history, marker='o', linestyle='-', color='b')
    plt.xlabel("Geração")
    plt.ylabel("Melhor R² na População")
    plt.title("Curva de Convergência do Algoritmo Genético")
    plt.grid(True)
    plt.xticks(range(1, len(r2_history) + 1))
    _save_plot(fig, results_path, "ga_convergence_curve.png")

def _plot_prediction_tracking(y_true: NDArray, y_pred: NDArray, results_path: str) -> None:
    """Gera e salva o gráfico de rastreamento de predições vs. valores reais."""
    fig = plt.figure(figsize=(15, 7))
    sample_indices = range(len(y_true))
    plt.plot(sample_indices, y_true.flatten(), label='Valores Reais', color='gray', marker='o', linestyle='-')
    plt.plot(sample_indices, y_pred.flatten(), label='Valores Preditos', color='darkorange', marker='x', linestyle='--')
    plt.xlabel("Índice da Amostra de Teste")
    plt.ylabel("CBR")
    plt.title("Rastreamento de Valores Preditos vs. Reais")
    plt.legend()
    plt.grid(True)
    _save_plot(fig, results_path, "prediction_tracking.png")


def analyze_and_plot(
    final_model: Any, scaler_y: StandardScaler, X_dev_scaled: NDArray,
    y_dev_real: NDArray, X_test_scaled: NDArray, y_test_real: NDArray,
    results_path: str, feature_names: List[str],
    generation_history: List[Dict],
    fitness_history: List[float],
    train_hist: Optional[List[float]] = None,
    val_hist: Optional[List[float]] = None,
) -> None:
    """Executa a suíte completa de análise de performance, métricas e gráficos."""
    console = Console()
    sns.set_style("whitegrid")
    logging.info("\n--- Etapa 6: Análise Detalhada do Melhor Modelo ---")

    pred_dev_real = _get_predictions(final_model, X_dev_scaled, scaler_y)
    pred_test_real = _get_predictions(final_model, X_test_scaled, scaler_y)

    metrics_dev = calculate_regression_metrics(y_dev_real, pred_dev_real)
    metrics_test = calculate_regression_metrics(y_test_real, pred_test_real)

    logging.info("Métricas de Performance", extra={"json_fields": {"dev_set": metrics_dev, "test_set": metrics_test}})

    console.print("\n[bold]--- Tabela de Performance (Treino vs. Teste) ---[/bold]")
    table = Table(title="Comparativo de Métricas")
    # ... (código da tabela sem alterações) ...
    table.add_column("Métrica", style="cyan")
    table.add_column("Treino (Dev Set)", style="yellow")
    table.add_column("Teste", style="green")
    for metric_name in metrics_dev:
        table.add_row(metric_name, f"{metrics_dev[metric_name]:.3f}", f"{metrics_test[metric_name]:.3f}")
    console.print(table)

    # --- INÍCIO DA CHAMADA DAS NOVAS ANÁLISES ---
    console.print("\n[bold]--- Análises Adicionais da Otimização e Performance ---[/bold]")
    _plot_ga_convergence(fitness_history, results_path)
    _plot_prediction_tracking(y_test_real, pred_test_real, results_path)
    # --- FIM DA CHAMADA DAS NOVAS ANÁLISES ---

    if config.MODEL_TO_OPTIMIZE == "RF":
        importances = final_model.feature_importances_
        feature_df = pd.DataFrame({"Feature": feature_names, "Importância": importances}).sort_values(by="Importância", ascending=False)
        fig = plt.figure(figsize=(10, max(6, len(feature_names) * 0.4)))
        sns.barplot(x="Importância", y="Feature", data=feature_df, palette="viridis")
        plt.title("Importância das Features (Random Forest)")
        fig.tight_layout()
        _save_plot(fig, results_path, "feature_importance.png")

    if config.MODEL_TO_OPTIMIZE == "MLP" and train_hist and val_hist:
        fig = plt.figure(figsize=(8, 5))
        plt.plot(train_hist, label="Loss de Treino")
        plt.plot(val_hist, label="Loss de Validação")
        plt.xlabel("Épocas")
        plt.ylabel("Loss (MAE Normalizado)")
        plt.title("Curva de Aprendizado do Modelo Final")
        plt.legend()
        _save_plot(fig, results_path, "learning_curve.png")

    fig_preds, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.scatterplot(x=y_test_real.flatten(), y=pred_test_real.flatten(), ax=axes[0])
    axes[0].plot([y_test_real.min(), y_test_real.max()], [y_test_real.min(), y_test_real.max()], "r--")
    axes[0].set(xlabel="CBR Real", ylabel="CBR Predito", title="Real vs. Predito (Teste)")
    axes[0].set_aspect("equal", "box")
    residuals = y_test_real.flatten() - pred_test_real.flatten()
    sns.histplot(residuals, kde=True, ax=axes[1])
    axes[1].set(xlabel="Resíduo (Real - Predito)", title="Distribuição dos Resíduos")
    fig_preds.tight_layout()
    _save_plot(fig_preds, results_path, "predictions_and_residuals.png")

    _analyze_performance_by_bins(y_test_real, pred_test_real, console)