"""
Módulo que implementa a classe base para o Algoritmo Genético (AG).

Este otimizador genérico é projetado para encontrar os melhores
hiperparâmetros para qualquer problema que possa ser definido por uma
classe de 'espaço' que forneça limites, decodificação e uma função de avaliação.
"""
import logging
import multiprocessing as mp
from typing import Any, Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray
from tqdm.auto import tqdm


class BaseGATuner:
    """
    Implementa um Algoritmo Genético para otimização de hiperparâmetros.

    Este otimizador executa um processo evolucionário que inclui seleção
    de elite, crossover e mutação para encontrar a melhor combinação de
    hiperparâmetros para um determinado problema.

    Atributos:
        space (Any): Objeto que define o espaço de busca.
        seed (int): Semente para reprodutibilidade.
        n_jobs (int): Número de processos paralelos para avaliação.
        pop (int): Tamanho da população.
        gens (int): Número máximo de gerações.
        elite (float): Fração da elite que sobrevive.
        mut (float): Probabilidade de mutação de um gene.
        cx (float): Probabilidade de crossover entre pais.
        patience (int): Gerações sem melhora para parada antecipada.
        repeats (int): Repetições na avaliação de cada indivíduo.
    """

    def __init__(self, space: Any, seed: int, n_jobs: int, **kwargs: Any) -> None:
        """Inicializa o otimizador do Algoritmo Genético."""
        self.__dict__.update(kwargs)
        self.space = space
        self.seed0 = seed
        self.n_jobs = n_jobs
        self.elite_n = max(1, int(self.elite * self.pop))
        self.rng = np.random.default_rng(seed)
        self.history: List[float] = []
        self.generation_history: List[Dict[str, Any]] = []

    def _rand(self) -> NDArray:
        """Gera um indivíduo (gene) aleatório dentro dos limites definidos."""
        gene = [
            self.rng.integers(lo, hi + 1) if t == "int" else self.rng.uniform(lo, hi)
            for (lo, hi), t in zip(self.space.bounds, self.space.types)
        ]
        return np.array(gene)

    def _mut(self, gene: NDArray) -> NDArray:
        """Aplica uma mutação a um indivíduo com base na probabilidade `self.mut`."""
        for i, gene_type in enumerate(self.space.types):
            if self.rng.random() < self.mut:
                lo, hi = self.space.bounds[i]
                if gene_type == "int":
                    gene[i] = np.clip(gene[i] + self.rng.integers(-1, 2), lo, hi)
                else:
                    noise = self.rng.normal(0, 0.1 * (hi - lo))
                    gene[i] = np.clip(gene[i] + noise, lo, hi)
        return gene

    def _cx(self, parent1: NDArray, parent2: NDArray) -> Tuple[NDArray, NDArray]:
        """Executa o crossover uniforme entre dois pais com probabilidade `self.cx`."""
        if self.rng.random() > self.cx:
            return parent1.copy(), parent2.copy()
        mask = self.rng.random(len(parent1)) < 0.5
        child1, child2 = parent1.copy(), parent2.copy()
        child1[mask], child2[mask] = parent2[mask], parent1[mask]
        return child1, child2

    def _score(self, gene: NDArray) -> float:
        """Calcula a pontuação de fitness de um indivíduo."""
        cfg = self.space.decode(gene)
        scores = [self.space.evaluate(cfg, i) for i in range(self.repeats)]
        return np.mean(scores)

    def _eval_pop(self, population: List[NDArray]) -> NDArray[np.float64]:
        """Avalia uma população inteira, usando paralelismo se `n_jobs > 1`."""
        if self.n_jobs == 1:
            return np.array([self._score(g) for g in population])
        with mp.Pool(self.n_jobs) as pool:
            return np.array(pool.map(self._score, population))

    def run(self) -> Tuple[NDArray, float, List[Dict[str, Any]], List[float], str]:
        """
        Executa o ciclo completo do Algoritmo Genético.

        Returns:
            Tuple[NDArray, float, List[Dict[str, Any]], List[float], str]:
                - O melhor indivíduo (gene) encontrado.
                - A melhor pontuação de fitness (menor é melhor).
                - O histórico de resumos de cada geração.
                - O histórico de fitness do melhor indivíduo de cada geração.
                - O motivo pelo qual a otimização terminou.
        """
        population = np.array([self._rand() for _ in range(self.pop)])
        fitness = self._eval_pop(population)
        bar = tqdm(range(self.gens), desc="Otimizando com GA", unit="gen", leave=False)
        termination_reason = "Máximo de gerações atingido"

        for gen in bar:
            elite_indices = fitness.argsort()[: self.elite_n]
            offspring = list(population[elite_indices])

            while len(offspring) < self.pop:
                p1, p2 = self.rng.choice(population, 2, replace=False)
                c1, c2 = self._cx(p1, p2)
                offspring.extend([self._mut(c1), self._mut(c2)])

            population = np.array(offspring)[: self.pop]
            fitness = self._eval_pop(population)
            best_fitness_in_gen = fitness.min()

            best_individual_in_gen = population[fitness.argmin()]
            gen_summary = {
                "generation": gen + 1, "total_generations": self.gens,
                "best_r2_in_gen": -best_fitness_in_gen, "avg_r2_in_gen": -np.mean(fitness),
                "best_config_in_gen": self.space.decode(best_individual_in_gen),
            }
            self.generation_history.append(gen_summary)
            logging.info("Progresso da Geração", extra={"json_fields": gen_summary})

            bar.set_postfix(best_R2=f"{-best_fitness_in_gen:.4f}")
            self.history.append(best_fitness_in_gen)

            if (len(self.history) > self.patience and
                np.std(self.history[-self.patience:]) < 1e-4):
                logging.warning("Parada antecipada: performance estabilizou.")
                termination_reason = f"Parada antecipada na geração {gen + 1}"
                break

        best_idx = fitness.argmin()
        return population[best_idx], fitness[best_idx], self.generation_history, self.history, termination_reason