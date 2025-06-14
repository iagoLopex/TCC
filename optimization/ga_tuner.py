# optimization/ga_tuner.py

import multiprocessing as mp
import numpy as np
from tqdm.auto import tqdm
import logging

class BaseGATuner:
    # ... (métodos __init__, _rand, _mut, etc. continuam os mesmos) ...
    """Algoritmo genético para otimização de hiperparâmetros."""
    def __init__(self, space, seed, n_jobs, **kwargs):
        self.__dict__.update(kwargs)
        self.space = space
        self.seed0 = seed
        self.n_jobs = n_jobs
        self.elite_n = max(1, int(self.elite * self.pop))
        self.rng = np.random.default_rng(seed)
        self.history = []

    def _rand(self):
        return np.array([self.rng.integers(lo, hi + 1) if t == 'int' else self.rng.uniform(lo, hi) 
                         for (lo, hi), t in zip(self.space.bounds, self.space.types)])

    def _mut(self, g):
        for i, t in enumerate(self.space.types):
            if self.rng.random() < self.mut:
                lo, hi = self.space.bounds[i]
                if t == 'int': g[i] = np.clip(g[i] + self.rng.integers(-1, 2), lo, hi)
                else: g[i] = np.clip(g[i] + self.rng.normal(0, 0.1 * (hi - lo)), lo, hi)
        return g

    def _cx(self, p1, p2):
        if self.rng.random() > self.cx: return p1.copy(), p2.copy()
        mask = self.rng.random(len(p1)) < 0.5
        c1, c2 = p1.copy(), p2.copy()
        c1[mask], c2[mask] = p2[mask], p1[mask]
        return c1, c2

    def _score(self, gene): 
        return np.mean([self.space.evaluate(self.space.decode(gene), i) for i in range(self.repeats)])

    def _eval_pop(self, pop):
        if self.n_jobs == 1: return np.array([self._score(g) for g in pop])
        with mp.Pool(self.n_jobs) as pool: return np.array(pool.map(self._score, pop))


    def run(self):
        pop = np.array([self._rand() for _ in range(self.pop)])
        fit = self._eval_pop(pop)
        
        # --- INÍCIO DA MODIFICAÇÃO ---
        generation_history = [] # Lista para guardar o resumo de cada geração
        bar = tqdm(range(self.gens), desc="Otimizando com GA", unit="gen")
        # --- FIM DA MODIFICAÇÃO ---

        for gen in bar:
            elite = pop[fit.argsort()[ : self.elite_n]]
            offspring = list(elite)

            while len(offspring) < self.pop:
                p1, p2 = self.rng.choice(pop, 2, replace=False)
                c1, c2 = self._cx(p1,p2)
                offspring.extend([self._mut(c1), self._mut(c2)])

            pop = np.array(offspring)[ : self.pop]
            fit = self._eval_pop(pop)
            best_fit = fit.min()

            # --- INÍCIO DA MODIFICAÇÃO: LOGGING APENAS PARA ARQUIVO E COLETA DE HISTÓRICO ---
            best_gen_idx = fit.argmin()
            best_in_gen = pop[best_gen_idx]
            
            # Prepara os dados para o log JSON e para o histórico do console
            gen_summary = {
                'generation': gen + 1,
                'total_generations': self.gens,
                'best_r2_in_gen': -best_fit,
                'avg_r2_in_gen': -np.mean(fit),
                'best_config_in_gen': self.space.decode(best_in_gen)
            }
            # Adiciona o resumo da geração à lista de histórico
            generation_history.append(gen_summary)
            # Loga os dados completos APENAS no arquivo JSON. Não aparecerá no console.
            logging.info("Progresso da Geração", extra={'json_fields': gen_summary})
            # --- FIM DA MODIFICAÇÃO ---

            bar.set_postfix(best_R2 = f"{-best_fit:.4f}")
            self.history.append(best_fit)

            if len(self.history) > self.patience and np.std(self.history[-self.patience:]) < 1e-4:
                logging.warning("Parada antecipada: performance estabilizou.")
                break

        # --- MODIFICAÇÃO: Retorna o histórico junto com os outros resultados ---
        return pop[fit.argmin()], fit.min(), generation_history