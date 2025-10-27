import numpy as np
import random
from sklearn.base import clone
from copy import deepcopy
from src.utils import evaluate_model


class GAFeatureSelector:
    def __init__(self,
                 estimator,
                 n_gen=40,
                 pop_size=30,
                 cx_prob=0.8,
                 mut_prob=0.02,
                 tournament_size=3,
                 random_state=42,
                 scoring='f1'):
        self.estimator = estimator
        self.n_gen = n_gen
        self.pop_size = pop_size
        self.cx_prob = cx_prob
        self.mut_prob = mut_prob
        self.tournament_size = tournament_size
        self.random_state = random_state
        self.scoring = scoring
        random.seed(random_state)
        np.random.seed(random_state)

    def _init_population(self, n_features):
        pop = []
        for _ in range(self.pop_size):
            mask = np.random.choice([0, 1], size=n_features, p=[0.5, 0.5])

            # Ensure the number of features is less than 50
            while mask.sum() > 50:
                ones = np.where(mask == 1)[0]
                mask[random.choice(ones)] = 0

            if mask.sum() == 0:
                mask[np.random.randint(0, n_features)] = 1

            pop.append(mask)
        return pop

    def _fitness(self, mask, X, y, cv=3):

        idx = np.where(mask == 1)[0]
        if len(idx) == 0:
            return 0.0


        # Takes only the first 300 rows to speed up the analysis process
        sample_size = min(300, X.shape[0])
        sample_idx = np.random.choice(X.shape[0], sample_size, replace=False)
        X_sample = X[sample_idx][:, idx]
        y_sample = y[sample_idx]

        model = clone(self.estimator)
        try:
            model.fit(X_sample, y_sample)
            score = model.score(X_sample, y_sample)
        except Exception:
            score = 0.0

        return score

    def _tournament(self, pop, fitnesses):
        best = None
        for _ in range(self.tournament_size):
            i = random.randrange(len(pop))
            if best is None or fitnesses[i] > fitnesses[best]:
                best = i
        return deepcopy(pop[best])

    def _crossover(self, parent1, parent2):
        if random.random() > self.cx_prob:
            return deepcopy(parent1), deepcopy(parent2)
        n = len(parent1)
        pt = random.randint(1, n - 1)
        child1 = np.concatenate([parent1[:pt], parent2[pt:]])
        child2 = np.concatenate([parent2[:pt], parent1[pt:]])
        return child1, child2

    def _mutate(self, individual):
        for i in range(len(individual)):
            if random.random() < self.mut_prob:
                individual[i] = 1 - individual[i]

        while individual.sum() > 50:
            ones = np.where(individual == 1)[0]
            individual[random.choice(ones)] = 0

        if individual.sum() == 0:
            individual[random.randrange(len(individual))] = 1

        return individual

    def fit(self, X, y, cv=3, verbose=False):
        n_features = X.shape[1]
        population = self._init_population(n_features)
        fitnesses = [self._fitness(ind, X, y, cv=cv) for ind in population]

        best_idx = int(np.argmax(fitnesses))
        best = deepcopy(population[best_idx])
        best_score = fitnesses[best_idx]
        history = []

        for gen in range(self.n_gen):
            new_pop = [deepcopy(best)]
            while len(new_pop) < self.pop_size:
                p1 = self._tournament(population, fitnesses)
                p2 = self._tournament(population, fitnesses)
                c1, c2 = self._crossover(p1, p2)
                c1 = self._mutate(c1)
                c2 = self._mutate(c2)
                new_pop.extend([c1, c2])
            new_pop = new_pop[:self.pop_size]
            population = new_pop
            fitnesses = [self._fitness(ind, X, y, cv=cv) for ind in population]
            gen_best_idx = int(np.argmax(fitnesses))
            gen_best_score = fitnesses[gen_best_idx]
            if gen_best_score > best_score:
                best_score = gen_best_score
                best = deepcopy(population[gen_best_idx])
            history.append(best_score)
            if verbose:
                print(f"Gen {gen+1}/{self.n_gen} Best {best_score:.4f}")

        self.best_mask_ = best
        self.best_score_ = best_score
        self.history_ = history
        return self

    def transform(self, X):
        idx = np.where(self.best_mask_ == 1)[0]
    
        # It only takes the top 50 features and ranks them from best to best.
        if len(idx) > 50:
            sorted_idx = np.argsort(self.best_mask_[idx])[::-1]
            idx = idx[sorted_idx][:50]  
    
        if hasattr(X, "iloc"):
            return X.iloc[:, idx]
        else:
            return X[:, idx]

    def fit_transform(self, X, y, cv=3, verbose=False):
        self.fit(X, y, cv=cv, verbose=verbose)
        return self.transform(X)

