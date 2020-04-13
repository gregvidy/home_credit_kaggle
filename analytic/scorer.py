import math
import numpy as np

class Scorer(object):
    def __init__(self, min_score=300, max_score=850, pdo=50, base_odds=math.exp(3), base=600):
        self.A = pdo / math.log(2)
        self.B = base - math.log(base_odds) * self.A
        self.min_score = min_score
        self.max_score = max_score
        self.lower_proba = 1 / (math.exp((max_score - self.B) / self.A) + 1)
        self.upper_proba = 1 / (math.exp((min_score - self.B) / self.A) + 1)

    def _to_score(self, proba):
        if proba < self.lower_proba:
            return self.max_score
        elif proba > self.upper_proba:
            return self.min_score
        else:
            return round(self.A * math.log((1 - proba) / proba) + self.B)

    def to_score(self, proba):
        if np.isscalar(proba):
            return self._to_score(proba)
        return np.vectorize(self._to_score)(proba)

    def _to_proba(self, score):
        return 1 / (1 + math.exp((score - self.B) / self.A))

    def to_proba(self, score):
        if np.isscalar(score):
            return self._to_proba(score)
        return np.vectorize(self._to_proba)(score)
