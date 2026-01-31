from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class MicroCluster:
    center: np.ndarray         # [D]
    n: int
    last_t: int
    sse: float                 # sum of squared errors (for radius estimate)

    @property
    def radius(self) -> float:
        # RMS radius
        if self.n <= 1:
            return 0.0
        return float(np.sqrt(self.sse / max(1, self.n - 1)))


class MiLOFEdgeDetector:
    """
    Streaming micro-cluster + LOF-like score approximation.

    - Maintain micro-clusters (centers) updated online.
    - Score x_t using kNN distances to centers -> local density -> LOF-like ratio.

    This is designed for fast, stable Edge gating (not a perfect MiLOF reproduction).
    """
    def __init__(
        self,
        k: int = 10,
        radius_factor: float = 2.5,
        max_clusters: int = 256,
        min_clusters_for_scoring: int = 20,
        decay_older_than: int = 24 * 14,   # 2 weeks in hours for pruning (by timestep index)
        eps: float = 1e-9,
    ):
        self.k = int(k)
        self.radius_factor = float(radius_factor)
        self.max_clusters = int(max_clusters)
        self.min_clusters_for_scoring = int(min_clusters_for_scoring)
        self.decay_older_than = int(decay_older_than)
        self.eps = float(eps)

        self.clusters: list[MicroCluster] = []
        self.t = 0

        # for optional standardization
        self.mu: np.ndarray | None = None
        self.sigma: np.ndarray | None = None

    def set_standardizer(self, mu: np.ndarray, sigma: np.ndarray) -> None:
        mu = np.asarray(mu, dtype=float)
        sigma = np.asarray(sigma, dtype=float)
        sigma = np.where(sigma <= 0, 1.0, sigma)
        self.mu, self.sigma = mu, sigma

    def _norm(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self.mu is None or self.sigma is None:
            return x
        return (x - self.mu) / self.sigma

    def _centers(self) -> np.ndarray:
        if not self.clusters:
            return np.empty((0, 0), dtype=float)
        return np.stack([c.center for c in self.clusters], axis=0)

    def _prune(self) -> None:
        if not self.clusters:
            return
        # remove very old clusters first
        now = self.t
        self.clusters = [c for c in self.clusters if (now - c.last_t) <= self.decay_older_than]

        # if still too many clusters: drop smallest n clusters
        if len(self.clusters) > self.max_clusters:
            self.clusters.sort(key=lambda c: c.n, reverse=True)
            self.clusters = self.clusters[: self.max_clusters]

    def partial_fit(self, x: np.ndarray) -> None:
        x = self._norm(x)
        self.t += 1

        if not self.clusters:
            self.clusters.append(MicroCluster(center=x.copy(), n=1, last_t=self.t, sse=0.0))
            return

        centers = self._centers()  # [M, D]
        d2 = np.sum((centers - x[None, :]) ** 2, axis=1)
        j = int(np.argmin(d2))
        nearest = self.clusters[j]
        dist = float(np.sqrt(d2[j]))

        # adaptive acceptance radius: cluster radius * factor + small slack
        thr = nearest.radius * self.radius_factor + 0.05

        if dist <= thr:
            # update micro-cluster (online mean update + SSE update)
            n_old = nearest.n
            c_old = nearest.center.copy()
            n_new = n_old + 1

            c_new = c_old + (x - c_old) / n_new
            # SSE update (Welford-like for vectors, approximate)
            # sse += ||x - c_new||^2 + n_old*||c_old - c_new||^2
            nearest.sse += float(np.sum((x - c_new) ** 2) + n_old * np.sum((c_old - c_new) ** 2))

            nearest.center = c_new
            nearest.n = n_new
            nearest.last_t = self.t
        else:
            # create new cluster
            self.clusters.append(MicroCluster(center=x.copy(), n=1, last_t=self.t, sse=0.0))

        self._prune()

    def score(self, x: np.ndarray) -> float:
        """
        LOF-like score:
          score = (avg_neighbor_lrd / lrd(x))
        where lrd is inverse of (avg reachability distance).
        Using centers as neighbor set.
        """
        x = self._norm(x)

        if len(self.clusters) < self.min_clusters_for_scoring:
            # not enough structure yet -> return neutral score
            return 1.0

        centers = self._centers()
        d = np.sqrt(np.sum((centers - x[None, :]) ** 2, axis=1)) + self.eps
        # kNN among centers
        k = min(self.k, len(d) - 1) if len(d) > 1 else 1
        nn_idx = np.argpartition(d, kth=k)[:k]
        nn_d = d[nn_idx]

        # reachability distance: max(d(x,nn), k-dist(nn))
        # approximate k-dist(nn) by distance from nn center to its k-th neighbor among centers
        # (fast approximation)
        kdists = []
        for i in nn_idx:
            di = np.sqrt(np.sum((centers - centers[i][None, :]) ** 2, axis=1)) + self.eps
            kk = min(k, len(di) - 1) if len(di) > 1 else 1
            kd = float(np.partition(di, kk)[kk])
            kdists.append(kd)
        kdists = np.asarray(kdists, dtype=float)

        reach = np.maximum(nn_d, kdists)
        lrd_x = 1.0 / (np.mean(reach) + self.eps)

        # neighbor lrd
        lrd_n = []
        for i in nn_idx:
            di = np.sqrt(np.sum((centers - centers[i][None, :]) ** 2, axis=1)) + self.eps
            kk = min(k, len(di) - 1) if len(di) > 1 else 1
            nn2 = np.argpartition(di, kth=kk)[:kk]
            nn2_d = di[nn2]

            kd2 = []
            for j in nn2:
                dj = np.sqrt(np.sum((centers - centers[j][None, :]) ** 2, axis=1)) + self.eps
                kd2.append(float(np.partition(dj, kk)[kk]))
            kd2 = np.asarray(kd2, dtype=float)

            reach2 = np.maximum(nn2_d, kd2)
            lrd = 1.0 / (np.mean(reach2) + self.eps)
            lrd_n.append(lrd)

        lrd_n = float(np.mean(lrd_n)) if lrd_n else lrd_x
        lof_like = (lrd_n / (lrd_x + self.eps))

        return float(lof_like)

    def predict(self, x: np.ndarray, tau: float) -> int:
        return int(self.score(x) >= float(tau))
