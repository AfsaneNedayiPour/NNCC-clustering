import time
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KDTree
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris, load_wine, load_digits
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score,
    adjusted_mutual_info_score,
    fowlkes_mallows_score
)
from matplotlib.patches import Circle
from pandas import read_csv
from scipy.io import arff
import pandas as pd


class MyClustering:
    def __init__(self, z=0.1, depth_cap=1, beta=0.5, min_overlap_points=0, alpha=0, eps_scale=1e-4):
        """
    ی

        Parameters:
        -----------
        z : float
             (threshold = mean(gamma) + z * std(gamma))
        depth_cap : int
        beta : float
        min_overlap_points : int
        alpha : float
        eps_scale : float

        """
        self.z = z
        self.depth_cap = depth_cap
        self.beta = beta
        self.min_overlap_points = min_overlap_points
        self.alpha = alpha
        self.eps_scale = eps_scale
        self.X = None
        self.y_true = None
        self.labels_my = None
        self.merged_labels = None
        self.n_clusters = None
        self.report = None
        self.runtime = None
        self.stage_times = None

    def _natural_neighbor_search(self, X, max_iter=50):
        """Natural Neighbor Search"""
        X = np.array(X)
        n = len(X)
        tree = KDTree(X)
        dist_all, idx_all = tree.query(X, k=min(max_iter +1 , n))
        r = 1
        Nb = np.zeros(n, dtype=int)
        NaN_neighbors = [set() for _ in range(n)]
        prev_num_zero = -1

        while True:
            for i in range(n):

                pj = idx_all[i,r]
                Nb[pj] += 1
                NaN_neighbors[pj].add(i)

            num_zero = np.sum(Nb == 0)
            if num_zero == prev_num_zero or r >= max_iter:
                break

            prev_num_zero = num_zero
            r += 1

        lam = r - 1
        return lam, NaN_neighbors, Nb, idx_all, dist_all


    def _compute_radii_and_select_heads(self, X, idx_all, dist_all):
        n = len(X)
        lam, NaN_neighbors, Nb, idx_all, dist_all = self._natural_neighbor_search(X)
        R = dist_all[:, 1:3].mean(axis=1)
        radius_candidates = [[] for _ in range(n)]

        for i in range(n):
            if len(NaN_neighbors[i]) < 2:
                continue
            neighbor_set = NaN_neighbors[i]
            closest_two = []
            for j in idx_all[i, 1:]:
                if j in neighbor_set:
                    closest_two.append(j)
                    if len(closest_two) == 2:
                        break
            if len(closest_two) <2:
                continue
            n1, n2 = closest_two

            R_new = (R[i] + R[n1] + R[n2]) / 3.0

            for idx in [i, n1, n2]:
                radius_candidates[idx].append(R_new)

        for i in range(n):
            if radius_candidates[i]:
                R[i] = min(radius_candidates[i])

        sorted_idx = np.argsort(R)
        D_prime = []
        prev_neighbors = set()

        for i in sorted_idx:
            if len(NaN_neighbors[i]) == 0:
                continue
            if NaN_neighbors[i].isdisjoint(prev_neighbors):
                D_prime.append(i)
                prev_neighbors.update(NaN_neighbors[i])

        D_prime = sorted(D_prime, key=lambda idx: R[idx])
        return D_prime, R, lam, NaN_neighbors, Nb



    def _compute_density_gamma(self, X, D_prime, R, lam, NaN_neighbors):

        D_prime = np.asarray(D_prime, dtype=int)

        if D_prime.size == 0:
            return np.array([], dtype=int), R, lam, NaN_neighbors, {}, np.array([]), np.array([]), np.array([])

        local_density = {}
        for idx in D_prime:
            local_density[idx] = 1 / (R[idx] + 1e-6)
        rho = np.array([local_density[idx] for idx in D_prime], dtype=float)

        delta = np.full(D_prime.size, np.inf, dtype=float)
        for i, idx in enumerate(D_prime):
            higher = [j for j in D_prime if local_density[j] > local_density[idx]]
            if len(higher) > 0:
                delta[i] = float(np.min(np.linalg.norm(X[idx] - X[higher], axis=1)))
            else:
                delta[i] = float(np.max(np.linalg.norm(X - X[idx], axis=1)))


        gamma = rho * delta


        mu = float(np.mean(gamma))
        sig = float(np.std(gamma))
        threshold = mu + self.z * sig

        final_heads = D_prime[np.where(gamma >= threshold)[0]]


        min_heads = 2
        if final_heads.size < min_heads and D_prime.size >= min_heads:
            top_idx = np.argsort(-gamma)[:min_heads]
            final_heads = D_prime[top_idx]

        return final_heads, R, lam, NaN_neighbors, local_density, delta, gamma

    def _expand_from_final_heads(self, final_heads, NaN_neighbors, X):

        heads = np.asarray(final_heads, dtype=int).ravel()
        n = len(NaN_neighbors)
        if heads.size == 0:
            return {}

        heads = [h for h in heads if 0 <= h < n]
        K = len(heads)
        if K == 0:
            return {}

        assigned = np.full(n, -1, dtype=int)
        groups = {h: set() for h in heads}
        visited = {h: set() for h in heads}
        frontier = {h: set() for h in heads}
        active = {h: True for h in heads}


        for h in heads:
            groups[h].add(h)
            visited[h].add(h)
            assigned[h] = h
            frontier[h] = {h}

        for depth in range(1, self.depth_cap + 1):
            proposals = {}

            for h in heads:
                if not active[h]:
                    continue

                cand = set()
                for node in frontier[h]:
                    cand.update(NaN_neighbors[node])

                cand -= visited[h]
                cand = {u for u in cand if assigned[u] in (-1, h)}

                for u in cand:
                    proposals.setdefault(u, []).append(h)

            if not proposals:
                break

            gained_this_round = {h: set() for h in heads}

            for u, hs in proposals.items():
                if len(hs) == 1:
                    h = hs[0]
                    if assigned[u] == -1:
                        gained_this_round[h].add(u)


            any_change = False
            for h in heads:
                if not active[h]:
                    continue

                new_nodes = gained_this_round[h]
                if new_nodes:
                    any_change = True
                    for u in new_nodes:
                        assigned[u] = h
                    groups[h].update(new_nodes)
                    visited[h].update(new_nodes)
                    frontier[h] = new_nodes
                else:
                    frontier[h] = set()

                if not new_nodes:
                    active[h] = False

            if not any_change:
                break

            if not any(active.values()):
                break

        return groups

    def _compute_circles_from_labels(self, X, labels):

        X = np.asarray(X)
        labels = np.asarray(labels)

        mask = labels >= 0
        uniq = np.unique(labels[mask])
        if uniq.size == 0:
            return np.empty((0, X.shape[1])), np.empty((0,)), np.array([], dtype=int)

        centers = np.zeros((uniq.size, X.shape[1]), dtype=float)
        radii = np.zeros(uniq.size, dtype=float)

        for i, lab in enumerate(uniq):
            Xc = X[labels == lab]
            if Xc.shape[0] == 0:
                continue

            mu = np.median(Xc, axis=0)
            d = np.linalg.norm(Xc - mu, axis=1)
            if d.size == 0:
                r = 0.0
            else:
                # محاسبه تطبیقی q بر اساس اندازه خوشه
                n_c = d.size
                q = 1.0 - 1.0 / np.sqrt(n_c) if n_c > 1 else 1.0
                q = float(np.clip(q, 0.80, 0.995))
                r = float(d.max()) if q >= 1.0 else float(np.quantile(d, q))

            centers[i] = mu
            radii[i] = r

        return centers, radii, uniq

    def _merge_by_circles(self, X, base_labels, centers, radii, label_order):

        X = np.asarray(X)
        base_labels = np.asarray(base_labels)
        centers = np.asarray(centers)
        radii = np.asarray(radii)
        label_order = np.asarray(label_order)

        K = len(label_order)
        if K == 0:
            return base_labels.copy(), 0

        dist_to_centers = cdist(X, centers)
        lab_to_idx = {lab: i for i, lab in enumerate(label_order)}

        med_r = float(np.median(radii)) if radii.size else 1.0
        eps = float(self.eps_scale * max(1.0, med_r))

        edges = []

        for i in range(K):
            li = label_order[i]
            mask_i = (base_labels == li)
            n_i = int(np.sum(mask_i))
            if n_i == 0:
                continue

            for j in range(i + 1, K):
                lj = label_order[j]
                mask_j = (base_labels == lj)
                n_j = int(np.sum(mask_j))
                if n_j == 0:
                    continue


                d_cent = float(np.linalg.norm(centers[i] - centers[j]))
                if d_cent > (self.beta * float(radii[i] + radii[j])):
                    continue


                thr = max(int(self.min_overlap_points), int(np.ceil(self.alpha * min(n_i, n_j))))

                mask_scope = mask_i | mask_j
                di = dist_to_centers[:, i]
                dj = dist_to_centers[:, j]
                ri = float(radii[i])
                rj = float(radii[j])

                in_both = (di <= ri + eps) & (dj <= rj + eps)
                overlap_mask = mask_scope & in_both
                overlap_count = int(np.sum(overlap_mask))

                if overlap_count >= thr:
                    edges.append((i, j))


        parent = list(range(K))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for i, j in edges:
            union(i, j)

        roots = [find(i) for i in range(K)]
        unique_roots = {r: idx for idx, r in enumerate(sorted(set(roots)))}
        comp = np.array([unique_roots[r] for r in roots], dtype=int)

        n_final = int(comp.max() + 1) if K > 0 else 0

        # ساخت merged_labels
        merged_labels = np.full_like(base_labels, -1)
        mask_valid = (base_labels != -1)
        for idx in np.where(mask_valid)[0]:
            k = lab_to_idx[base_labels[idx]]
            merged_labels[idx] = int(comp[k])

        return merged_labels, n_final

    def plot_robust_circles(self,X, labels, centers, radii, label_order,
                            show_points=True, show_noise=True):
        """
        X: (n,2)
        labels: (n,)  ( labels_my)
        centers, radii, label_order: ی compute_circles_from_labels_robust
        """
        X = np.asarray(X)
        labels = np.asarray(labels)

        X2 = X
        if X.shape[1] > 2:
            from sklearn.decomposition import PCA
            X2 = PCA(n_components=2, random_state=0).fit_transform(X)
        plt.figure(dpi=1000)
        fig, ax = plt.subplots(figsize=(5, 4))


        if show_points:
            mask_cluster = (labels != -1)
            ax.scatter(X[mask_cluster, 0], X[mask_cluster, 1],
                       c=labels[mask_cluster], cmap="tab10", s=18)

            if show_noise and np.any(labels == -1):
                ax.scatter(X[labels == -1, 0], X[labels == -1, 1],
                           c="lightgray", s=14)


        for mu, r, lab in zip(centers, radii, label_order):
            cx, cy = float(mu[0]), float(mu[1])
            ax.scatter([cx], [cy], c="black", s=120, marker="x")
            if r > 0:
                circ = Circle((cx, cy), float(r), fill=False, linestyle="--", linewidth=1.5, alpha=0.7)
                ax.add_patch(circ)


        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.axis("equal")
        plt.savefig('aggcluster_circle.tif', dpi=1000)
        plt.show()




    def fit(self, X, y_true=None):

        self.X = np.array(X)
        self.y_true = np.array(y_true) if y_true is not None else None
        self.stage_times = {}
        t_total = time.perf_counter()



        t = time.perf_counter()
        lam, NaN_neighbors, Nb,idx_all, dist_all  = self._natural_neighbor_search(X, max_iter=50)
        self.stage_times["S0: NaN "] = time.perf_counter()
        t = time.perf_counter()
        D_prime, R, lam, NaN_neighbors, Nb = self._compute_radii_and_select_heads(self.X,idx_all, dist_all)
        self.stage_times["S1: radius and initial heads "] = time.perf_counter()

        t = time.perf_counter()
        final_heads, R, lam, NaN_neighbors, local_density, delta, gamma = self._compute_density_gamma(
            self.X, D_prime, R, lam, NaN_neighbors
        )
        self.stage_times["S2: density and gamma "] = time.perf_counter() - t

        t = time.perf_counter()
        expanded_groups = self._expand_from_final_heads(final_heads, NaN_neighbors, self.X)
        self.stage_times["S3: cluster expansion "] = time.perf_counter() - t

        n = self.X.shape[0]
        self.labels_my = np.full(n, -1, dtype=int)
        for cid, (head, members) in enumerate(expanded_groups.items()):
            for idx in members:
                self.labels_my[idx] = cid


        t = time.perf_counter()
        centers_base, radii_base, label_order = self._compute_circles_from_labels(self.X, self.labels_my)
        self.stage_times["S4: circle construction "] = time.perf_counter() - t


        t = time.perf_counter()
        self.merged_labels, self.n_clusters = self._merge_by_circles(
            self.X,
            self.labels_my,
            centers_base,
            radii_base,
            label_order
        )
        self.stage_times["S5: circle merging "] = time.perf_counter() - t
        self.runtime = time.perf_counter() - t_total
        print("runtime of each stage")

        for name, t in self.stage_times.items():
            print(f"{name:35s}: {t:6f} s")
        print(f"{'Total Runtime' :35s}: {self.runtime:.6f} s")
        print(f"\nLabels shape: {mycode1.get_labels().shape}")
        print(f"Number of clusters: {mycode1.get_n_clusters()}")

        # advanced propagation
        self.merged_labels = self._advanced_label_propagation(
            X=self.X,
            labels=self.merged_labels,
            NaN_neighbors=NaN_neighbors,

            threshold=0.55,
            confidence_threshold=0.75,
            entropy_threshold=0.7,

            max_iter=5,
            min_labeled_neighbors=2,

            distance_weighting=True
        )


        # ی after merge
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(self.X[:, 0], self.X[:, 1], c=self.merged_labels, cmap='tab10', s=30)
        plt.title('NNCC')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()
        self.plot_robust_circles(self.X, self.labels_my, centers_base, radii_base, label_order)

        if self.y_true is not None:
            mask_valid = (self.merged_labels != -1)
            from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
            acc = clustering_accuracy(self.y_true[mask_valid], self.merged_labels[mask_valid])
            nmi = normalized_mutual_info_score(self.y_true[mask_valid], self.merged_labels[mask_valid])
            ari = adjusted_rand_score(self.y_true[mask_valid], self.merged_labels[mask_valid])
            ami = adjusted_mutual_info_score(self.y_true[mask_valid], self.merged_labels[mask_valid])
            fmi = fowlkes_mallows_score(self.y_true[mask_valid], self.merged_labels[mask_valid])

            print(f"\n=== Evaluation Results ===")
            print(f"Runtime: {self.runtime:.4f} seconds")
            print(f"ACC: {acc:.4f}")
            print(f"NMI: {nmi:.4f}")
            print(f"ARI: {ari:.4f}")
            print(f"AMI: {ami:.4f}")
            print(f"FMI: {fmi:.4f}")
            print(f"Number of clusters: {self.n_clusters}")

        return self

    def get_labels(self):
        """ی after merge"""
        return self.merged_labels

    def get_labels_before_merge(self):
        return self.labels_my

    def get_runtime(self):
        return self.runtime

    def get_n_clusters(self):
        return self.n_clusters

    def _advanced_label_propagation(
            self,
            X,
            labels,
            NaN_neighbors,
            threshold=0.7,
            confidence_threshold=0.6,
            entropy_threshold=1.0,
            max_iter=20,
            min_labeled_neighbors=2,
            distance_weighting=True
    ):
        """
        Advanced iterative label propagation
        using natural neighbors.

        Parameters
        ----------
        X : ndarray
            data

        labels : ndarray
            cluster labels (-1 means unlabeled)

        NaN_neighbors : list of sets
            natural neighbors

        threshold : float
            minimum labeled-neighbor ratio

        confidence_threshold : float
            minimum dominance confidence

        entropy_threshold : float
            maximum entropy allowed

        max_iter : int
            maximum iterations

        min_labeled_neighbors : int
            minimum labeled neighbors

        distance_weighting : bool
            weighted voting by inverse distance
        """

        X = np.asarray(X)
        labels_new = labels.copy()

        n = len(labels_new)

        for iteration in range(max_iter):

            updates = {}
            changes = 0

            unlabeled_indices = np.where(labels_new == -1)[0]

            if len(unlabeled_indices) == 0:
                break

            for i in unlabeled_indices:

                neighbors = list(NaN_neighbors[i])

                if len(neighbors) == 0:
                    continue

                neighbor_labels = labels_new[neighbors]

                labeled_mask = (neighbor_labels != -1)

                labeled_neighbors = np.array(neighbors)[labeled_mask]
                labeled_labels = neighbor_labels[labeled_mask]


                if len(labeled_neighbors) < min_labeled_neighbors:
                    continue


                labeled_ratio = len(labeled_neighbors) / len(neighbors)

                if labeled_ratio < threshold:
                    continue

                # ---------- weighted voting ----------
                vote_scores = {}

                for neigh_idx, lab in zip(labeled_neighbors, labeled_labels):

                    if distance_weighting:

                        dist = np.linalg.norm(X[i] - X[neigh_idx])

                        weight = 1.0 / (dist + 1e-8)

                    else:
                        weight = 1.0

                    vote_scores[lab] = vote_scores.get(lab, 0.0) + weight

                if len(vote_scores) == 0:
                    continue

                # ---------- confidence ----------
                labels_list = np.array(list(vote_scores.keys()))
                scores = np.array(list(vote_scores.values()))

                total_score = np.sum(scores)

                probs = scores / (total_score + 1e-12)

                best_idx = np.argmax(probs)

                best_label = labels_list[best_idx]

                confidence = probs[best_idx]

                # ---------- entropy ----------
                entropy = -np.sum(probs * np.log(probs + 1e-12))

                # ---------- decision ----------
                if confidence >= confidence_threshold and entropy <= entropy_threshold:
                    updates[i] = best_label


            for idx, lab in updates.items():
                labels_new[idx] = lab
                changes += 1


            if changes == 0:
                break

        return labels_new


def clustering_accuracy(y_true, y_pred):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    classes = np.unique(y_true)
    clusters = np.unique(y_pred)

    cost_matrix = np.zeros((clusters.size, classes.size), dtype=np.int64)
    for i, c in enumerate(clusters):
        for j, k in enumerate(classes):
            cost_matrix[i, j] = np.sum((y_pred == c) & (y_true == k))

    row_ind, col_ind = linear_sum_assignment(cost_matrix.max() - cost_matrix)
    correct = cost_matrix[row_ind, col_ind].sum()
    acc = correct / y_true.size
    return acc


def evaluate_clustering(y_true, y_pred):
    """
     ACC, NMI, ARI, AMI, FMI
    """
    acc = clustering_accuracy(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    ami = adjusted_mutual_info_score(y_true, y_pred)
    fmi = fowlkes_mallows_score(y_true, y_pred)

    return {
        "ACC": acc,
        "NMI": nmi,
        "ARI": ari,
        "AMI": ami,
        "FMI": fmi
    }


def load_parkin_data():
    data = read_csv('parkinsons.csv')

    y_true = data["status"]

    X1 = data.drop("status", 1)
    X = X1.drop("name", 1)
    X = StandardScaler().fit_transform(X)
    return X, y_true


def load_iris_data():
    data = load_iris()
    X = data.data  # فقط ویژگی‌ها، برچسب‌ها لازم نیست
    y_true = data.target
    X = StandardScaler().fit_transform(X)  # نرمال‌سازی اختیاری اما بهتر
    return X, y_true


from sklearn.datasets import load_wine


def load_wine_data():
    data = load_wine()
    X = data.data
    y_true = data.target
    X = StandardScaler().fit_transform(X)
    return X, y_true


from sklearn.datasets import load_digits


def load_optdigits_data():
    data = load_digits()
    X = data.data
    y_true = data.target
    X = StandardScaler().fit_transform(X)
    return X, y_true


def load_seeds_data():
    data = read_csv('seeds.csv')
    X = data.drop("V8", 1)
    y_true = data["V8"]
    X = StandardScaler().fit_transform(X)
    return X, y_true


def load_glass_data():
    data = read_csv('glass_csv.csv')
    X = data.drop("Type", 1)
    y_true = data["Type"]
    X = StandardScaler().fit_transform(X)
    return X, y_true


def load_dermatology_data():
    data = read_csv('dermatology_csv.csv')
    X1 = data.drop("class", 1)
    X = X1.drop("age", 1)
    y_true = data["class"]
    X = StandardScaler().fit_transform(X)
    return X, y_true


def load_segment_data():
    data = read_csv('segment_csv.csv')
    X = data.drop("class", 1)
    y_true = data["class"]
    X = StandardScaler().fit_transform(X)
    return X, y_true


def load_yeast_data():
    data = read_csv('yeast_csv.csv')
    X = data.drop("class_protein_localization", 1)
    y_true = data["class_protein_localization"]
    X = StandardScaler().fit_transform(X)
    return X, y_true


def load_vehicle_data():
    data = read_csv('Vehicle.csv')
    X = data.drop("Class", 1)
    y_true = data["Class"]
    X = StandardScaler().fit_transform(X)
    return X, y_true


def load_face_data():
    oliv = datasets.fetch_olivetti_faces()
    X = oliv.data
    y = oliv.target
    X = PCA(n_components=50, whiten=True, random_state=0).fit_transform(X)
    y_true = y
    return X, y_true



def minmax01(X):
    X = X.astype(float)
    mn = X.min(axis=0, keepdims=True)
    mx = X.max(axis=0, keepdims=True)
    return (X - mn) / (mx - mn + 1e-12)


def load_wdbc_data():
    data = read_csv('wdbc_csv.csv')
    X = data.drop("Class", 1)
    y_true = data["Class"]
    X = StandardScaler().fit_transform(X)
    return X, y_true


def load_heart_data():
    data = read_csv('heart-statlog_csv.csv')
    X = data.drop("class", 1)
    y_true = data["class"]
    X = StandardScaler().fit_transform(X)
    le = LabelEncoder()
    y_true = le.fit_transform(y_true)
    return X, y_true


if __name__ == "__main__":


    """
    data = np.loadtxt('waveform.data', delimiter=',')
    X = data[:, :-1]
    y_true = data[:, -1]
    """

    with open('aggregation.txt') as f:
        points = [tuple(map(float, i.split('\t')[0:2])) for i in f]
        X = np.array(points)
        y_true = []
    with open('aggregation.txt') as f:
        for line in f:
            parts = line.strip().split('\t')
            x1, x2, lab = parts[:3]
            y_true.append(int(lab))
    y_true = np.array(y_true)
    mycode1 = MyClustering(z=0.2, depth_cap=4
                           , beta=1.2)
    mycode1.fit(X, y_true)
    merged_labels = mycode1.get_labels()

    lam, NaN_neighbors, Nb, idx_all, dist_all = mycode1._natural_neighbor_search(X, max_iter=50)
    D_prime, R, lam, NaN_neighbors, Nb = mycode1._compute_radii_and_select_heads(X,idx_all, dist_all)
    final_heads, R, lam, NaN_neighbors, local_density, delta, gamma = mycode1._compute_density_gamma(X, D_prime, R, lam, NaN_neighbors)
    print("Dprime", D_prime)
    print("f", final_heads)


    plt.figure(dpi=1000)
    fig, ax = plt.subplots(figsize=(5, 4))
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "lines.linewidth": 1.2,
        "lines.markersize": 4
    })
    ax.scatter(X[:, 0], X[:, 1], c='blue', s=0.5)

    for idx in range(len(X)):
        circle = Circle(
            (X[idx, 0], X[idx, 1]),
            R[idx],
            fill=False,
            edgecolor='orange',
            linewidth=1
        )
        ax.add_patch(circle)

    plt.axis('equal')
    plt.savefig('aggcircle.tif', dpi=1000)
    plt.show()

    plt.figure(dpi=1000)
    plt.figure(figsize=(3.42, 2.10))
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "lines.linewidth": 1.2,
        "lines.markersize": 4
    })
    scatter = plt.scatter(X[:, 0], X[:, 1], c=merged_labels, cmap='tab10', s=30)
    plt.savefig('flameNNCClast4.tif', dpi=1000)
    plt.show()



