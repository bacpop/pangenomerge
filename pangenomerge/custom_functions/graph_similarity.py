"""Pair-level clustering-similarity metrics for scoring a merge against a truth graph.

These back `--mode test`, and are also imported directly by the thesis benchmarking
scripts. They live in the package so that the tool and the write-up cannot drift into
reporting different things under the same metric name.

Three partitions of the same set of genes are involved throughout:

    C  the component clustering -- the union of the input panaroo graphs, each cluster made
       globally unique. This is pangenomerge's "do nothing" starting point.
    M  the merged clustering produced by pangenomerge.
    T  the truth clustering -- panaroo run over all the isolates at once.

pangenomerge only ever *coarsens* C: it groups whole component clusters together and never
splits one. Two consequences drive everything here.

1. A per-gene random labelling is the wrong null. Leaving C untouched already scores
   ARI 0.66 / AMI 0.93 against T, so standard scores compress every real difference into
   the fourth decimal. Both metric families below therefore put 0 at C, not at random.

2. Because merging only ever *adds* co-clustered pairs, any pair of genes that C groups
   but T separates is wrong forever. If C does not refine T -- and in practice it does not
   -- then a score of 1 against T is unreachable, and a metric anchored at T reports a
   number against an impossible target.

Hence two families:

    pcARI / pcNMI   post-clustering:  0 = C, 1 = T
    acARI / acNMI   attainability-corrected:  0 = C, 1 = C*, the best merge reachable from C

pcARI has a clean identity. Writing b for pairs wrongly co-clustered and c for pairs wrongly
separated, RI = 1 - (b+c)/N, so N cancels:

    pcARI = (RI(M,T) - RI(C,T)) / (1 - RI(C,T)) = 1 - (b_M + c_M) / (b_C + c_C)

i.e. the fraction of the component clustering's pairwise errors that the merge removed.

The MI side uses *normalized* MI, not raw MI, deliberately. Raw MI is monotonic under
coarsening, so MI(M,T) <= MI(C,T) always: it can only ever penalise a correct merge, and
ranks "do nothing" above every method. NMI's denominator shrinks as M coarsens, so a good
merge can raise it.
"""

import logging
from collections import defaultdict
from math import nan

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def _labels(mapping, keys):
    return [mapping[k] for k in keys]


def encode(labels):
    """Cluster labels -> contiguous int32 codes.

    Every metric here needs labels factorised, and np.unique over an object array of ~1M
    Python strings is the single most expensive operation in the whole scorer. Doing it once
    and passing integers to everything downstream (sklearn included) avoids repeating it per
    metric per scenario.
    """
    arr = np.asarray(labels, dtype=object)
    codes, idx = np.unique(arr, return_inverse=True)
    return idx.astype(np.int32), codes


def contingency(labels_a, labels_b):
    """Sparse contingency table between two labellings of the same ordered gene list.

    Accepts either raw labels or the int-code arrays produced by encode().
    """
    a_idx, a_codes = encode(labels_a) if not _is_codes(labels_a) else (labels_a, None)
    b_idx, b_codes = encode(labels_b) if not _is_codes(labels_b) else (labels_b, None)
    data = np.ones(len(a_idx), dtype=np.int64)
    n_a, n_b = int(a_idx.max()) + 1, int(b_idx.max()) + 1
    table = coo_matrix((data, (a_idx, b_idx)), shape=(n_a, n_b)).tocsr()
    # When the input was already codes, the code *is* the row/column index, so callers that
    # map labels back through these arrays (best_reachable_merge) get an identity mapping.
    if a_codes is None:
        a_codes = np.arange(n_a)
    if b_codes is None:
        b_codes = np.arange(n_b)
    return table, a_codes, b_codes


def _is_codes(x):
    return isinstance(x, np.ndarray) and x.dtype.kind in "iu"


def _n_choose_2(x):
    x = np.asarray(x, dtype=np.float64)
    return x * (x - 1.0) / 2.0


def _pair_counts_from_table(table, n):
    """Pair counts from an existing contingency table (avoids rebuilding it)."""
    counts = np.asarray(table.data, dtype=np.float64)
    a = float(_n_choose_2(counts).sum())
    pairs_x = float(_n_choose_2(np.asarray(table.sum(axis=1)).ravel()).sum())
    pairs_t = float(_n_choose_2(np.asarray(table.sum(axis=0)).ravel()).sum())
    n = float(n)
    return {"a": a, "b": pairs_x - a, "c": pairs_t - a,
            "n_pairs": n * (n - 1.0) / 2.0}


def pair_counts(labels_x, labels_t):
    """Pair-level agreement between a clustering X and the truth T.

        a  pairs co-clustered in both X and T
        b  pairs co-clustered in X only  (wrongly merged)
        c  pairs co-clustered in T only  (wrongly split)
    """
    table, _, _ = contingency(labels_x, labels_t)
    return _pair_counts_from_table(table, len(labels_x))


def _set_partitions(k):
    """All set partitions of range(k), as restricted growth strings."""
    if k == 0:
        return
    a = [0] * k
    while True:
        yield list(a)
        i = k - 1
        while i > 0:
            ceiling = max(a[:i]) + 1
            if a[i] <= ceiling - 1:
                a[i] += 1
                for j in range(i + 1, k):
                    a[j] = 0
                break
            i -= 1
        else:
            return


def _grouping_cost(groups, sizes, s):
    """Pairwise error of a grouping of C-blocks within one connected component.

    Merging blocks i and j costs |Bi||Bj| - s_ij wrongly-merged pairs; keeping them apart
    costs s_ij wrongly-split pairs. Pairs inside a single C-block are fixed either way and
    are excluded, so this is comparable across groupings but is not an absolute error.
    """
    k = len(sizes)
    cost = 0.0
    for i in range(k):
        for j in range(i + 1, k):
            if groups[i] == groups[j]:
                cost += sizes[i] * sizes[j] - s[i, j]
            else:
                cost += s[i, j]
    return cost


def _optimal_grouping(sizes, s, exact_max_blocks=8):
    """Minimise pairwise error over groupings of one component's C-blocks.

    Exhaustive below `exact_max_blocks` (Bell(8) = 4140, trivial); greedy agglomerative
    above it, merging the most beneficial pair of groups until no merge helps.
    """
    k = len(sizes)
    if k == 1:
        return [0]

    if k <= exact_max_blocks:
        return min(_set_partitions(k), key=lambda g: _grouping_cost(g, sizes, s))

    groups = list(range(k))
    while True:
        best_delta, best_pair = 0.0, None
        labels = sorted(set(groups))
        for gi_idx, gi in enumerate(labels):
            for gj in labels[gi_idx + 1:]:
                members_i = [i for i in range(k) if groups[i] == gi]
                members_j = [j for j in range(k) if groups[j] == gj]
                delta = sum(sizes[i] * sizes[j] - 2.0 * s[i, j]
                            for i in members_i for j in members_j)
                if delta < best_delta:
                    best_delta, best_pair = delta, (gi, gj)
        if best_pair is None:
            return groups
        gi, gj = best_pair
        groups = [gi if g == gj else g for g in groups]


def best_reachable_merge(labels_c, labels_t, exact_max_blocks=8):
    """Best clustering reachable from C by merging whole component clusters.

    pangenomerge can only group whole component clusters, so the reachable clusterings are
    exactly the coarsenings of C. This returns the coarsening minimising pairwise error
    against T -- the true ceiling for acARI/acNMI.

    An earlier version simply merged every C-block in a connected component of the
    C-block/T-block bipartite graph. That drives c to zero but chains truth COGs together
    whenever one component cluster straddles two of them, adding more b than it removes;
    real merges then scored above the supposed ceiling (acARI > 1, and 3.5 for M. tb).
    Each component is now optimised on its own: merging blocks i and j trades s_ij
    wrongly-split pairs for |Bi||Bj| - s_ij wrongly-merged ones, so it is worth doing only
    when s_ij exceeds half of |Bi||Bj|.
    """
    table, c_codes, t_codes = contingency(labels_c, labels_t)
    n_c, n_t = table.shape
    table = table.tocsr()

    coo = table.tocoo()
    rows = np.concatenate([coo.row, coo.col + n_c])
    cols = np.concatenate([coo.col + n_c, coo.row])
    data = np.ones(len(rows), dtype=np.int8)
    bip = coo_matrix((data, (rows, cols)), shape=(n_c + n_t, n_c + n_t))
    n_comp, comp_labels = connected_components(bip, directed=False)

    c_blocks_by_component = defaultdict(list)
    for c_i in range(n_c):
        c_blocks_by_component[comp_labels[c_i]].append(c_i)
    t_per_component = defaultdict(set)
    for t_i in range(n_t):
        t_per_component[comp_labels[n_c + t_i]].add(t_i)

    block_group = {}
    next_group = 0
    n_chained = n_split = n_greedy = 0

    for comp, c_blocks in c_blocks_by_component.items():
        k = len(c_blocks)
        multi_truth = len(t_per_component[comp]) > 1
        if multi_truth:
            n_chained += 1

        if k == 1 or not multi_truth:
            # one truth COG in play: merging every block is optimal (all crossing pairs
            # are same-COG, so together costs 0 and apart costs s_ij > 0)
            for c_i in c_blocks:
                block_group[c_i] = next_group
            next_group += 1
            continue

        sub = table[c_blocks, :]
        sizes = np.asarray(sub.sum(axis=1)).ravel().astype(np.float64)
        s = (sub @ sub.T).toarray().astype(np.float64)
        if k > exact_max_blocks:
            n_greedy += 1
        grouping = _optimal_grouping(sizes, s, exact_max_blocks)

        remap = {}
        for c_i, g in zip(c_blocks, grouping):
            if g not in remap:
                remap[g] = next_group
                next_group += 1
            block_group[c_i] = remap[g]
        if len(set(grouping)) > 1:
            n_split += 1

    code_to_idx = {code: i for i, code in enumerate(c_codes)}
    labels_cstar = [block_group[code_to_idx[lab]] for lab in labels_c]

    diagnostics = {
        "n_component_clusters": int(n_c),
        "n_truth_clusters": int(n_t),
        "n_cstar_clusters": int(len(set(labels_cstar))),
        "n_chained_components": int(n_chained),
        "n_components_left_split": int(n_split),
        "n_components_greedy": int(n_greedy),
        "n_connected_components": int(n_comp),
    }
    return labels_cstar, diagnostics


def _rescale(value, low, high):
    """Linear rescale putting `low` at 0 and `high` at 1; nan if the anchors coincide."""
    span = high - low
    if span <= 0:
        return nan
    return (value - low) / span


def metrics_from_contingency(table, n, pc, with_ami=True):
    """ARI / MI / AMI / NMI computed from an already-built contingency table.

    sklearn's public scorers each re-factorise the label arrays and rebuild the contingency
    internally, which is the dominant cost when the labels are ~1M Python strings. Every
    quantity they need is already available here:

      ARI  is pure pair counting -- pc carries a, Sum C(a_i,2) and Sum C(b_j,2)
      MI   sklearn accepts a precomputed contingency directly
      AMI  MI, marginal entropies, and E[MI] (sklearn's Cython routine, given our table)
      NMI  MI over the same arithmetic-mean normaliser

    with_ami=False skips E[MI], which is the only expensive term here: it is O(R*C) over
    ~3000x3000 clusters (~10M gammaln evaluations) and measured at ~120 s, against ~1 s for
    everything else combined. Only AMI needs it -- NMI, and therefore pcNMI/acNMI, do not.

    Matches sklearn's adjusted_rand_score / mutual_info_score / adjusted_mutual_info_score /
    normalized_mutual_info_score with their default average_method='arithmetic'.
    """
    from sklearn.metrics.cluster._supervised import mutual_info_score
    from sklearn.metrics.cluster._expected_mutual_info_fast import (
        expected_mutual_information)

    # --- ARI (Hubert-Arabie) straight from the pair decomposition ---
    sum_comb = pc["a"]
    sum_a = pc["a"] + pc["b"]          # pairs co-clustered in X
    sum_b = pc["a"] + pc["c"]          # pairs co-clustered in T
    expected = sum_a * sum_b / pc["n_pairs"] if pc["n_pairs"] else 0.0
    max_index = 0.5 * (sum_a + sum_b)
    ari = (sum_comb - expected) / (max_index - expected) if max_index != expected else 1.0

    dense = table.toarray() if hasattr(table, "toarray") else np.asarray(table)
    mi = mutual_info_score(None, None, contingency=dense)

    # marginal entropies from the table, avoiding another pass over the labels
    def _entropy(sizes):
        sizes = np.asarray(sizes, dtype=np.float64)
        sizes = sizes[sizes > 0]
        p = sizes / sizes.sum()
        return float(-(p * np.log(p)).sum())

    h_x = _entropy(np.asarray(dense.sum(axis=1)).ravel())
    h_t = _entropy(np.asarray(dense.sum(axis=0)).ravel())
    normalizer = 0.5 * (h_x + h_t)          # average_method='arithmetic'

    nmi = mi / normalizer if normalizer > 0 else 0.0

    # MI = H(T) - H(T|X) = H(X) - H(X|T), so sklearn's homogeneity/completeness are just
    # MI over the respective marginal entropy -- no extra pass over the labels needed.
    # Their harmonic mean (V-measure) is 2*MI/(h_x+h_t) = NMI under arithmetic averaging,
    # so it is deliberately not returned: it would duplicate NMI exactly.
    #   low completeness -> under-merging (a truth COG still split across nodes)
    #   low homogeneity  -> over-merging  (a node mixing genes from different truth COGs)
    homogeneity = mi / h_t if h_t > 0 else 1.0
    completeness = mi / h_x if h_x > 0 else 1.0

    out = {"ARI": ari, "MI": mi, "NMI": nmi,
           "homogeneity": homogeneity, "completeness": completeness}
    if with_ami:
        emi = expected_mutual_information(dense, int(n))
        denom = normalizer - emi
        if abs(denom) < np.finfo(np.float64).eps:
            denom = np.finfo(np.float64).eps if denom >= 0 else -np.finfo(np.float64).eps
        out["AMI"] = (mi - emi) / denom
    return out


class GraphScorer:
    """Score many merges against one fixed (truth, component) pair.

    Everything depending only on T and C -- the gene universe, RI(C,T), NMI(C,T), the
    component pair counts and the reachable ceiling C* -- is computed once here rather
    than per merge. C* is the expensive part, and the method sweeps score dozens of merges
    against the same truth, so rebuilding it each time dominated the runtime (3h08m for
    M. tuberculosis when six scenarios each rebuilt the baseline).

    The gene universe is fixed at T n C (minus unmapped 'error' ids). A merge is expected
    to cover it -- pangenomerge only regroups component clusters, so its output holds every
    gene the components held. Anything missing is dropped with a warning and that merge's
    baseline is recomputed on the reduced set, since silently scoring on a shifting gene
    set would make runs incomparable.
    """

    def __init__(self, truth_map, component_map, compute_attainable=True):
        from sklearn.metrics import rand_score, normalized_mutual_info_score

        self.universe = sorted((set(truth_map) & set(component_map)) - {"error"})
        if not self.universe:
            raise ValueError("truth and component clusterings share no seqIDs")
        # factorise once -- see encode(); everything downstream works on int codes
        self.truth, _ = encode(_labels(truth_map, self.universe))
        self.component, _ = encode(_labels(component_map, self.universe))

        pc_c = pair_counts(self.component, self.truth)
        self.err_component = pc_c["b"] + pc_c["c"]
        self.b_component = pc_c["b"]      # irreducible: merging cannot separate these
        self.c_component = pc_c["c"]
        self.ri_component = rand_score(self.truth, self.component)
        self.nmi_component = normalized_mutual_info_score(self.truth, self.component)
        # C is maximally homogeneous and minimally complete by construction, so these
        # anchor the scale that homogeneity/completeness of a merge move along
        table_c, _, _ = contingency(self.component, self.truth)
        ent_c = metrics_from_contingency(table_c, len(self.truth), pc_c, with_ami=False)
        self.homogeneity_component = ent_c["homogeneity"]
        self.completeness_component = ent_c["completeness"]

        self.compute_attainable = compute_attainable
        if compute_attainable:
            cstar, diag = best_reachable_merge(self.component, self.truth)
            pc_cstar = pair_counts(cstar, self.truth)
            self.err_cstar = pc_cstar["b"] + pc_cstar["c"]
            self.nmi_cstar = normalized_mutual_info_score(self.truth, cstar)
            self.cstar_diagnostics = diag
            self.cstar_labels = cstar

    def score(self, merged_map, label=None, standard_scores=True):
        """Score one merge.

        standard_scores=False skips sklearn's ARI/MI/AMI. Those are the dominant cost and
        are not needed when comparing methods against each other -- the corrected metrics
        and the absolute pair counts carry that comparison. RI is always available because
        it comes free from the pair decomposition.
        """
        missing = [s for s in self.universe if s not in merged_map]
        if missing:
            logging.warning(
                f"{label or 'merge'}: {len(missing)} of {len(self.universe)} genes absent "
                f"from the merged clustering; scoring on the remainder.")
            keep = np.array([i for i, s in enumerate(self.universe) if s in merged_map])
            truth = self.truth[keep]
            component = self.component[keep]
            merged, _ = encode([merged_map[self.universe[i]] for i in keep])
            pc_c = pair_counts(component, truth)
            err_component = pc_c["b"] + pc_c["c"]
            ri_component = rand_score(truth, component)
            nmi_component = normalized_mutual_info_score(truth, component)
            table_c, _, _ = contingency(component, truth)
            ent_c = metrics_from_contingency(table_c, len(truth), pc_c, with_ami=False)
            hom_component = ent_c["homogeneity"]
            com_component = ent_c["completeness"]
        else:
            truth = self.truth
            merged, _ = encode(_labels(merged_map, self.universe))
            err_component = self.err_component
            ri_component = self.ri_component
            nmi_component = self.nmi_component
            hom_component = self.homogeneity_component
            com_component = self.completeness_component

        out = {"n_seqIDs": len(truth)}
        table_m, _, _ = contingency(merged, truth)
        pc_m = _pair_counts_from_table(table_m, len(truth))
        # RI straight from the pair decomposition -- identical to sklearn's rand_score but
        # reuses the contingency table already built here
        out["RI"] = 1.0 - (pc_m["b"] + pc_m["c"]) / pc_m["n_pairs"]
        std = metrics_from_contingency(table_m, len(truth), pc_m,
                                       with_ami=standard_scores)
        nmi_merged = std["NMI"]
        # ARI and MI are effectively free from the table; only AMI costs anything
        out["ARI"] = std["ARI"]
        out["MI"] = std["MI"]
        out["homogeneity"] = std["homogeneity"]
        out["completeness"] = std["completeness"]
        if standard_scores:
            out["AMI"] = std["AMI"]
        err_merged = pc_m["b"] + pc_m["c"]
        out.update({
            "pairs_total": pc_m["n_pairs"],
            "err_merged": err_merged, "err_component": err_component,
            "b_merged": pc_m["b"], "c_merged": pc_m["c"],
            "b_component": self.b_component, "c_component": self.c_component,
            "RI_component": ri_component,
            "NMI_merged": nmi_merged, "NMI_component": nmi_component,
            "homogeneity_component": hom_component,
            "completeness_component": com_component,
        })
        out["pcARI"] = _rescale(out["RI"], ri_component, 1.0)
        out["pcNMI"] = _rescale(nmi_merged, nmi_component, 1.0)

        if not self.compute_attainable:
            return out

        out.update({f"cstar_{k}": v for k, v in self.cstar_diagnostics.items()})
        out["err_cstar"] = self.err_cstar
        out["NMI_cstar"] = self.nmi_cstar
        # fraction of *removable* error removed, vs pcARI's fraction of *total* error
        out["acARI"] = _rescale(err_component - err_merged, 0.0,
                                err_component - self.err_cstar)
        out["acNMI"] = _rescale(nmi_merged, nmi_component, self.nmi_cstar)

        if out["acARI"] is not nan and out["acARI"] > 1.0:
            d = self.cstar_diagnostics
            logging.warning(
                f"acARI = {out['acARI']:.4f} > 1: the merge beat the reachable ceiling. "
                f"{d['n_chained_components']} of {d['n_connected_components']} components "
                f"chain multiple truth COGs, so C* may understate the ceiling.")
        return out


def graph_similarity_scores(truth_map, merged_map, component_map,
                            compute_attainable=True):
    """One-shot wrapper around GraphScorer.

    Use GraphScorer directly when scoring several merges against the same truth: this
    rebuilds the whole baseline, including C*, on every call.
    """
    return GraphScorer(truth_map, component_map, compute_attainable).score(merged_map)
