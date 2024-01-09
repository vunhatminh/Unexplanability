import numpy as np
import pandas as pd
import math
from scipy.special import softmax
from scipy import stats
from pgmpy.estimators import HillClimbSearch, BicScore
from pgmpy.models import BayesianModel
from pgmpy.inference import VariableElimination


big_num = 100000


def perturb_features_on_column(x, columns, random_mode='random',
                               background=None):
    ''' Perturb a single column of a matrix x.
    '''
    x_perturb = x.copy()
    if random_mode == 'random':
        for col in columns:
            x_perturb[col] = np.random.random(size=x_perturb.shape[0])
    if random_mode == 'mean':
        for col in columns:
            x_perturb[col] = np.mean(x_perturb[col])
    if random_mode == 'add_big':
        for col in columns:
            x_perturb[col] = x_perturb[col] + big_num
    if random_mode == 'permute':
        if background is None:
            background = x.copy()
        for col in columns:
            x_perturb[col] = np.random.permutation(background[col])[:x_perturb.shape[0]]
    return x_perturb


def explain(model, to_explain, target, num_samples, pred_threshold,
            p_threshold=0.05, num_top_nodes=None, groups=None, background=None):
    '''
    Explain the model using the Bayesian model structure.

    Parameters:
    -----------
    model: XGBoost model
        The model to explain.
    to_explain: DataFrame
        The data to explain.
    target: int
        The index of the output class to be explained.
    num_samples: int
        The number of samples to use in the explanation.
    pred_threshold: float
        The threshold for the predicted probability.
    p_threshold: float
        The threshold for the p-value.
    num_top_nodes: int
        The number of top nodes to use in the explanation. If None, use all.
    groups: list
        The list of groups that should be explained together.
    '''
    # target - output dimension
    # sample_idx - which sample to explain
    samples = []
    pred_samples = []
    pred_torch = model.predict_proba(to_explain)
    soft_pred = np.asarray(
        [
            softmax(np.asarray(pred_torch[node_]))
            for node_ in range(to_explain.shape[0])
        ])
    num_nodes = to_explain.shape[1]
    # Saving the column order, to be able to restore it later
    col_order = to_explain.columns.tolist()
    # Map nodes to their column indices
    if groups:
        # Counting how the size of our design matrix will be,
        # after we merge the nodes.
        merged_nodes_count = 0
        for name, group in groups.items():
            merged_nodes_count += len(group) - 1
        num_nodes -= merged_nodes_count
        # Assigning of indices of columns to the nodes
        node_mapping = {}
        # First assign all singular nodes
        for node in range(to_explain.shape[1]):
            if not any(node in group for name, group in groups.items()):
                node_mapping[col_order[node]] = [col_order[node]]
        # Then assign all the groups
        for name, group in groups.items():
            node_mapping[name] = [col_order[node] for node in group]
    else:
        # If there are no groups, we just have all nodes
        node_mapping = {
            col: [col] for col in col_order
            }
    # Step 1 of the algorithm
    for iteration in range(num_samples):
        x_perturb = to_explain.copy()
        sample = []
        for id, columns in node_mapping.items():
            seed = np.random.randint(2)
            if seed == 1:
                latent = 1
                x_perturb = perturb_features_on_column(x_perturb,
                                                       node_mapping[id],
                                                       random_mode='permute',
                                                       background=background)
            else:
                latent = 0
            sample.append(latent)
        pred_perturb_torch = model.predict_proba(x_perturb)
        soft_pred_perturb = np.asarray(
            [
                softmax(np.asarray(pred_perturb_torch[node_]))
                for node_ in range(to_explain.shape[0])
            ])
        sample_bool = []
        for col in range(to_explain.shape[0]):
            soft_pred_with_threshold = (
                soft_pred_perturb[col, target] + pred_threshold)
            if soft_pred_with_threshold < soft_pred[col, target]:
                sample_bool.append(1)
            else:
                sample_bool.append(0)

        samples.append(sample)
        pred_samples.append(sample_bool)

    samples = np.asarray(samples)
    samples = np.repeat(samples, to_explain.shape[0], axis=0)
    pred_samples = np.asarray(pred_samples).flatten()
    data = pd.DataFrame(samples, columns=node_mapping.keys())
    data['target'] = pred_samples
    # Step 2 of the algorithm
    p_values = []
    dependent_neighbors = []
    dependent_neighbors_p_values = []
    for feature, columns in node_mapping.items():
        # Is feature dependent on target?
        chi2, p = chi_square(feature,
                             'target',
                             [],
                             data)
        p_values.append(p)
        if p < p_threshold:
            # If feature is dependent on target,
            # add to dependent_neighbors
            dependent_neighbors.append(feature)
            dependent_neighbors_p_values.append(p)
    pgm_stats = dict(zip(range(num_nodes), p_values))

    pgm_nodes = []
    if num_top_nodes is None:
        pgm_nodes = dependent_neighbors
    else:
        top_p = np.min((num_top_nodes, num_nodes - 1))
        ind_top_p = np.argpartition(p_values, top_p)[0:top_p]
        pgm_nodes = [data.columns[node] for node in ind_top_p]

    if groups:
        return pgm_nodes, data, pgm_stats, node_mapping
    return pgm_nodes, data, pgm_stats


def search_MK(data, target, nodes, p_value_threshold=0.05, verbose=True):
    data.columns = data.columns.astype(str)
    nodes = [str(node) for node in nodes]
    MB = nodes
    while True:
        count = 0
        for node in nodes:
            evidences = MB.copy()
            evidences.remove(node)
            if verbose:
                print(target, node, evidences)
            _, p = chi_square(target, node, evidences, data[nodes + [target]])
            if p > p_value_threshold:
                MB.remove(node)
                count = 0
            else:
                count = count + 1
                if count == len(MB):
                    return MB


def generalize_target(x):
    if x > 10:
        return x - 10
    else:
        return x


def generalize_others(x):
    if x == 2:
        return 1
    elif x == 12:
        return 11
    else:
        return x


def generate_evidence(evidence_list):
    return dict(zip(evidence_list, [1 for node in evidence_list]))


def pgm_generate(target, data, subnodes, child=None, p_value_threshold=0.05,
                 verbose=True):
    # Step 3 of the algorithm
    if not subnodes:
        raise ValueError('No subnodes')
    subnodes = [str(node) for node in subnodes]
    target = 'target'
    subnodes_no_target = [node for node in subnodes if node != target]

    data.columns = data.columns.astype(str)

    MK_blanket = search_MK(data, target, subnodes_no_target.copy(),
                           p_value_threshold=p_value_threshold,
                           verbose=verbose)

    if child is None:
        est = HillClimbSearch(data[subnodes_no_target],
                              scoring_method=BicScore(data))
        pgm_no_target = est.estimate()
        for node in MK_blanket:
            if node != target:
                pgm_no_target.add_edge(node, target)

        # Create the pgm
        pgm_explanation = BayesianModel()
        for node in pgm_no_target.nodes():
            pgm_explanation.add_node(node)
        for edge in pgm_no_target.edges():
            pgm_explanation.add_edge(edge[0],
                                     edge[1])

        # Fit the pgm
        data_ex = data[subnodes].copy()
        data_ex[target] = data[target].apply(generalize_target)
        for node in subnodes_no_target:
            data_ex[node] = data[node].apply(generalize_others)
        pgm_explanation.fit(data_ex)
    else:
        data_ex = data[subnodes].copy()
        data_ex[target] = data[target].apply(generalize_target)
        for node in subnodes_no_target:
            data_ex[node] = data[node].apply(generalize_others)

        est = HillClimbSearch(data_ex, scoring_method=BicScore(data_ex))
        pgm_w_target_explanation = est.estimate()

        # Create the pgm
        pgm_explanation = BayesianModel()
        for node in pgm_w_target_explanation.nodes():
            pgm_explanation.add_node(node)
        for edge in pgm_w_target_explanation.edges():
            pgm_explanation.add_edge(edge[0], edge[1])

        # Fit the pgm
        data_ex = data[subnodes].copy()
        data_ex[target] = data[target].apply(generalize_target)
        for node in subnodes_no_target:
            data_ex[node] = data[node].apply(generalize_others)
        pgm_explanation.fit(data_ex)

    return pgm_explanation


def pgm_conditional_prob(target, pgm_explanation, evidence_list):
    evidence_list = [str(node) for node in evidence_list]
    pgm_infer = VariableElimination(pgm_explanation)
    for node in evidence_list:
        if node not in list(pgm_infer.variables):
            print("Not valid evidence list.")
            return None
    evidences = generate_evidence(evidence_list)
    elimination_order = [node for node in list(
        pgm_infer.variables) if node not in evidence_list]
    elimination_order = [node for node in elimination_order if node != target]
    q = pgm_infer.query([target], evidence=evidences,
                        elimination_order=elimination_order,
                        show_progress=False)
    return q.values[0]


def chi_square(X, Y, Z, data):
    """
    Modification of Chi-square conditional independence test from pgmpy
    Tests the null hypothesis that X is independent from Y given Zs.
    Parameters
    ----------
    X: int, string, hashable object
        A variable name contained in the data set
    Y: string, hashable object
        A variable name contained in the data set, different from X
    Zs: list of variable names
        A list of variable names contained in the data set, different from X
        and Y. This is the separating set that (potentially) makes X and Y
        independent.
        Default: []
    Returns
    -------
    chi2: float
        The chi2 test statistic.
    p_value: float
        The p_value, i.e. the probability of observing the computed chi2
        statistic (or an even higher value), given the null hypothesis
        that X _|_ Y | Zs.
    sufficient_data: bool
        A flag that indicates if the sample size is considered sufficient.
        As in [4], require at least 5 samples per parameter (on average).
        That is, the size of the data set must be greater than
        `5 * (c(X) - 1) * (c(Y) - 1) * prod([c(Z) for Z in Zs])`
        (c() denotes the variable cardinality).
    References
    ----------
    [1] Koller & Friedman, Probabilistic Graphical Models - Principles and
    Techniques, 2009 Section 18.2.2.3 (page 789)
    [2] Neapolitan, Learning Bayesian Networks, Section 10.3 (page 600ff)
        http://www.cs.technion.ac.il/~dang/books/Learning%20Bayesian%20Networks(Neapolitan,%20Richard).pdf
    [3] Chi-square test
    https://en.wikipedia.org/wiki/Pearson%27s_chi-squared_test#Test_of_independence
    [4] Tsamardinos et al., The max-min hill-climbing BN structure learning
        algorithm, 2005, Section 4
    """
    X = str(X)
    Y = str(Y)
    if isinstance(Z, (frozenset, list, set, tuple)):
        Z = list(Z)
    Z = [str(z) for z in Z]
    data.columns = data.columns.astype(str)
    state_names = {
        var_name: data.loc[:, var_name].unique() for var_name in data.columns
    }
    row_index = state_names[X]
    column_index = pd.MultiIndex.from_product(
            [state_names[Y]] + [state_names[z] for z in Z], names=[Y] + Z
        )

    XYZ_state_counts = pd.crosstab(
                index=data[X], columns=[data[Y]] + [data[z] for z in Z],
                rownames=[X], colnames=[Y] + Z
            )

    if not isinstance(XYZ_state_counts.columns, pd.MultiIndex):
        XYZ_state_counts.columns = pd.MultiIndex.from_arrays(
            [XYZ_state_counts.columns])
    XYZ_state_counts = XYZ_state_counts.reindex(
            index=row_index, columns=column_index
        ).fillna(0)

    if Z:
        # marginalize out Y
        XZ_state_counts = XYZ_state_counts.sum(axis=1,
                                               level=list(range(1,
                                                                len(Z)+1)))
        # marginalize out X
        YZ_state_counts = XYZ_state_counts.sum().unstack(Z)
    else:
        XZ_state_counts = XYZ_state_counts.sum(axis=1)
        YZ_state_counts = XYZ_state_counts.sum()
    Z_state_counts = YZ_state_counts.sum()  # marginalize out both

    XYZ_expected = np.zeros(XYZ_state_counts.shape)

    r_index = 0
    for X_val in XYZ_state_counts.index:
        X_val_array = []
        if Z:
            for Y_val in XYZ_state_counts.columns.levels[0]:
                temp = (XZ_state_counts.loc[X_val]
                        * YZ_state_counts.loc[Y_val]
                        / Z_state_counts)
                X_val_array = X_val_array + list(temp.to_numpy())
            XYZ_expected[r_index] = np.asarray(X_val_array)
            r_index += 1
        else:
            for Y_val in XYZ_state_counts.columns:
                temp = (XZ_state_counts.loc[X_val]
                        * YZ_state_counts.loc[Y_val]
                        / Z_state_counts)
                X_val_array = X_val_array + [temp]
            XYZ_expected[r_index] = np.asarray(X_val_array)
            r_index += 1

    observed = XYZ_state_counts.to_numpy().reshape(1, -1)
    expected = XYZ_expected.reshape(1, -1)
    observed, expected = zip(
        *((o, e) for o, e in zip(observed[0], expected[0])
            if not (e == 0 or math.isnan(e))))
    chi2, significance_level = stats.chisquare(observed, expected)

    return chi2, significance_level
