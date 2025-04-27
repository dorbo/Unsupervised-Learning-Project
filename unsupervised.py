from pathlib import Path

import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import prince
import umap

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA

from sklearn.metrics import silhouette_score

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture

from sklearn.manifold import TSNE


DATASET_NAME = "vishakhdapat/customer-segmentation-clustering"
DATASET_PATH = "customer_segmentation.csv"
IRRELEVANT_COLUMNS = ['ID', 'Dt_Customer']

N_COMPONENTS_MIN = 5
N_COMPONENTS_MAX = 19
N_CLUSTERS_MIN = 2
N_CLUSTERS_MAX = 19

MCA_N_COMPONENTS = 1
ICA_N_COMPONENTS = 7
N_CLUSTERS = 5

DBSCAN_EPS = 0.1
DBSCAN_MIN_SAMPLES = 3

TSNE_PERPLEXITY = 30
TSNE_MAX_ITER = 250

CLUSTERING_RUN_AMOUNT = 128

def achieve_dataset_dataframe(name, path):
    prefix_path = kagglehub.dataset_download(name)
    return pd.read_csv(Path(prefix_path) / path)

def clear_df(df, columns):
    df = df.drop(columns=columns)
    return df.dropna()

def mca(df, n_components):
    mca = prince.MCA(n_components=n_components)

    categorial_df = df.select_dtypes(include=['object', 'category'])
    mca_result = mca.fit(categorial_df).transform(categorial_df)
    mca_result.columns = [f'MCA_{i + 1}' for i in range(n_components)]
    
    return pd.concat([df.drop(columns=categorial_df), mca_result], axis=1)

def scale_df(df):
    scaler = StandardScaler()
    return scaler.fit_transform(df)

def pca(df, n_components):
    pca_obj = PCA(n_components=n_components)
    pca_result = pca_obj.fit_transform(df)
    return pd.DataFrame(pca_result, columns=[f'PCA_{i + 1}' for i in range(n_components)])

def ica(df, n_components):
    ica_obj = FastICA(n_components=n_components)
    ica_result = ica_obj.fit_transform(df)
    return pd.DataFrame(ica_result, columns=[f'ICA_{i + 1}' for i in range(n_components)])

def kmeans(df, n_clusters):
    kmeans_obj = KMeans(n_clusters=n_clusters)
    kmeans_obj.fit(df)
    return kmeans_obj.labels_

def dbscan(df, eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES):
    dbscan_obj = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan_obj.fit_predict(df)
    return labels

def gmm(df, n_clusters):
    gmm = GaussianMixture(n_components=n_clusters)
    gmm.fit(df)
    return gmm.predict(df)

def hierarchial_agglomerative(df, n_clusters):
    agglomerative_clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    return agglomerative_clustering.fit_predict(df)

def tsne(df, n_components=2, perplexity=TSNE_PERPLEXITY, max_iter=TSNE_MAX_ITER):
    tsne_obj = TSNE(n_components=n_components, perplexity=perplexity, max_iter=max_iter)
    tsne_result = tsne_obj.fit_transform(df)
    return pd.DataFrame(tsne_result, columns=[f'TSNE_{i + 1}' for i in range(n_components)])

def umap_(df, n_components=2):
    umap_model = umap.UMAP(n_components=n_components)
    umap_result = umap_model.fit_transform(df)
    return pd.DataFrame(umap_result, columns=[f'UMAP_{i + 1}' for i in range(n_components)])

def plot(result, labels=None, title='', algorithm='tsne'):
    plt.figure(figsize=(8, 6))

    if labels is not None:
        plt.scatter(result[f'{algorithm.upper()}_1'], result[f'{algorithm.upper()}_2'], c=labels)
    else:
        plt.scatter(result[f'{algorithm.upper()}_1'], result[f'{algorithm.upper()}_2'], c=labels)
    
    plt.title(f'{algorithm.upper()} 2D {title}')
    plt.xlabel(f'{algorithm.upper()}_1')
    plt.ylabel(f'{algorithm.upper()}_2')

def get_dbscan_scores_by_cluster_amount(df):
    scores = {}
    max_cluster_amount = 0
    for min_samples in range(1,5):
        for eps in range(10, 50, 5):
            print(f'dbscan {eps}, {min_samples}')
            eps /= 10
            labels = dbscan(df, eps=eps, min_samples=min_samples)
            has_noise = -1 in labels
            score = silhouette_score(df, labels)
            labels_set_size = len(set(labels))
            cluster_amount = labels_set_size - 1 if has_noise else labels_set_size
            
            max_cluster_amount = max(max_cluster_amount, cluster_amount)
            scores[cluster_amount] = max(scores.get(cluster_amount, 0), score)
    
    scores_by_cluster_amount = [0] * (max_cluster_amount + 1)
    for k, v in scores.items():
        scores_by_cluster_amount[k] = v
    return scores_by_cluster_amount

def get_clustering_scores(df):
        component_scores = {
            'dbscan': get_dbscan_scores_by_cluster_amount(df),
            'kmeans': [],
            'gmm': []
        }
        for n_clusters in range(N_CLUSTERS_MIN, N_CLUSTERS_MAX + 1):
            print(f'kmean, gmm, {n_clusters}')
            kmeans_labels = kmeans(df, n_clusters=n_clusters)
            gmm_labels = gmm(df, n_clusters=N_CLUSTERS)
            component_scores['kmeans'].append(silhouette_score(df, kmeans_labels))
            component_scores['gmm'].append(silhouette_score(df, gmm_labels))
        return component_scores

def get_dim_reduction_and_clustering_scores(df):
    scores = []
    for n_components in range(N_COMPONENTS_MIN, N_COMPONENTS_MAX + 1):
        pca_result = pca(df, n_components=n_components)
        ica_result = ica(df, n_components=n_components)
        scores.append({
            'pca': get_clustering_scores(pca_result),
            'ica': get_clustering_scores(ica_result)
        })
    return scores

def get_clusters_total_dbscan(scores, alg):
    max_clusters = 0
    for i in range(len(scores)):
        clusters = len(scores[i][alg]['dbscan'])
        if clusters > max_clusters:
            max_clusters = clusters
    return max_clusters

def get_2d_array(scores, alg, clustering):
    components_total = N_COMPONENTS_MAX - N_COMPONENTS_MIN
    clusters_total = N_CLUSTERS_MAX - N_CLUSTERS_MIN if clustering != 'dbscan' else get_clusters_total_dbscan(scores, alg)
    array = np.zeros((components_total, clusters_total))
    for i in range(components_total):
        for j in range(clusters_total):
            if j < len(scores[i][alg][clustering]):
                array[i][j] = scores[i][alg][clustering][j]
            else:
                array[i][j] = 0
    return array      

def create_heatmap_figure(ax, arr, dim, clustering):
    ax.set_title(f'{dim} + {clustering}')
    sns.heatmap(arr, ax=ax, cmap='viridis', cbar_kws={'shrink': 0.5})
    ax.set_xlabel('Clusters Amount')
    ax.set_xticks(np.arange(arr.shape[1]) + 0.5)
    ax.set_yticks(np.arange(arr.shape[0]) + 0.5)
    
    if clustering != 'dbscan':
        ax.set_xticklabels(np.arange(N_CLUSTERS_MIN, N_CLUSTERS_MIN + arr.shape[1]))
        ax.set_yticklabels(np.arange(N_COMPONENTS_MIN, N_COMPONENTS_MIN + arr.shape[0]))
    else:
        ax.set_yticklabels(np.arange(1, 1 + arr.shape[0]))
    ax.set_ylabel('Dimentions Amount')
    

def generate_graphs(scores):
    _, axes = plt.subplots(3, 2, figsize=(10, 10))
    for i, clustering_algorithm in enumerate(['kmeans', 'dbscan', 'gmm']):
        for j, dim_algorithm in enumerate(['pca', 'ica']):
            array = get_2d_array(scores, dim_algorithm, clustering_algorithm)
            create_heatmap_figure(axes[i, j], array, dim_algorithm, clustering_algorithm)
    plt.tight_layout()
    plt.show()

def get_best_algorithm_dimention_and_clusters(scores):
    max_score = 0
    index = None
    for dim in range(len(scores)):
        for dim_algorithm in ['pca', 'ica']:
            for clustering_algorithm in ['kmeans', 'dbscan', 'gmm']:
                for i, score in enumerate(scores[dim][dim_algorithm][clustering_algorithm]):
                    if score > max_score:
                        max_score = score
                        index = dim + N_COMPONENTS_MIN, dim_algorithm, clustering_algorithm, i + (N_CLUSTERS_MIN if clustering_algorithm != 'dbscan' else 0)
                        print(score, index)
    print(index)
    return index

def check_clustering(df):
    scores = get_dim_reduction_and_clustering_scores(df)
    generate_graphs(scores)
    return get_best_algorithm_dimention_and_clusters(scores)

def check_dimention_reduction_clustering_algorithm_and_amount():
    df = achieve_dataset_dataframe(DATASET_NAME, DATASET_PATH)
    df = clear_df(df, columns=IRRELEVANT_COLUMNS)
    df = mca(df, n_components=MCA_N_COMPONENTS)
    df = scale_df(df)
    best_results = check_clustering(df)
    print(f'Best Results: {best_results}')
    # Best Results:
    # 7 Dimentions using ICA +
    # 5 Clusters using GMM

def run_kmeans_many_times(df):
    kmeans_results = []
    best_score = 0
    best_labels = None
    for _ in range(CLUSTERING_RUN_AMOUNT):
        labels = kmeans(df, n_clusters=N_CLUSTERS)
        score = silhouette_score(df, labels)
        kmeans_results.append(score)
        if score > best_score:
            best_score = score
            best_labels = labels

    return kmeans_results, best_labels

def run_gmm_many_times(df):
    gmm_results = []
    best_score = 0
    best_labels = None
    for _ in range(CLUSTERING_RUN_AMOUNT):
        labels = gmm(df, n_clusters=N_CLUSTERS)
        score = silhouette_score(df, labels)
        gmm_results.append(score)
        if score > best_score:
            best_score = score
            best_labels = labels

    return gmm_results, best_labels

def paired_t_test_clustering_algorithms(df):
    kmeans_results, kmeans_best_labels = run_kmeans_many_times(df)
    gmm_results, gmm_best_labels = run_gmm_many_times(df)
    
    _, p_val = stats.ttest_rel(kmeans_results, gmm_results)
    print(f'kmeans_results, gmm_results {p_val}') # 5.422998556940823e-41
    print(f'kmeans mean: {np.mean(kmeans_results)}') # 0.2018806140581344
    print(f'gmm mean: {np.mean(gmm_results)}')  # 0.14514109029726902
    
    if np.mean(kmeans_results) > np.mean(gmm_results):
        return kmeans_best_labels
    return gmm_best_labels

def test_clustering():
    df = achieve_dataset_dataframe(DATASET_NAME, DATASET_PATH)
    df = clear_df(df, columns=IRRELEVANT_COLUMNS)
    df = mca(df, n_components=MCA_N_COMPONENTS)
    df = scale_df(df)
    df = ica(df, n_components=ICA_N_COMPONENTS)
    best_labels = paired_t_test_clustering_algorithms(df)
    
    tsne_result = tsne(df)
    plot(tsne_result, best_labels, 'KMEANS', 'tsne')

    umap_result = umap_(df)
    plot(umap_result, best_labels, 'KMEANS', 'umap')

    plt.show()
