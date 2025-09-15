#%%
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans

#%% Import data

df = pd.DataFrame(columns=['title','authors','citation','abstract'])
for i in range(122, 124):
    for j in range(1,26):
        if i == 123 and j > 6:
            break
        else: 
            fname = f'prl_data/vol{i}_issue{j}'
            df_temp = pd.read_csv(fname)
            df_temp.drop(columns=['Unnamed: 0'], inplace=True)
            df = pd.concat([df,df_temp], axis=0)

display(df.head())
print(df.shape)

#%%

print(df.isnull().sum())
print(df.shape)
df_full = df.loc[~df.abstract.isnull(), :]

df_full.shape

# %% vectorize and reduce data

vectorizer = TfidfVectorizer(min_df=2, max_df=0.5, 
                             stop_words="english", use_idf=True)

def transform(df, vectorizer, dimensions):
    data = df.abstract
    trans_data = vectorizer.fit_transform(data)

    print(f'Transformed data contains: {str(trans_data.shape[0])} samples and {str(trans_data.shape[1])} features')

    svd = TruncatedSVD(dimensions)
    pipe = make_pipeline(svd, Normalizer(copy=False))
    reduced_data = pipe.fit_transform(trans_data)

    return reduced_data, svd

reduced_data, svd = transform(df_full, vectorizer, 500)
print(f'Reduced data if of {type(reduced_data)} type')
print(f'Reduced data contains: {str(reduced_data.shape[0])} samples and {str(reduced_data.shape[1])} features')

# %% make kmeans

num_clusters = 20

def cluster(data, num_clusters):
    km = KMeans(n_clusters=num_clusters,
                n_init='auto',
                max_iter=100,
                random_state=0)
    km.fit(data)
    
    #Eval, silhouette
    cluster_labels = km.predict(data)
    silhouette_avg = silhouette_score(data, cluster_labels)
    print(f'silhouette score: {round(silhouette_avg, 5)}')
    
    # Calculate number of samples in each group
    cluster_ids, cluster_sizes = np.unique(km.labels_, return_counts=True)
    print(f"Number of elements assigned to each cluster: {cluster_sizes}")
    
    return km

km = cluster(reduced_data, num_clusters)

#%% DBSCAN test
from sklearn.cluster import DBSCAN, HDBSCAN

def dbscan_cluster(data):
    # min_samples == minimum points ≥ dataset_dimensions + 1
    dbs = HDBSCAN(min_cluster_size=9) #eps=0.24, min_samples=5)
    dbs.fit(data)
    
    labels = dbs.labels_
    #Eval, silhouette
    silhouette_avg = silhouette_score(data, labels)
    print(f'silhouette score: {round(silhouette_avg, 5)}')
    
    # Calculate number of samples in each group
    cluster_ids, cluster_sizes = np.unique(labels, return_counts=True)
    print(f'Number of classes: {len(cluster_ids)}')
    print(f"Number of elements assigned to each cluster: {cluster_sizes}")
    return

dbscan_cluster(reduced_data)

# %% eval kmeans

def evaluate(km, svd, num_clusters):
    print('Clustering report\n')
    print('Most descriminatice words per cluster:')
    
    original_space_centroids = svd.inverse_transform(km.cluster_centers_)
    ordered_centroids = original_space_centroids.argsort()[:,::-1]

    terms = vectorizer.get_feature_names_out()
    for i in range(num_clusters):
        print(f'Cluster {str(i)}:')
        cl_terms = ''
        for ind in ordered_centroids[i,:10]:        
            cl_terms += terms[ind] + ' '
        print(cl_terms + '\n')
    return 

evaluate(km, svd, num_clusters)

# %% Evaluate KMeans copied from sklearn
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples, silhouette_score

X = reduced_data
range_n_clusters = [15, 20, 25, 30, 35, 40]

for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, n_init="auto", random_state=10)
    cluster_labels = clusterer.fit_predict(X)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print(
        "For n_clusters =",
        n_clusters,
        "The average silhouette_score is :",
        silhouette_avg,
    )

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(
        X[:, 0], X[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
    )

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="o",
        c="white",
        alpha=1,
        s=200,
        edgecolor="k",
    )

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(
        "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
        % n_clusters,
        fontsize=14,
        fontweight="bold",
    )

plt.show()

# %% Try LDA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


def display_topics(model, features, no_top_words=10):
    for topic, word_vector in enumerate(model.components_):
        total = word_vector.sum()
        largest = word_vector.argsort()[::-1]
        print(f'\nTopic: {topic}')
        for i in range(0, no_top_words):
            print(f'{str(features[largest[i]])} ({round(word_vector[largest[i]]*100.0/total,2)})')


prl_cv = CountVectorizer(min_df=2, max_df=0.5, stop_words='english')
prl_vectors = prl_cv.fit_transform(df_full.abstract)

lda_model = LatentDirichletAllocation(n_components=10, random_state=42)
W_lda_prl = lda_model.fit_transform(prl_vectors)
H_lda_prl = lda_model.components_

display_topics(lda_model, prl_cv.get_feature_names_out())

# %%
import pyLDAvis.lda_model

lda_display = pyLDAvis.lda_model.prepare(lda_model, prl_vectors,
                                       prl_cv, sort_topics=False)
pyLDAvis.display(lda_display)

# %% Try cleaning with spacy
import spacy

spacy.prefer_gpu()
nlp = spacy.load("en_core_web_md")


# %%
