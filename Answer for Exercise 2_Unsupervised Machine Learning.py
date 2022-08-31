###################################################
""" Code written for the Exercise 2 Unsupervised Machine Learning of the Python and Machine Learning.
       Written by Worku,Biniyam
       Submitted to Michael E. Rose (PhD)
       Date of submission date August 30, 2022.
"""
# ------------------------- QUESTION ONE---------------
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris

from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.metrics import silhouette_score
# -------------------------------------------------------------
# 1.a
# loading the dataset
from sklearn.datasets import fetch_california_housing

# ------------------------------------------------
# Problem 1.b Extracting the Polynomial Features

fetch_california = fetch_california_housing()
X = fetch_california["data"]

poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
poly_tra = poly.fit_transform(X)
poly_fin = poly.get_feature_names_out(fetch_california["feature_names"])

print(poly.n_output_features_)   # To print the number of polynomial Features
# Accordingly the fetch_california_housing have 44 polynomial Features.
# ------------------------------------------------__________________
# Problem 1.c Saving the dataset into Pandas DataFrame
df = pd.DataFrame(poly_tra, columns=poly_fin)
df["Y"] = fetch_california["target"]
df.to_csv("output/polynomials.csv")  # saving the output to csv format.

# ------------------------------------------------------------
# PROBLEMS 2. Principal Component Analysis
# ------------------------------------------------------------
# 2.a Reading the saved dataset from the Pandas DataFrame
olympy = pd.read_csv("./data/olympics.csv", sep=',', index_col=0)
print(olympy.describe())

# The score variable is a cumulative rank of the participants, it is a composite function of other variables. I ///
# decided to drop the score variable.

olympy = olympy.drop(columns='score')
# -2.b -----------------------------------------------------------
# Am using StandardScaler because it standardize features by removing the mean and scaling to unit variance.

scaler = StandardScaler()
scaler.fit(olympy)
olympy_scaled = pd.DataFrame(scaler.transform(olympy))

# 2.c-----------------------------------------------------------
pca = PCA(random_state=3)
pca.fit(olympy_scaled)  # If we are going to save it, change the scaled name to the new name 2b

Scaled_olymp = pd.DataFrame(pca.components_, columns=olympy.columns)

print(pca.components_)  # Inspecting the components for the most prominently components.
# List of Prominent Variables
# First component: 110m Hurdles (0.43348), 100m sprint (0.41588) and Long running (|-0.39405|)
# Second component: Discus throw (0.5033), Poil (0.48354) and 1500m run (0.42097)
# Third component: Haut (0.85499), 100m sprint (0.26747) and 1500m running (|-0.22255|)
# Note that the value of the variable is indicated in the apprentice.

# -2.d-----------------------------------------------------------
df_var = pd.DataFrame(pca.explained_variance_ratio_)

cumultv = df_var.cumsum()

# At least Seven components are needed to explain the 90% of the dataset. The Seventh component explain 93.24%
# of the dataset.
# -----------------------------------------------------------
# -----------------------------------------------------------
# QUESTION 3
# ------------------------------------------------------------
# 3.a Loading the dataset from the sklearn and saving to pandas DataFrame for future tasks.
iris = load_iris()
x = pd.DataFrame(iris["data"])
# ------------------------------------------------------------
# 3.b Scalling the dataset

scaler = StandardScaler()
scaler.fit(x)
x_scaled = pd.DataFrame(scaler.transform(x))

# ------------------------------------------------------------
# 3.c Estimating the kmeans, Agglomerative and DBSCAN model, and saving to the pandas DataFrame.
# K Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(x_scaled)

# Agglomerative Clustering
agg = AgglomerativeClustering(n_clusters=3)
agg.fit(x_scaled)

# DBSCAN
dbscan = DBSCAN(eps=1, min_samples=2, metric='euclidean')
dbscan.fit(x_scaled)

combined = pd.DataFrame({"kmeans": kmeans.labels_, "agg": agg.labels_, "dbscan": dbscan.labels_})
# ------------------------------------------------------------
# 3.d
# Printing the silhouette Score for the silhouette Score

print(silhouette_score(x_scaled, kmeans.labels_))
print(silhouette_score(x_scaled, agg.labels_))
print(silhouette_score(x_scaled, dbscan.labels_))

# We should give attention and treat noise assignment of the DBSCAN is because the DBSCAN could cluster an observation
# to one of the cluster which does not have any relationship with the cluster. The wrongly clustered observation will
# represented by -1.
# The third model is the model with the highest noise.
# ------------------------------------------------------------
# 3.e Adding new variable to the pandas DataFrame.
combined["sepal width"] = x[1]
combined["petal width"] = x[2]
# ------------------------------------------------------------
# 3.f Renaming the noise variable.

combined["dbscan"] = combined["dbscan"].replace(-1, "Noise")

# ------------------------------------------------------------
# 3.g Plotting and saving Figure
id_vars = ["sepal width", "petal width"]
value_vars = ['kmeans', 'agg', 'dbscan']
melted = combined.melt(id_vars=id_vars, value_vars=value_vars)
fig = sns.relplot(x="sepal width", y="petal width", data=melted, hue="value", col="variable")
fig.savefig("./output/cluster_petal.pdf")
# --------------------------------- END ------------------------------------------------------------
