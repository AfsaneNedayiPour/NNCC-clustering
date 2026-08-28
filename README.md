# NNCC-clustering
Natural Neighbor and Circle-Based Clustering
NNCC (Natural Neighbor and Circle-Based Clustering) is a density-based clustering algorithm that automatically identifies representative cluster heads using Natural Neighbors, adaptive local radii, and a density-peak criterion. The method further employs robust cluster circles, geometric cluster merging, and iterative label propagation to obtain the final clustering structure.
**Features**
Automatic natural-neighbor identification
Adaptive local-radius estimation
Candidate representative-point selection
Density peak detection based on the Gamma criterion
Automatic cluster-head selection
Cluster expansion using Natural Neighbor relationships
Robust cluster-center estimation
Adaptive cluster-circle construction
Circle-based preliminary cluster merging
Iterative label propagation for unlabeled samples
Distance-weighted Natural Neighbor voting
Confidence- and entropy-based label assignment
Supports synthetic and real-world datasets
Provides clustering evaluation metrics:
ACC
NMI
ARI
AMI
FMI
Reports execution time and the number of detected clusters

**The NNCC algorithm consists of six main stages:**

**Natural Neighbor Search**

Construct a KD-tree for efficient nearest-neighbor search.
Identify Natural Neighbor relationships.
Estimate the natural-neighbor depth.

**Adaptive Radius Estimation**

Estimate a local radius for each sample using its nearest neighbors.
Refine the radius using Natural Neighbor information.
Select representative candidate points.

**Density Peak Selection**

Estimate local density from the adaptive radius.
Calculate the distance to higher-density candidate points.
Compute the Gamma score.
Select final cluster heads using a statistical threshold.

**Natural Neighbor Cluster Expansion**

Initialize clusters from the selected cluster heads.
Expand clusters through Natural Neighbor relationships.
Control the expansion depth using depth_cap.

**Cluster Circle Construction and Merging**

Estimate robust cluster centers using coordinate-wise medians.
Construct adaptive circles around preliminary clusters.
Evaluate geometric relationships between cluster circles.
Merge sufficiently overlapping preliminary clusters.

**Advanced Label Propagation**

Identify remaining unlabeled samples.
Propagate labels through Natural Neighbor relationships.
Use inverse-distance-weighted voting.
Apply confidence and entropy criteria to determine reliable assignments.

**Requirements**

Python 3.8+
Required packages:
numpy
scipy
scikit-learn
matplotlib
pandas
Install dependencies:
pip install numpy scipy scikit-learn matplotlib pandas
Alternatively:
pip install -r requirements.txt

**usage**
from nncc import MyClustering
model = MyClustering(
    z=0.2,
    depth_cap=4,
    beta=1.2
)
model.fit(X, y_true)
labels = model.get_labels()
n_clusters = model.get_n_clusters()
runtime = model.get_runtime()
print("Number of clusters:", n_clusters)
print("Runtime:", runtime)
The class does not require the number of clusters as an input.

**Parameters**
z: Statistical coefficient used for Gamma-based cluster-head selection
depth_cap: Maximum Natural Neighbor expansion depth
beta: Coefficient controlling the circle-based cluster merging criterion
min_overlap_points: Minimum number of overlapping points required for merging
alpha: Relative overlap threshold based on cluster size
eps_scale: Numerical tolerance used in geometric calculations

**Evaluation Metrics**

The implementation reports:

Clustering Accuracy (ACC)
Normalized Mutual Information (NMI)
Adjusted Rand Index (ARI)
Adjusted Mutual Information (AMI)
Fowlkes–Mallows Index (FMI)
Runtime
Number of detected clusters
The evaluation can be performed using:
results = evaluate_clustering(y_true, labels)
Example:
ACC: 0.9873
NMI: 0.9821
ARI: 0.9754
AMI: 0.9816
FMI: 0.9782
The clustering accuracy is calculated using the Hungarian algorithm to find the optimal correspondence between predicted clusters and ground-truth classes.
**Supported Datasets**
The implementation contains loaders for several commonly used datasets, including:

Aggregation
Parkinsons
Iris
Wine
OptDigits
Seeds
Glass
Dermatology
Segment
Yeast
Vehicle
Olivetti Faces
WDBC
Heart-Statlog
Several datasets are loaded directly from scikit-learn, while others are expected as local CSV/TXT files.

For high-dimensional Olivetti Faces data, PCA is used to reduce the feature space to 50 dimensions:
PCA(n_components=50,whiten=True,random_state=0)
**Runtime Analysis**
The implementation records the total execution time:
runtime = model.get_runtime()
Stage-level execution times are also available through:
model.stage_times
The main computational stages include:
S0: Natural Neighbor Search
S1: Radius Estimation and Initial Head Selection
S2: Density and Gamma Computation
S3: Cluster Expansion
S4: Circle Construction
S5: Circle Merging
For experimental studies, multiple runs can be performed and the mean and standard deviation can be reported.
**Visualization**
The implementation provides visualization of both clustering results and local circles.
Cluster Visualization
plt.scatter(X[:, 0],X[:, 1],c=merged_labels,cmap='tab10',s=30)
Local Circle Visualization
The implementation can draw a circle around every sample according to its estimated local radius.
The resulting visualization can help illustrate how local density and neighborhood structure influence the clustering process.
For example:
model.plot_robust_circles(X,model.get_labels_before_merge(),centers,radii,label_order)
For datasets with more than two dimensions, PCA can be used to obtain a two-dimensional representation for visualization.

**Citation**

If you use this implementation or the NNCC methodology in your research, please cite the corresponding paper:
@article{NNCC,

  title   = {Natural Neighbor and Circle-Based Clustering},

  author  = {Afsaneh NedayiPourAsl},

  journal = {Engineering Applications of Artificial intelligence},

  year    = {2026}

}
