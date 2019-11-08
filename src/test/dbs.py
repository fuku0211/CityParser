from pyclustering.cluster.dbscan import dbscan
from pyclustering.cluster import cluster_visualizer
from pyclustering.utils import read_sample
from pyclustering.samples.definitions import FCPS_SAMPLES
# Sample for cluster analysis.
sample = read_sample(FCPS_SAMPLES.SAMPLE_CHAINLINK)
# Create DBSCAN algorithm.
dbscan_instance = dbscan(sample, 0.7, 3)
# Start processing by DBSCAN.
dbscan_instance.process()
# Obtain results of clustering.
clusters = dbscan_instance.get_clusters()
noise = dbscan_instance.get_noise()
# Visualize clustering results
visualizer = cluster_visualizer()
visualizer.append_clusters(clusters, sample)
visualizer.append_cluster(noise, sample, marker='x')
visualizer.show()