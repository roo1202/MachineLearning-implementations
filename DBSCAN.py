import numpy as np

def dbscan(points, eps, min_samples):
    # Inicializar etiquetas de cluster (-1 significa no asignado)
    labels = np.full(points.shape[0], -1)
    cluster_id = 0
    
    def region_query(point_idx):
        distances = np.linalg.norm(points - points[point_idx], axis=1)
        return np.where(distances < eps)[0]
    
    def expand_cluster(point_idx, neighbors):
        labels[point_idx] = cluster_id
        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]
            if labels[neighbor_idx] == -1:
                labels[neighbor_idx] = cluster_id
            elif labels[neighbor_idx] == 0:
                labels[neighbor_idx] = cluster_id
                new_neighbors = region_query(neighbor_idx)
                if len(new_neighbors) >= min_samples:
                    neighbors = np.append(neighbors, new_neighbors)
            i += 1
    
    for point_idx in range(points.shape[0]):
        if labels[point_idx] != -1:
            continue
        neighbors = region_query(point_idx)
        if len(neighbors) < min_samples:
            labels[point_idx] = -1  # Marcar como ruido
        else:
            cluster_id += 1
            expand_cluster(point_idx, neighbors)
    
    return labels
