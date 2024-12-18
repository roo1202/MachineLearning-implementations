import numpy as np

def kmeans(points, K, max_iterations=100):
    # Generar centroides aleatorios
    centroids = points[np.random.choice(points.shape[0], K, replace=False)]
    
    for _ in range(max_iterations):
        # Asignar cada punto al centroide más cercano
        clusters = [[] for _ in range(K)]
        for point in points:
            distances = np.linalg.norm(point - centroids, axis=1)
            closest_centroid = np.argmin(distances)
            clusters[closest_centroid].append(point)
        
        # Calcular nuevos centroides
        new_centroids = np.array([np.mean(cluster, axis=0) if cluster else centroids[i] for i, cluster in enumerate(clusters)])
        
        # Verificar convergencia
        if np.all(centroids == new_centroids):
            break
        
        centroids = new_centroids
    
    return centroids, clusters


def kmedians(points, K, max_iterations=100):
    # Seleccionar medioides aleatoriamente entre los datos
    medoids = points[np.random.choice(points.shape[0], K, replace=False)]
    
    def calculate_cost(points, medoids, clusters):
        cost = 0
        for i, cluster in enumerate(clusters):
            for point in cluster:
                cost += np.linalg.norm(point - medoids[i])
        return cost
    
    for _ in range(max_iterations):
        # Asignar cada punto al medioide más cercano
        clusters = [[] for _ in range(K)]
        for point in points:
            distances = np.linalg.norm(point - medoids, axis=1)
            closest_medoid = np.argmin(distances)
            clusters[closest_medoid].append(point)
        
        # Calcular nuevos medioides
        new_medoids = np.copy(medoids)
        for i, cluster in enumerate(clusters):
            if len(cluster) == 0:
                continue
            cluster = np.array(cluster)
            medoid_costs = np.sum(np.abs(cluster[:, np.newaxis] - cluster[np.newaxis, :]), axis=2)
            new_medoids[i] = cluster[np.argmin(np.sum(medoid_costs, axis=1))]
        
        # Calcular el costo actual
        current_cost = calculate_cost(points, medoids, clusters)
        
        # Verificar convergencia
        if np.all(medoids == new_medoids):
            break
        
        # Intentar intercambiar cada punto no medioide con el medioide
        for i in range(K):
            for point in points:
                if point not in medoids:
                    temp_medoids = np.copy(medoids)
                    temp_medoids[i] = point
                    temp_clusters = [[] for _ in range(K)]
                    for p in points:
                        distances = np.linalg.norm(p - temp_medoids, axis=1)
                        closest_temp_medoid = np.argmin(distances)
                        temp_clusters[closest_temp_medoid].append(p)
                    new_cost = calculate_cost(points, temp_medoids, temp_clusters)
                    if new_cost < current_cost:
                        medoids = temp_medoids
                        clusters = temp_clusters
                        current_cost = new_cost
        
        medoids = new_medoids
    
    return medoids, clusters

def fuzzy_kmeans(points, K, max_iterations=100, m=2):
    # Inicializar la matriz de membresía aleatoriamente
    num_points = points.shape[0]
    membership_matrix = np.random.rand(num_points, K)
    membership_matrix = membership_matrix / membership_matrix.sum(axis=1, keepdims=True)
    
    for _ in range(max_iterations):
        # Calcular los centroides difusos
        centroids = np.zeros((K, points.shape[1]))
        for j in range(K):
            numerator = np.sum((membership_matrix[:, j] ** m)[:, np.newaxis] * points, axis=0)
            denominator = np.sum(membership_matrix[:, j] ** m)
            centroids[j] = numerator / denominator
        
        # Actualizar la matriz de membresía
        for i in range(num_points):
            for j in range(K):
                distances = np.linalg.norm(points[i] - centroids, axis=1)
                if distances[j] == 0:
                    membership_matrix[i, :] = 0
                    membership_matrix[i, j] = 1
                    break
                else:
                    membership_matrix[i, j] = 1 / np.sum((distances[j] / distances) ** (2 / (m - 1)))
        
        # Normalizar la matriz de membresía
        membership_matrix = membership_matrix / membership_matrix.sum(axis=1, keepdims=True)
    
    return centroids, membership_matrix
