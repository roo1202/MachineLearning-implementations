import numpy as np
from collections import Counter

def _euclidean_distance(x1, x2):
        """
        Calcula la distancia euclidiana entre dos vectores n-dimensionales.
        
        Parámetros:
        -----------
        x1 : array-like
            Primer vector.
        x2 : array-like
            Segundo vector.
        
        Retorna:
        --------
        float
            La distancia euclidiana entre x1 y x2.
        """
        x1 = np.array(x1)
        x2 = np.array(x2)
        return np.sqrt(np.sum((x1 - x2) ** 2))

class KNNClassifier:
    def __init__(self,distance_method=_euclidean_distance, n_neighbors=3): 
        """
        Constructor del clasificador KNN.
        
        Parámetros:
        -----------
        n_neighbors : int
            Número de vecinos más cercanos a considerar.
        """
        self.n_neighbors = n_neighbors
        self.distance_method = distance_method
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        "Entrena" el modelo guardando los datos.
        
        Parámetros:
        -----------
        X : array-like de shape (n_muestras, n_características)
            Datos de entrenamiento.
        y : array-like de shape (n_muestras,)
            Etiquetas de entrenamiento.
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        return self

    def predict(self, X):
        """
        Predice las etiquetas para las muestras en X.
        
        Parámetros:
        -----------
        X : array-like de shape (n_muestras_prueba, n_características)
            Datos de prueba a clasificar.
        
        Retorna:
        --------
        y_pred : array-like de shape (n_muestras_prueba,)
            Etiquetas predichas.
        """
        X = np.array(X)
        y_pred = []

        for x in X:
            # Calculamos todas las distancias del punto x a cada punto de entrenamiento
            distancias = [self.distance_method(x, x_train) for x_train in self.X_train]

            # Obtenemos los índices de los n_neighbors más cercanos
            idx_vecinos = np.argsort(distancias)[:self.n_neighbors]

            # Obtenemos las etiquetas de estos vecinos
            etiquetas_vecinos = self.y_train[idx_vecinos]

            # La etiqueta predicha es la más común entre los vecinos
            etiqueta = Counter(etiquetas_vecinos).most_common(1)[0][0]
            y_pred.append(etiqueta)

        return np.array(y_pred)

