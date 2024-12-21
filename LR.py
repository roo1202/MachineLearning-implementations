
import numpy as np

class LogisticRegression:
    """
    Implementación básica de Regresión Logística con descenso de gradiente.
    """

    def __init__(self, learning_rate=0.01, n_iterations=1000):
        """        
        :param learning_rate: Tasa de aprendizaje usada en el descenso de gradiente.
        :param n_iterations: Número de iteraciones para realizar las actualizaciones.
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []

    def _sigmoid(self, z):
        """
        Función de activación sigmoide.
        :param z: Producto escalar (X·w + b).
        :return: Valor entre 0 y 1 correspondiente a la probabilidad.
        """
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        """
        Entrena el modelo ajustando los pesos y el sesgo (bias) usando descenso de gradiente.

        :param X: Matriz de características de entrenamiento, de dimensión (n_muestras, n_features).
        :param y: Vector/array de etiquetas reales, de dimensión (n_muestras, ).
        """
        # Aseguramos que y sea un vector columna
        y = y.reshape(-1, )
        
        n_samples, n_features = X.shape

        # Inicialización de parámetros
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        # Iteraciones de descenso de gradiente
        for i in range(self.n_iterations):
            # Modelo lineal: z = X·w + b
            z = np.dot(X, self.weights) + self.bias

            # Aplicamos función sigmoide
            y_hat = self._sigmoid(z)

            # Cálculo de la función de costo (Binary Cross-Entropy)
            cost = -(1 / n_samples) * np.sum(y * np.log(y_hat + 1e-8) + (1 - y) * np.log(1 - y_hat + 1e-8))

            # Gradientes
            dw = (1 / n_samples) * np.dot(X.T, (y_hat - y))
            db = (1 / n_samples) * np.sum(y_hat - y)

            # Actualizar parámetros
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Almacenamos el costo en cada iteración
            self.cost_history.append(cost)

    def predict_proba(self, X):
        """
        Predice probabilidades de la clase positiva (1).
        
        :param X: Matriz de características, de dimensión (n_muestras, n_features).
        :return: Vector de probabilidades de cada muestra.
        """
        z = np.dot(X, self.weights) + self.bias
        return self._sigmoid(z)

    def predict(self, X, threshold=0.5):
        """
        Predice las etiquetas (0 ó 1) a partir de un umbral.
        
        :param X: Matriz de características.
        :param threshold: Umbral para la clasificación binaria (por defecto 0.5).
        :return: Vector de etiquetas predichas.
        """
        proba = self.predict_proba(X)
        return np.where(proba >= threshold, 1, 0)
