import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import random

class SVMClassifier:
    def __init__(self, C=1.0, eta=0.001, epochs=1000):
        self.C = C
        self.eta = eta
        self.epochs = epochs
        self.w = None
        self.b = None
    
    # Función para realizar el producto punto entre dos vectores
    def dot(self, u, v):
        return sum([ui * vi for ui, vi in zip(u, v)])
    
    # Función para sumar dos vectores
    def vector_add(self, u, v):
        return [ui + vi for ui, vi in zip(u, v)]
    
    # Función para restar dos vectores
    def vector_sub(self, u, v):
        return [ui - vi for ui, vi in zip(u, v)]
    
    # Función para multiplicar un vector por un escalar
    def scalar_mul(self, s, v):
        return [s * vi for vi in v]

    def fit(self, X_train, y_train):
        # Inicializamos los parámetros
        self.w = [0.0 for _ in range(len(X_train[0]))]
        self.b = 0.0

        for epoch in range(self.epochs):
            indices = list(range(len(X_train)))
            random.shuffle(indices)
            for i in indices:
                xi = X_train[i]
                yi = y_train[i]
                
                # Calculamos el margen
                margin = yi * (self.dot(self.w, xi) + self.b)
                
                # Calculamos el subgradiente
                if margin >= 1:
                    grad_w = self.w[:]  # copia de w
                    grad_b = 0.0
                else:
                    grad_w = self.vector_sub(self.w, self.scalar_mul(self.C * yi, xi))
                    grad_b = -self.C * yi
                
                # Actualizamos parámetros
                self.w = self.vector_sub(self.w, self.scalar_mul(self.eta, grad_w))
                self.b = self.b - self.eta * grad_b

    def predict(self, x):
        val = self.dot(self.w, x) + self.b
        return 1 if val >= 0 else -1

    def score(self, X_test, y_test):
        correct = 0
        for i in range(len(X_test)):
            pred = self.predict(X_test[i])
            if pred == y_test[i]:
                correct += 1
        return correct / len(X_test)


# Ejemplo de uso con datos sintéticos
if __name__ == "__main__":
    # Generamos un dataset sintético simple
    # Dos clases linealmente separables en 2D.
    # Clase +1 alrededor de (2, 2), Clase -1 alrededor de (-2, -2)
    random.seed(0)
    X = []
    y = []
    for i in range(50):
        X.append([random.gauss(2, 1), random.gauss(2, 1)])
        y.append(1)
        X.append([random.gauss(-2, 1), random.gauss(-2, 1)])
        y.append(-1)

    X = np.array(X)
    y = np.array(y)

    # Dividimos el dataset en entrenamiento (60%) y prueba (40%)
    split = int(0.6 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_test,  y_test  = X[split:], y[split:]

    # Entrenamos el modelo
    my_svm = SVMClassifier(C=1.0, eta=0.001, epochs=1000)
    my_svm.fit(X_train, y_train)

    # Evaluamos en el conjunto de prueba
    accuracy = my_svm.score(X_test, y_test)
    print("Pesos finales:", my_svm.w)
    print("Bias final:", my_svm.b)
    print("Exactitud en el conjunto de prueba:", accuracy)

################################################################################

    # Comparación con SVM de sklearn

    # Entrenar un clasificador SVM de sklearn con kernel lineal
    clf = svm.SVC(kernel='linear', C=1.0)
    clf.fit(X_train, y_train)

    # Realizar predicciones
    y_pred = clf.predict(X_test)

    # Evaluar el modelo
    acc = accuracy_score(y_test, y_pred)
    print("Exactitud del modelo SVM sklearn:", acc)
    print("Reporte de clasificación:")
    print(classification_report(y_test, y_pred))

    # Crear la malla de puntos para visualizar la frontera de decisión
    x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
    y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                        np.linspace(y_min, y_max, 200))

    # Predecir sobre el grid
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Graficar las regiones y los puntos
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
    plt.scatter(X_train[:,0], X_train[:,1], c=y_train, edgecolors='k', label="Entrenamiento")
    plt.scatter(X_test[:,0], X_test[:,1], c=y_test, edgecolors='k', alpha=0.6, label="Prueba")
    plt.legend()
    plt.title("Frontera de decisión SVM (sklearn, kernel lineal)")
    plt.show()