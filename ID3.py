import math
from collections import Counter

class DecisionTreeClassifier:
    def __init__(self):
        self.tree = None
        self.atributos = None

    def fit(self, examples, atributos):
        """
        Entrena el árbol de decisión con el algoritmo ID3 dado un conjunto de ejemplos y sus atributos.
        """
        self.atributos = atributos[:]
        self.tree = self._id3(examples, self.atributos)

    def predict(self, ejemplo):
        """
        Predice la clase de un ejemplo usando el árbol de decisión entrenado.
        """
        return self._predecir(ejemplo, self.tree)

    def _entropia(self, examples):
        """
        Calcula la entropía del conjunto de ejemplos.
        """
        total = len(examples)
        if total == 0:
            return 0

        counter = Counter(e['clase'] for e in examples)
        entropy = 0
        for count in counter.values():
            p = count / total
            entropy -= p * math.log2(p)
        return entropy

    def _ganancia_de_informacion(self, examples, atributo):
        """
        Calcula la ganancia de información al dividir el conjunto de ejemplos por un atributo dado.
        """
        # Entropía original
        E = self._entropia(examples)
        total = len(examples)

        # Valores posibles del atributo
        valores = set(e[atributo] for e in examples)

        # Calcular entropía ponderada de las particiones
        entropia_ponderada = 0
        for v in valores:
            subset = [e for e in examples if e[atributo] == v]
            p = len(subset) / total
            entropia_ponderada += p * self._entropia(subset)

        # Ganancia de información
        return E - entropia_ponderada

    def _mejor_atributo(self, examples, atributos):
        """
        Selecciona el atributo con mayor ganancia de información.
        """
        best_attr = None
        best_gain = -float('inf')
        for attr in atributos:
            gain = self._ganancia_de_informacion(examples, attr)
            if gain > best_gain:
                best_gain = gain
                best_attr = attr
        return best_attr

    def _id3(self, examples, atributos):
        """
        Construye el árbol de decisión usando el algoritmo ID3 de manera recursiva.
        Retorna un árbol representado como un diccionario anidado.
        
        Caso base:
        - Si todos los ejemplos tienen la misma clase, retornar esa clase.
        - Si no hay atributos disponibles, retornar la clase mayoritaria.
        
        Caso recursivo:
        - Seleccionar el mejor atributo para dividir.
        - Crear un nodo para ese atributo.
        - Para cada valor del atributo, crear una rama y recursivamente construir el subárbol.
        """
        # Si todos los ejemplos son de la misma clase, devolver esa clase
        clases = [e['clase'] for e in examples]
        if len(set(clases)) == 1:
            return clases[0]

        # Si no hay más atributos, retornar la clase mayoritaria
        if not atributos:
            return Counter(clases).most_common(1)[0][0]

        # Seleccionar el mejor atributo
        attr = self._mejor_atributo(examples, atributos)

        # Crear nodo
        tree = {attr: {}}

        # Particionar ejemplos por los valores del atributo seleccionado
        valores = set(e[attr] for e in examples)
        for v in valores:
            subset = [e for e in examples if e[attr] == v]
            # Llamada recursiva eliminando el atributo usado
            sub_atributos = [a for a in atributos if a != attr]
            subtree = self._id3(subset, sub_atributos)
            tree[attr][v] = subtree

        return tree

    def _clase_mayoritaria(self, arbol):
        # Devuelve la clase mayoritaria de un subárbol
        if not isinstance(arbol, dict):
            return arbol
        mayoritarias = []
        for val in arbol.values():
            mayoritarias.append(self._clase_mayoritaria(val))
        return Counter(mayoritarias).most_common(1)[0][0]

    def _predecir(self, ejemplo, arbol):
        # Predicción con el árbol entrenado
        if not isinstance(arbol, dict):
            return arbol
        # Tomar el atributo raíz
        atributo = list(arbol.keys())[0]
        valor = ejemplo.get(atributo, None)
        ramas = arbol[atributo]
        if valor in ramas:
            return self._predecir(ejemplo, ramas[valor])
        else:
            # Valor no visto en entrenamiento, predecir mayoritaria
            clases = []
            for val in ramas.values():
                if isinstance(val, dict):
                    # Extraer hoja mayoritaria
                    clases.append(self._clase_mayoritaria(val))
                else:
                    clases.append(val)
            # Retorna la clase mayoritaria de las ramas
            return Counter(clases).most_common(1)[0][0]

