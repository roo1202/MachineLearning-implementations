import math
from collections import defaultdict, Counter

class MultinomialNaiveBayesTextClassifier:
    def __init__(self):
        self.vocab = set()
        self.class_priors = {}
        self.word_counts = {}
        self.class_counts = {}
        self.total_words_in_class = {}
    
    def fit(self, documents, labels):
        """
        Entrena el modelo Naive Bayes Multinomial.
        
        Parámetros:
        - documents: Lista de strings, cada uno es un documento de entrenamiento.
        - labels: Lista de etiquetas de clase asociadas a cada documento.
        """
        # Inicializar contadores
        self.word_counts = defaultdict(lambda: Counter())  # {clase: Counter({palabra: conteo})}
        self.class_counts = Counter(labels)                 # frecuencia de cada clase
        total_docs = len(documents)
        
        # Construir vocabulario y contar palabras
        for doc, label in zip(documents, labels):
            words = doc.split()
            for w in words:
                self.word_counts[label][w] += 1
                self.vocab.add(w)
        
        # Calcular priors de clase: P(C) = count(C) / total_docs
        self.class_priors = {c: self.class_counts[c]/total_docs for c in self.class_counts}
        
        # Contar total de palabras por clase
        self.total_words_in_class = {c: sum(self.word_counts[c].values()) for c in self.word_counts}
        
    def predict(self, document):
        """
        Predice la clase para un documento nuevo.
        
        Usaremos:
        P(C|D) ∝ P(C)*∏_w P(w|C)^{count(w,D)}
        
        En log:
        log P(C|D) = log P(C) + ∑_{w in D} count(w,D)*log P(w|C)
        """
        words = document.split()
        
        # Calcular log-likelihood para cada clase
        class_scores = {}
        for c in self.class_priors:
            # Iniciar con el log del prior
            score = math.log(self.class_priors[c])
            
            # Para cada palabra en el documento, sumar log P(w|C)
            for w in words:
                # P(w|C) = (count(w,C) + 1) / (total_words_in_class(C) + |V|)
                # Se aplica Laplace smoothing (adicion de 1)
                count_w_c = self.word_counts[c][w] if w in self.word_counts[c] else 0
                p_w_c = (count_w_c + 1) / (self.total_words_in_class[c] + len(self.vocab))
                score += math.log(p_w_c)
            
            class_scores[c] = score
        
        # Devolver la clase con mayor score
        return max(class_scores, key=class_scores.get)


# Ejemplo de uso
# Datos de entrenamiento: varios documentos sobre deportes, tecnología y política.
documents = [
    # Deportes
    "el equipo gano el partido de futbol con un gol en el ultimo minuto",
    "la seleccion nacional de baloncesto jugara la final del campeonato",
    "el corredor gano la maraton estableciendo un nuevo record",
    "el club deportivo anuncio el fichaje de un nuevo delantero",
    "el tenista vencio a su rival en un partido intenso",
    
    # Tecnología
    "la nueva version del smartphone incluye una camara de alta resolucion",
    "la empresa de tecnologia lanzo un procesador mas rapido y eficiente",
    "el software de inteligencia artificial permitira reconocer imagenes",
    "analistas predicen que los autos electricos dominaran el mercado",
    "la conferencia de desarrolladores presento innovaciones en dispositivos moviles",
    
    # Política
    "el candidato prometio reformas en educacion y salud publicas",
    "el congreso aprobo la nueva ley de proteccion ambiental",
    "el gobierno anuncio inversiones en infraestructura",
    "la oposicion critico la gestion economica del presidente",
    "el ministerio de relaciones exteriores nego las acusaciones"
]

labels = [
    "deportes", "deportes", "deportes", "deportes", "deportes",
    "tecnologia", "tecnologia", "tecnologia", "tecnologia", "tecnologia",
    "politica", "politica", "politica", "politica", "politica"
]

# Entrenamiento
clf = MultinomialNaiveBayesTextClassifier()
clf.fit(documents, labels)

# Pruebas de predicción con varios ejemplos
test_docs = [
    "el futbolista anoto dos goles en el campeonato nacional",
    "la empresa anuncio un nuevo dispositivo portatil con pantalla flexible",
    "el ministro hablo sobre las nuevas politicas economicas",
    "el club deportivo busca un entrenador con experiencia en liga europea",
    "analistas anticipan nuevos avances en inteligencia artificial y computacion cuantica",
    "la oposicion planea una rueda de prensa para denunciar corrupcion",
    "la ciclista alcanzo la meta y establecio record mundial",
    "la startup presento un software de analisis de datos avanzado",
    "el presidente discutio las relaciones internacionales con diplomatas",
    "el delantero firmo un contrato millonario con el equipo local"
]

# Predecir y mostrar resultados
for doc in test_docs:
    prediccion = clf.predict(doc)
    print("Documento:", doc)
    print("Predicción:", prediccion)
    print("----")
