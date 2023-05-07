# <center> Redes Neuronales </center>

--------

## ¿Qué son?


Las redes neuronales son modelos computacionales inspirados en el funcionamiento del cerebro humano. Se basan en la interconexión de unidades de procesamiento llamadas "neuronas artificiales" o "nodos", que a su vez están organizadas en capas. Estas capas se comunican entre sí mediante conexiones ponderadas, y cada neurona realiza una operación matemática simple en los datos de entrada.

Las redes neuronales pueden aprender a través de la exposición a ejemplos y la retroalimentación, lo que les permite realizar tareas como clasificación, reconocimiento de patrones, predicción y procesamiento de datos. La estructura y las conexiones ponderadas de una red neuronal se ajustan durante el proceso de aprendizaje para mejorar su rendimiento en una tarea específica.

Existen diferentes tipos de redes neuronales, pero una de las más comunes es la red neuronal artificial feedforward, también conocida como perceptrón multicapa. Esta red consta de una capa de entrada, una o varias capas ocultas y una capa de salida. La información fluye desde la capa de entrada a través de las capas ocultas hasta la capa de salida, donde se produce el resultado final.

Las redes neuronales han demostrado ser muy efectivas en una amplia gama de aplicaciones, como reconocimiento de imágenes, procesamiento del lenguaje natural, conducción autónoma, pronóstico del tiempo, análisis financiero, entre otros. Su capacidad para aprender y adaptarse a partir de datos los convierte en una herramienta poderosa en el campo del aprendizaje automático y la inteligencia artificial.

-------------------

## Relación con el deep learning

El deep learning, o aprendizaje profundo, es una rama del aprendizaje automático (machine learning) que se basa en el uso de redes neuronales artificiales de múltiples capas. Las redes neuronales profundas son capaces de aprender y extraer características y representaciones de alto nivel a partir de conjuntos de datos complejos y no estructurados.

A diferencia de las redes neuronales tradicionales, que suelen tener solo una o pocas capas ocultas, las redes neuronales profundas tienen muchas capas ocultas, lo que les permite aprender y representar características más abstractas y sofisticadas. Cada capa en una red neuronal profunda procesa la información recibida de la capa anterior y la pasa a la siguiente capa, permitiendo una representación jerárquica de los datos.

El deep learning ha ganado popularidad en los últimos años debido a su capacidad para abordar problemas difíciles y lograr resultados impresionantes en diversas áreas, como reconocimiento de imágenes, procesamiento del lenguaje natural, visión por computadora, generación de contenido creativo, traducción automática, entre otros.

El entrenamiento de redes neuronales profundas puede ser computacionalmente intensivo y requiere grandes cantidades de datos de entrenamiento. Sin embargo, los avances en hardware y la disponibilidad de conjuntos de datos masivos han facilitado el uso y la aplicación del deep learning en una amplia gama de campos, impulsando el desarrollo de nuevas tecnologías y aplicaciones innovadoras.

---------------------

## Cerebro humano

El cerebro humano está compuesto por miles de millones de neuronas interconectadas. Las neuronas son las células fundamentales del sistema nervioso y son responsables de la transmisión de señales eléctricas y químicas en el cerebro. Cada neurona consta de un cuerpo celular, dendritas (que reciben señales de otras neuronas) y una única prolongación larga llamada axón (que transmite señales a otras neuronas).

La comunicación entre las neuronas se realiza a través de sinapsis, que son conexiones especializadas en las que las señales químicas, llamadas neurotransmisores, se liberan desde el axón de una neurona y se transmiten a las dendritas de las neuronas vecinas. Esta transmisión de señales entre neuronas permite el procesamiento de la información y el funcionamiento del cerebro.

En cuanto a la cantidad de neuronas en el cerebro humano, se estima que hay alrededor de 86 mil millones de neuronas en promedio. Sin embargo, esta cifra puede variar y existen estimaciones que van desde 70 mil millones hasta más de 100 mil millones de neuronas.

En cuanto a la comparación entre el cerebro humano y una computadora, hay diferencias significativas. El cerebro humano es altamente complejo y se caracteriza por su capacidad para procesar información de manera masivamente paralela, es decir, muchas operaciones se llevan a cabo simultáneamente en diferentes regiones del cerebro. Además, el cerebro puede adaptarse y aprender de manera flexible a medida que se expone a nuevos estímulos y experiencias.

Por otro lado, las computadoras actuales están basadas en arquitecturas de procesamiento secuencial, lo que significa que ejecutan instrucciones de manera secuencial una tras otra. Aunque las computadoras han avanzado mucho en términos de velocidad y capacidad de procesamiento, todavía están lejos de igualar la potencia y eficiencia del cerebro humano en tareas complejas como reconocimiento de patrones, comprensión del lenguaje natural y toma de decisiones.

Sin embargo, las redes neuronales artificiales, inspiradas en el funcionamiento del cerebro, y el desarrollo del deep learning han permitido avances significativos en el procesamiento de información y en la capacidad de las computadoras para realizar tareas que antes se consideraban exclusivas del cerebro humano. Aunque aún hay mucho por descubrir y desarrollar, las computadoras han logrado replicar ciertos procesos cognitivos y han demostrado capacidades impresionantes en áreas como el reconocimiento de imágenes y el procesamiento del lenguaje natural.

## Ruta de aprendizaje (Python)

Antes de adentrarte en la implementación de redes neuronales en Python, hay varios temas que es útil tener un conocimiento básico. Aquí hay algunos temas fundamentales:

1. Programación en Python: Es importante tener un buen dominio de la programación en Python, ya que es uno de los lenguajes más utilizados para el desarrollo de redes neuronales y aprendizaje automático. Familiarízate con la sintaxis, estructuras de control, funciones y manejo de datos en Python.

2. Fundamentos de matemáticas: Las redes neuronales implican operaciones matemáticas, por lo que es útil tener conocimientos básicos en álgebra lineal, cálculo diferencial y estadística. Conceptos como matrices, vectores, derivadas, funciones de activación y probabilidades son fundamentales para comprender el funcionamiento de las redes neuronales.

3. Aprendizaje automático (Machine Learning): Es importante tener una comprensión general de los conceptos básicos del aprendizaje automático, como los conjuntos de entrenamiento y prueba, la validación cruzada, la selección de modelos, el sobreajuste (overfitting) y la evaluación de modelos. Esto te ayudará a comprender el contexto en el que se aplican las redes neuronales y cómo se integran en el campo del aprendizaje automático.

4. Bibliotecas de Python: Familiarízate con las bibliotecas de Python ampliamente utilizadas para implementar redes neuronales, como TensorFlow, Keras y PyTorch. Estas bibliotecas proporcionan herramientas y funciones específicas para construir y entrenar redes neuronales de manera eficiente.

5. Redes neuronales básicas: Aprende sobre los conceptos fundamentales de las redes neuronales, como las neuronas artificiales, las capas ocultas, las funciones de activación, la propagación hacia adelante y hacia atrás, el descenso del gradiente y la función de pérdida. Comprender estos elementos te ayudará a construir y entrenar tus propias redes neuronales.

6. Preprocesamiento de datos: Antes de alimentar los datos a una red neuronal, es importante preprocesarlos adecuadamente. Aprende sobre técnicas de normalización, codificación de variables categóricas, manejo de valores faltantes y división de conjuntos de datos en entrenamiento y prueba.

A medida que te adentres en la implementación de redes neuronales, podrás profundizar en aspectos más avanzados, como arquitecturas de redes neuronales más complejas, regularización, optimización, técnicas de visualización y más. Pero estos temas mencionados anteriormente te proporcionarán una base sólida para comenzar a implementar redes neuronales en Python.

-------------


## Aplicaciones


1. Industria financiera:
   - Detección de fraudes: Las redes neuronales se utilizan para detectar patrones de transacciones fraudulentas y alertar a las instituciones financieras sobre posibles actividades sospechosas.
   - Pronóstico de mercado: Las redes neuronales pueden analizar datos históricos y patrones del mercado para predecir tendencias y fluctuaciones en los precios de las acciones, divisas u otros instrumentos financieros.

2. Medicina:
   - Diagnóstico médico: Las redes neuronales pueden analizar datos médicos, como imágenes de resonancia magnética (IRM) o resultados de análisis de sangre, para ayudar en el diagnóstico de enfermedades o detección temprana de condiciones médicas.
   - Descubrimiento de medicamentos: Las redes neuronales pueden acelerar el proceso de descubrimiento de medicamentos al analizar grandes bases de datos moleculares y predecir la eficacia de compuestos químicos para tratar enfermedades específicas.

3. Investigación científica:
   - Astronomía: Las redes neuronales se utilizan para analizar grandes volúmenes de datos astronómicos y ayudar en la identificación de objetos celestes, clasificación de galaxias o detección de eventos astrofísicos.
   - Biología: Las redes neuronales se aplican para el análisis de secuencias genéticas, modelado de proteínas y predicción de la estructura y función de biomoléculas.

4. Industria del transporte:
   - Conducción autónoma: Las redes neuronales son esenciales en el desarrollo de sistemas de conducción autónoma. Ayudan a interpretar datos de sensores como cámaras y lidar, permitiendo que los vehículos tomen decisiones en tiempo real basadas en el entorno y las condiciones de la carretera.

5. Industria de servicios:
   - Recomendación de productos: Las redes neuronales se utilizan para realizar recomendaciones personalizadas en plataformas de comercio electrónico o servicios de streaming, basándose en el historial de compras o visualizaciones del usuario, así como en patrones de comportamiento similares.

Estos son solo algunos ejemplos de cómo las redes neuronales se aplican en diferentes industrias. El uso de estas tecnologías sigue en crecimiento y se encuentra en constante evolución, abriendo nuevas oportunidades en diversos campos. Practicamente estamos en presencia de un algoritmo aplicado a cada aspecto de la vida humana pues cualquier cosa que pensamos las neuronas estan involucradas.

-----------------------------

## Ejemplo y diferencias 

Aquí tienes un ejemplo sencillo de implementación de una red neuronal en Python utilizando la biblioteca TensorFlow. En este caso, se creará una red neuronal para predecir la clasificación (gato o perro) de imágenes basadas en características simuladas.

```python
import numpy as np
import tensorflow as tf

# Generación de datos simulados
np.random.seed(0)
num_samples = 1000
features = np.random.randn(num_samples, 3)  # Características de las imágenes simuladas
labels = np.random.randint(2, size=num_samples)  # Etiquetas de clasificación (0 o 1)

# División de datos en conjuntos de entrenamiento y prueba
train_ratio = 0.8
train_size = int(train_ratio * num_samples)
train_features = features[:train_size]
train_labels = labels[:train_size]
test_features = features[train_size:]
test_labels = labels[train_size:]

# Creación del modelo de red neuronal
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(3,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compilación del modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenamiento del modelo
model.fit(train_features, train_labels, epochs=10, batch_size=32, verbose=1)

# Evaluación del modelo en el conjunto de prueba
test_loss, test_accuracy = model.evaluate(test_features, test_labels)
print(f'Precisión en el conjunto de prueba: {test_accuracy}')

# Predicción de nuevas imágenes
new_features = np.random.randn(5, 3)  # Nuevas características simuladas
predictions = model.predict(new_features)
print('Predicciones:')
for i in range(len(predictions)):
    print(f'Imagen {i+1}: {predictions[i][0]}')

```

En este ejemplo, se generan datos simulados para características de imágenes (3 características) y etiquetas de clasificación (0 o 1). Luego, se divide el conjunto de datos en conjuntos de entrenamiento y prueba.

El modelo de red neuronal se crea utilizando la API secuencial de TensorFlow. Consiste en una capa oculta densamente conectada con 16 neuronas y función de activación ReLU, seguida de una capa de salida con una neurona y función de activación sigmoide.

El modelo se compila especificando el optimizador, la función de pérdida y las métricas a utilizar. En este caso, se utiliza el optimizador Adam y la pérdida de entropía cruzada binaria.

A continuación, el modelo se entrena utilizando el conjunto de entrenamiento durante 10 épocas, con un tamaño de lote de 32.

Una vez entrenado el modelo, se evalúa su rendimiento en el conjunto de prueba mediante la función `evaluate`.

Finalmente, se generan características simuladas para nuevas imágenes y se utilizan para realizar predicciones con el modelo entrenado.

La principal diferencia entre las redes neuronales y otros métodos de predicción, como los modelos lineales o los árboles de decisión, radica en su capacidad para aprender representaciones complejas y no lineales de los datos. Mientras que los modelos lineales y los árboles de decisión se basan en relaciones lineales y reglas de decisión explícitas, las redes neuronales pueden aprender automáticamente características más abstractas y representaciones de alto nivel a partir de los datos, lo que les permite capturar patrones más complejos y realizar predicciones.

--------------------

En este repositorio escribo mi ruta de aprendizaje que he obtenido en estos años.

La estructura es la siguiente:

### Librerias


Para trabajar con redes neuronales en Python, hay varias bibliotecas que son ampliamente utilizadas y recomendadas. Aquí hay algunas de las principales bibliotecas que debes dominar:

1. TensorFlow: TensorFlow es una de las bibliotecas más populares para el desarrollo de redes neuronales y aprendizaje automático en Python. Proporciona una amplia gama de herramientas y funciones para la construcción y entrenamiento de redes neuronales, incluyendo API de alto nivel como Keras. TensorFlow también permite el despliegue en diferentes plataformas, como dispositivos móviles y sistemas embebidos.

2. Keras: Keras es una biblioteca de alto nivel construida sobre TensorFlow. Proporciona una interfaz sencilla y fácil de usar para crear y entrenar redes neuronales. Keras se ha convertido en una opción popular debido a su enfoque en la simplicidad y la capacidad de crear prototipos rápidamente.

3. PyTorch: PyTorch es otra biblioteca de aprendizaje automático de código abierto que se ha vuelto muy popular en los últimos años. Permite construir y entrenar redes neuronales de manera eficiente y ofrece un enfoque más dinámico y flexible en comparación con TensorFlow. PyTorch es ampliamente utilizado en la investigación y ha ganado popularidad en la comunidad de la visión por computadora.

4. scikit-learn: Aunque scikit-learn no es específicamente una biblioteca de redes neuronales, es una biblioteca de aprendizaje automático muy útil y ampliamente utilizada en Python. Proporciona implementaciones de una amplia gama de algoritmos de aprendizaje automático, incluyendo redes neuronales básicas. Scikit-learn es ideal para tareas de aprendizaje automático más tradicionales y para la construcción de modelos de referencia.

5. Theano: Theano es una biblioteca de Python de bajo nivel diseñada para la construcción de modelos y la ejecución de operaciones matemáticas en GPUs. Aunque ha sido superada en popularidad por TensorFlow y PyTorch, Theano todavía se utiliza en algunas aplicaciones y puede ser útil para aquellos que deseen trabajar a un nivel más bajo y tener un mayor control sobre la implementación de redes neuronales.

Estas son solo algunas de las bibliotecas más utilizadas para trabajar con redes neuronales en Python. Dominar estas bibliotecas te brindará una base sólida para construir y entrenar modelos de redes neuronales en diferentes dominios.

### Fundamentos matematicos 

Los fundamentos matemáticos necesarios para comprender las redes neuronales incluyen los siguientes conceptos:

1. Álgebra lineal: Las redes neuronales utilizan matrices y vectores para representar los datos y los parámetros del modelo. Es importante comprender las operaciones básicas de álgebra lineal, como la multiplicación de matrices, la suma de vectores y la resolución de sistemas de ecuaciones lineales.

2. Cálculo diferencial: El cálculo diferencial es esencial para el entrenamiento de redes neuronales mediante el algoritmo de descenso del gradiente. Es importante entender conceptos como la derivada de una función, el gradiente y la regla de la cadena.

3. Teoría de la probabilidad y estadística: Muchos modelos de redes neuronales se basan en conceptos probabilísticos y estadísticos. Es útil tener un conocimiento básico de la teoría de la probabilidad, incluyendo conceptos como distribuciones de probabilidad, variables aleatorias y esperanza matemática. Además, la estadística proporciona herramientas para el análisis de datos y la evaluación del rendimiento de los modelos.

4. Funciones de activación: Las funciones de activación son elementos clave en las redes neuronales, ya que determinan la salida de una neurona. Es importante comprender diferentes funciones de activación, como la función sigmoide, la función ReLU (Rectified Linear Unit) y la función softmax.

5. Optimización: El entrenamiento de una red neuronal implica la optimización de una función de pérdida. Es importante comprender conceptos básicos de optimización, como el descenso del gradiente, el aprendizaje por lotes (batch learning) y la actualización de los pesos del modelo.

6. Teoría de grafos: Las redes neuronales se pueden representar como grafos de nodos y conexiones. La comprensión de los conceptos básicos de teoría de grafos, como nodos, aristas y caminos, puede ayudar a entender la estructura y el funcionamiento de una red neuronal.

### Modelos reales

Hay muchos proyectos con código y visualización disponibles que pueden ser útiles para aprender sobre redes neuronales y su implementación. Aquí tienes un ejemplo de un proyecto que puedes consultar:

Proyecto: Clasificación de imágenes con redes neuronales convolucionales (CNN)

Descripción: En este proyecto, puedes aprender a construir y entrenar una red neuronal convolucional (CNN) para clasificar imágenes. Utilizarás la biblioteca TensorFlow y el conjunto de datos MNIST, que contiene imágenes de dígitos escritos a mano.

Pasos del proyecto:

1. Preparación de datos: Descarga el conjunto de datos MNIST, que incluye imágenes de dígitos escritos a mano junto con sus etiquetas de clasificación correspondientes. Explora y visualiza algunas imágenes del conjunto de datos para familiarizarte con ellos.

2. Construcción del modelo: Define y configura tu modelo de CNN utilizando capas convolucionales, capas de agrupación (pooling) y capas totalmente conectadas. Puedes usar la API Keras, que es una interfaz de alto nivel de TensorFlow, para facilitar la construcción del modelo.

3. Entrenamiento del modelo: Ajusta los hiperparámetros de tu modelo y entrena la red neuronal utilizando el conjunto de datos de entrenamiento. Observa cómo el modelo mejora su rendimiento a medida que avanza el entrenamiento.

4. Evaluación del modelo: Evalúa el rendimiento del modelo utilizando el conjunto de datos de prueba. Calcula métricas como la precisión y la matriz de confusión para evaluar la calidad de las clasificaciones realizadas por el modelo.

5. Predicción de imágenes nuevas: Utiliza el modelo entrenado para hacer predicciones sobre nuevas imágenes de dígitos escritos a mano. Visualiza las imágenes junto con las predicciones realizadas por el modelo para ver cómo se comporta en ejemplos del mundo real.

Recursos:
- Tutorial de TensorFlow: Clasificación de dígitos MNIST con CNN: https://www.tensorflow.org/tutorials/quickstart/beginner

Este proyecto te permitirá adquirir experiencia práctica en la implementación de redes neuronales convolucionales y en la clasificación de imágenes. También podrás visualizar los resultados y ver cómo se comporta el modelo en diferentes casos. Recuerda que este es solo un ejemplo y hay muchos otros proyectos disponibles que puedes explorar para aprender más sobre redes neuronales y su aplicación en diferentes dominios.

## Proyecto Personal
Deseo emprender un proyecto personal en el que implementaré redes neuronales con el objetivo de desarrollar un modelo de predicción de precios de acciones. Este modelo utilizará tanto datos históricos como información sobre situaciones externas relevantes para mejorar su capacidad predictiva. Además, se busca utilizar este modelo para optimizar mi portafolio de inversiones.

Descripción general:

Recopilación de datos: Necesitarás recopilar datos históricos de precios de acciones y también datos relacionados con situaciones externas que puedan afectar los precios de las acciones, como noticias económicas, eventos políticos, indicadores financieros, etc. Puedes obtener estos datos de fuentes públicas o suscribirte a servicios especializados de datos financieros.

Preprocesamiento de datos: Una vez que tengas los datos, deberás realizar un preprocesamiento para asegurarte de que estén en el formato adecuado para el entrenamiento del modelo. Esto puede implicar la limpieza de datos faltantes, la normalización de los valores, la selección de características relevantes, etc.

Diseño del modelo: Debes decidir qué tipo de arquitectura de red neuronal utilizar para el pronóstico de precios de acciones. Las redes neuronales recurrentes (RNN) o las redes neuronales convolucionales (CNN) pueden ser opciones adecuadas, ya que pueden aprender patrones secuenciales en los datos. También puedes considerar el uso de arquitecturas más avanzadas, como las redes neuronales de atención (Attention-based Neural Networks).

Entrenamiento del modelo: Dividirás tus datos en conjuntos de entrenamiento, validación y prueba. Utilizarás el conjunto de entrenamiento para ajustar los parámetros del modelo y la validación para ajustar los hiperparámetros y prevenir el sobreajuste. El objetivo es encontrar el modelo que mejor se ajuste a tus datos.

Evaluación del modelo: Una vez entrenado el modelo, evaluarás su rendimiento utilizando el conjunto de prueba. Calcularás métricas como el error medio absoluto (MAE) o el error cuadrático medio (MSE) para evaluar la precisión de las predicciones del modelo.

Optimización del portafolio: Una vez que tengas un modelo de predicción de precios de acciones, puedes utilizarlo para optimizar tu portafolio. Puedes combinar las predicciones del modelo con técnicas de optimización de portafolio, como la optimización de Markowitz, para encontrar la asignación óptima de tus inversiones.

Es importante tener en cuenta que predecir los precios de las acciones es un desafío complejo y que los mercados financieros son altamente impredecibles. Los modelos de predicción pueden ser útiles como herramientas complementarias en la toma de decisiones de inversión, pero siempre se debe ejercer precaución y realizar un análisis exhaustivo.
