# Image Classifier

## Sinopsis
Este proyecto implementa una herramienta escrita en *Python 3* con la finalidad de detectar entidades presentes en imágenes.
Para cumplir con este propósito, se emplearon dos enfoques diferentes:
* Bag of words:
  - Se apoya en el uso de algoritmos para extraer "features" de una imagen (SIFT, SURF, BRIEF, ORB, FAST).
  - Durante la fase de entrenamiento se extraen los features de todas las imágenes del conjunto de entrenamiento. Luego se aplica un algoritmo de clustering sobre estos, con el objetivo de identificar conjuntos de features que describen partes significativas de las categorías de entidades.
  - Se construye un histograma para cada imagen del conjunto de entrenamiento. Por cada cluster identificado en la fase previa se calcula la cantidad de features de este que aparecen en la imagen.
  - Se entrena un clasificador SVM a partir de los histogramas.
  - Para clasificar una imagen, se calcula su histograma (del mismo modo que durante la fase de entrenamiento) y este se clasifica empleando el clasificador previamente entrenado.
* Convolutional Neural Networks (CNNs):
  - Se empleó el framework "keras" sobre "tensorflow", que dispone de varias redes neuronales reconocidas ya implementadas y entrenadas. Asimismo, se definió una red neuronal convolucional "sencilla" con la finalidad de establecer comparaciones.
  - (__entrenadas__) MobileNet, InceptionResNetV2, InceptionV3, Xception. Se trata de CNNs previamente entrenadas con un dataset de 1000 categorías de ImageNet.
  - (__ajustables__) MobileNet, InceptionV3, Xception. Han sido entrenadas en el dataset antes mencionado, pero este conocimiento puede ser transferido a otros conjuntos de entrenamiento (especialmente con objetos semejantes pero nombres diferentes a los de ImageNet, por ejemplo: airplane - warplane)
  - (__entrenable__) SimpleCNN. Red neuronal convolucional de complejidad baja.

## Entrada
Antes de ejecutar el programa es necesario disponer de los módulos que aparecen nombrados en el archivo "requirements.txt"
El programa se ejecuta mediante el comando:

```bash
python classifier.py <config>.json
```

`<config>.json`: fichero en formato JSON con las especificaciones de las configuraciones con las que desea ejecutarse el programa. Un ejemplo de configuración sería:
```json
[
  {
    "dataset": "testing/tiny",
    "classifiers": [
      {
        "type": "classic",
        "features": "sift",
        "model": "bag of words"
      },
      {
        "type": "cnn",
        "trained": false,
        "model": "simple cnn"
      },
      {
        "type": "cnn",
        "trained": false,
        "model": "mobilenet"
      }
    ]
  }
]
```
En primer lugar vemos una lista, se trata de la lista de configuraciones que se desean probar. Cada configuracion consiste en un diccionario con dos llaves:
* `dataset`: Dirección del dataset que desea emplearse.
* `classifiers`: Lista de especificaciones de los modelos con que será probado el dataset en cuestión.
En particular los modelos se especifican mediante una llave `type`, que siempre es `"classic"` para el modelo "bag of words" o `"cnn"` para las CNNs.
En el caso de un modelo de tipo `"classic"`, además se deberán especificar el nombre del modelo mediante la llave `"model"` (solo disponible `"bag of words"`) y el algoritmo de extracción de features que desea emplearse (`"sift"`, `"surf"`, `"brief"`, `"orb"` o `"fast"`).
Para las CNNs, se especifica además del tipo, una llave `"trained"` con valor booleano, que indica si el modelo a emplear será entrenado (o transferido el conocimiento según sea el caso), o se utilizará uno entrenado previamente. Además se incluirá una llave `"model"`, indicando el nombre del modelo de CNN a emplear.

## Datasets

Los datasets deberán ser carpetas con 3 subcarpetas: "train", "validation", "test", cada una de ellas dividida en carpetas para cada categoría, que contendrán las imágenes correspondientes. Para imágenes que contengan más de una categoría se propone incluirlas en las carpetas de cada una de estas.

## Salida
El programa reporta la duración del período de entrenamiento (si corresponde), así como información relativa a la validación de este. Al finalizar el entrenamiento, se imprime el resultado para cada caso de prueba, así como la efectividad del algoritmo hasta ese momento.
