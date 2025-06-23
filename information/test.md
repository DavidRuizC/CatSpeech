En nuestro archivo test hemos creado una nueva función que nos permite valorar como de bien es capaz de traducir nuestro modelo. Esta función se llama **test_model_end_to_end** y toma como entrada los parametros guardados de nuestro modelo y devuelve la traducción obtenida.

La función realiza el siguiente procedimiento:
- Define un modelo con una estructura como la nuestra, la cual se puede ver en [[arquitectures]]
- Le asigna los pesos encontrados durante el entrenamiento
- Procesa el audio de entrada para poder pasarlo a través de nuestro modelo. Para ello además de normalizar y eliminar los silencios lo pasamos a través del tokenizer empleado, en nuestro caso un Wav2Vec
- Predecimos empleando el modelo con los pesos hayados previamente
- Aplicamos el decoder del propio tokenizer para encontrar en forma de texto el resultado de nuestro modelo

Posteriormente se ha implementado una funcion de test que hace uso del `test_loader` para evaluar nuestro conjunto.

Actualmente el modelo no funciona de manera idónea por lo que los resultados de este test son secuencias de tokens que no forman frases con sentido y que por tanto no nos permiten transcribir el audio. Queda pendiente mejorar la arquitectura y entrenar durante mas epochs para obtener unos mejores resultados.
