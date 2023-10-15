# Audio to Text for Sentiment Analysis

L'objectif de ce travail est de construire un modèle capable de transcrire un audio en français en texte, puis d'utiliser ce texte comme entrée dans un modèle d'analyse de sentiment. Le modèle déterminera si le texte (obtenu à partir de l'audio) est associé à un sentiment "positif" ou "négatif".

## PARTIE 1 : Modèle ASR

Le modèle qui servira à transcrire l'audio en texte a été téléchargé depuis Hugging Face : [Whisper Large](https://huggingface.co/openai/whisper-large-v2). Il s'agit d'un modèle qui a été entraîné pendant 680 000 heures sur des données audio annotées et qui est capable de généraliser sans nécessiter un fine-tuning. Ce modèle est multilingue, ce qui signifie qu'il peut être utilisé pour effectuer des transcriptions audio en français.

## PARTIE 2 : Modèle d'Analyse de Sentiment

Dans la deuxième partie, l'objectif est de prendre le texte obtenu après la transcription de l'audio et de l'utiliser comme entrée dans un modèle d'analyse de sentiment. Pour ce faire, nous allons entraîner un modèle d'analyse de sentiment en utilisant les données disponibles sur [Kaggle](https://www.kaggle.com/datasets/djilax/allocine-french-movie-reviews) collectées par Théophile Blard. Ces données se trouvent dans le répertoire [allocine_dataset](https://github.com/TheophileBlard/french-sentiment-analysis-with-bert/tree/master/allocine_dataset).

### Description des Données

Les données se composent de trois fichiers, dont train.csv contient environ 80 % des données (soit 160 000 avis), tandis que les fichiers de validation et de test contiennent chacun 10 %. Chaque fichier comporte trois variables :

- film-url, contenant un lien vers la critique originale du film.
- review, contenant la critique du film (d'une longueur maximale de 2 000 caractères).
- polarity, une variable binaire indiquant si la critique est positive ou négative.

Pour plus de détails sur la collecte de données et l'étiquetage, vous pouvez consulter la page GitHub du projet mentionnée ci-dessus. Les étiquettes sont les suivantes : 
```
{   
    1 : "positif",
    0 : "négatif"
}
```

### Modèle d'Entraînement

Pour affiner notre modèle avec les données d'Allociné, nous avons utilisé le modèle "Camembert Base" comme modèle de base. Ce choix est justifié par le fait qu'il s'agit d'un modèle de langue de pointe pour le français basé sur l'architecture RoBERTa.

#### Analyse Exploratoire des Données

L'analyse exploratoire des données a révélé, entre autres, que la variable cible est équilibrée, avec presque autant de textes étiquetés "positif" que de textes étiquetés "négatif".

#### Tokenisation et Entraînement

L'un des avantages du transfert d'apprentissage en NLP est que les modèles de base sont fournis avec leur propre mécanisme de tokenisation. Par exemple, le modèle Camembert est pré-entraîné sur un corpus de texte français robuste, ce qui nous permet d'appliquer directement cette tokenisation à notre jeu de données sans compromettre la qualité des données.

#### Déploiement du Modèle sur Hugging Face

En raison des limites de mémoire graphique et du temps d'entraînement nécessaires, le modèle a été entraîné sur un échantillon de taille 5 000 et testé sur un échantillon de taille 2 000. Pour plus de détails sur l'entraînement du modèle, veuillez consulter le notebook "sentiment_analyse_bert_model_bi_class.ipynb".

### Démonstration

Pour effectuer une démonstration, commencez par cloner le dépôt en utilisant la commande suivante :

```
git clone https://github.com/BIGMOUSSA/audio_to_text_for_sentiment_analysis.git
```

Pour tester le modèle, utilisez le notebook `demo.ipynb` et suivez les étapes pas à pas. Assurez-vous d'avoir un fichier audio en français à disposition.

Pour la version console, vous pouvez utilisé le fichier main.py avec 
```
py main.py

```

#### Chargement de l'Audio

Fournissez le lien vers le fichier audio que vous souhaitez transcrire.

#### Inférence 1 : Transcription en Texte

La fonction `transcribe_audio` prend un lien vers le fichier audio en tant qu'argument et applique un modèle de transcription pour produire du texte en français.

#### Inférence 2 : Analyse Sentimentale du Texte

La fonction `inference` combine la fonction de transcription `transcribe_audio` et la fonction d'analyse de sentiment `analyse_sentiment_text` pour produire un résultat sous forme de dictionnaire :

```
{
    "transcription" : "J'aime le cours de deep learning",
    "sentiment" : "Positif"
}
```

### Conclusion

Ce projet vise à combiner les modèles de transcription audio et d'analyse de sentiment. Pour utiliser le modèle, il vous suffit de fournir un fichier audio en français, et le modèle vous renverra la transcription ainsi que le sentiment associé à cette transcription.