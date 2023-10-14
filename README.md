# audio_to_text_for_sentiment_analysis

L'objectif de ce travail est de construire un modèle capable, à partir d'un audio en français, de faire la transcription textuelle puis d'utiliser ce texte comme input dans un modèle d'analyse de sentiment. Ansi le modèle dira si le texte (obtenu à partir de l'audio) renvoi à un sentiment "positif" ou "négatif"

## PART1 : Modèle ASR

Le modèle qui servira à transcrire l'audio en texte a été télécharger sur huggingface  : https://huggingface.co/openai/whisper-large-v2. C'est un modèle qui a été entrainé pendant 680 000 heures d'audio labelisés et est capable de généraliser sans être finetuner. C'est un modèle multilinguistique et donc permet de faire l'inférence sur des enregistrements audio en français.


## PART 2 : modèle d'analyse de sentiment

Dans la deuxième partie, l'objectif est de prendre le texte obtenu après la transcription de l'audio comme input dans
le modèle d'analyse de sentiment. Pour ce faire, nous allons entrainer un modèle d'analyse de sentiment sur les données https://www.kaggle.com/datasets/djilax/allocine-french-movie-reviews. Ces données ont été crées et collectées par Théophile Blard [ https://github.com/TheophileBlard/french-sentiment-analysis-with-bert/tree/master/allocine_dataset]. 

### Description des données
Le jeu de données est divisé en trois fichiers, avec train.csv contenant environ 80 % des données (160 000 avis), et les fichiers de validation et de test contenant 10 % chacun. Chaque fichier comporte 3 variables :

    film-url contient un lien vers la critique originale du film.
    review contient la critique du film (d'une longueur maximale de 2 000 caractères).
    polarity est une variable binaire indiquant si la critique est positive ou négative.

Les détails sur la collecte de données et l'étiquetage peuvent être consultés sur la page GitHub du projet mentionné ci-dessus.
{   1 : "positif",
    0 : "negatif"
    }

### Modèle d'entrainement

Pour finetuner notre modèle avec les données de allo cinéma, nous avons utiliser "camembert-base" comme modèle de base. Le choix se justifie par le fait que c'est un modèle de langue de pointe pour le français basé sur le modèle Roberta. 

#### Analyse exploratoire des données
L'analyse exploratoire sur les données révèle entre autre que la variable cible est équilibrée, on constate qu'il y'a preque autant de texte étiquetté "positif" que de texte labellisé comme "négatif"

#### tokenisation et entrainement
L'une des avantages du transfert learning en NLP, c'est que les modèles de base viennent avec leur tokenisation. Par exemple, dans l'exemple de camembert, il y'a un corpus robuste basé sur le traitement des textes en français. Ainsi, nous pouvons directement appliquer cette tokenization à notre dataset sans perdre la qualité des données.

#### deployement du modèle dans huggingface

### Demonstration

#### chargement de l'audio

#### Inférence 1 :Transcription en texte

#### Inférence 2 : Analyse sentimental du texte


### Conclusion


