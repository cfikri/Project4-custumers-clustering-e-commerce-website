import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# La fonction *valeurs_manquantes* ci-dessous permet de déterminer le nombre et le pourcentage de valeurs manquantes (à 0.1% près) de chaque features d'un dataset.

def valeurs_manquantes(DataFrame):
    effectif = DataFrame.isna().sum()
    taux = DataFrame.isna().mean().round(3)*100
    result = pd.DataFrame({'effectif' : effectif, 'taux' : taux})
    return result.loc[result.effectif !=0, :] 

# La fonction *stats* ci-dessous prend en argument un DataFrame et renvoie un tableau contenant les principaux indicateurs statistiques de ses variables (effectif, moyenne, écart-type, médiane, quartiles, min et max).

def stats(DataFrame):
    return DataFrame.describe().round(3).T

# La fonction *test_std* prend en arguments un DataFrame et un entier n, et un renvoie pour chaque variable le taux (à 0.01% près) de valeurs situées en dehors de l'intervalle [moyenne - n.ecart-type , moyenne + n.ecart-type]. Cette fonction permet donc de connaitre le taux d'outliers de chaque variable selon la méthode des sigmas.

def test_std(DataFrame,n):
    features = stats(DataFrame).index
    outliers = pd.DataFrame()
    for feature in features:
        mean = stats(DataFrame).loc[feature,'mean']
        std = stats(DataFrame).loc[feature,'std']
        condition = ((DataFrame[feature] > mean + n*std) | (DataFrame[feature] < mean - n*std))
        outliers[feature] = condition
    nbr_outliers = outliers.sum()
    taux_outliers = outliers.mean().round(4)*100
    return pd.DataFrame({'nbr_outliers' : nbr_outliers, 'taux_outliers' : taux_outliers})

# La fonction *test_interquartile* prend en argument un DataFrame et un renvoie pour chaque variable le taux (à 0.01% près) de valeurs situées en dehors de l'intervalle [median - 1.5.ecart-interquartile , median + 1.5.ecart-interquartile]. Cette fonction permet donc de connaitre le taux d'outliers de chaque variable selon la méthode interquartile.

def test_interquartile(DataFrame):
    features = stats(DataFrame).index
    outliers = pd.DataFrame()
    for feature in features:
        Q1 = stats(DataFrame).loc[feature,'25%']
        Q3 = stats(DataFrame).loc[feature,'75%']
        IQ = Q3 - Q1
        condition = ((DataFrame[feature] > Q3 + 1.5*IQ) | (DataFrame[feature] < Q1 - 1.5*IQ))
        outliers[feature] = condition
    nbr_outliers = outliers.sum()
    taux_outliers = outliers.mean().round(4)*100
    return pd.DataFrame({'nbr_outliers' : nbr_outliers, 'taux_outliers' : taux_outliers})

# Suppriment les outliers détectés avec la méthode du z-score ou la méthode interquartile.

def S_outliers_drop(DataFrame,n):
    features = stats(DataFrame).index
    result = DataFrame
    for feature in features:
        mean = stats(DataFrame).loc[feature,'mean']
        std = stats(DataFrame).loc[feature,'std']
        condition = ((DataFrame[feature] > mean + n*std) | (DataFrame[feature] < mean - n*std))
        result[feature].mask(condition == True, pd.NA, inplace = True)
    return result

def IQ_outliers_drop(DataFrame):
    features = stats(DataFrame).index
    result = DataFrame
    for feature in features:
        Q1 = stats(DataFrame).loc[feature,'25%']
        Q3 = stats(DataFrame).loc[feature,'75%']
        IQ = Q3 - Q1
        condition = ((DataFrame[feature] > Q3 + 1.5*IQ) | (DataFrame[feature] < Q1 - 1.5*IQ))
        result[feature].mask(condition == True, pd.NA, inplace = True)
    return result

# La fonction *stats_extend* ci-dessous vise à présenter sous-forme de tableau les principaux indicateurs statistiques d'une DataFrame :
# - Les indicateurs de tendance centrale : moyenne et médiane ;
# - Les indicateurs de dispersion : étendue ,écart-type, quartiles et écart-interquartile ;
# - Les indicateurs de forme : skewness (asymétrie) et kurtosis (aplatissement).

def stats_extend(DataFrame):
    result = stats(DataFrame)
    quantitatif = DataFrame.select_dtypes(include=['int', 'float'])
    result.rename(columns = {'25%':'Q1', '50%':'med', '75%':'Q3' }, inplace=True)
    del result['count']
    result['etendue'] = result['max'] - result['min']
    result['IQR'] = result['Q3'] - result['Q1']
    result['skew'] = quantitatif.skew()
    result['kurtosis'] = quantitatif.kurtosis()
    return result

# La fonction *variance* ci-dessous permet de calculer la variance d'un échantillon donné, c'est à dire la somme des carrés des écarts de ses valeurs à leur moyenne. Cette fonction prend en argument un array ou un DataFrame et renvoie un array ou une Series contenant la variance de chaque colonne du tableau.

def variance(donnees):
    return (donnees.std(ddof=0)**2)*len(donnees)

# La fonction **correlation_graph** ci-dessous affiche le cercle de correlation dans le plan factoriel choisi. Elle prend trois arguments :
# - *pca* : Il s'agit de l'ACP appliquée aux données scalées ;
# - *x_y* : Il s'agit des indices des composantes principales (plan factoriel) choisies ;
# - features : Il s'agit de la liste des noms des variables que l'on souhaite représenter.

def correlation_graph(pca, x_y,features):
    
    pcs = pca.components_
    scree = (pca.explained_variance_ratio_*100).round(1)
    
    # Extrait les indices des composantes principales retenues.
    x,y = x_y

    # Taille de l'image (en inches).
    fig, ax = plt.subplots(figsize=(12, 12))

    # Pour chacune de nos variables par un vecteur (une flèche) avec le nom de la variable à côté.
    for i in range(pcs.shape[1]):
        ax.arrow(0,0, pcs[x,i], pcs[y,i], head_width=0.04, head_length=0.06, width=0.01)
        plt.text(pcs[x,i] + 0.05, pcs[y,i] + 0.05, features[i])

    # Affichage des lignes horizontales et verticales.
    plt.plot([-1, 1], [0, 0], color='grey', ls='--')
    plt.plot([0, 0], [-1, 1], color='grey', ls='--')

    # Nom des axes, avec le pourcentage d'inertie expliqué.
    plt.xlabel(f'CP{x+1} ({scree[x]}%)')
    plt.ylabel(f'CP{y+1} ({scree[y]}%)')
    
    # Affichage du titre.
    plt.title(f'Cercle des corrélations dans le plan factoriel (CP{x+1},CP{y+1})')

    # Traçage du cercle unité.
    t = np.linspace(0, 2*np.pi, 100)
    plt.plot(np.cos(t), np.sin(t), color='red')

    # Réglage des axes et affichage de la figure.
    plt.axis('equal')
    plt.show()
