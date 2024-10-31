import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('medical_examination.csv')

# 2
df['overweight'] = np.where(df['weight'] / (df['height'] * 0.01)**2 >= 25, 1, 0)

# 3
df['cholesterol'] = np.where(df['cholesterol'] == 1, 0, 1)
df['gluc'] = np.where(df['gluc'] == 1, 0, 1)



# 4
def draw_cat_plot():
    categorical_vars = sorted(['cholesterol','gluc','smoke','alco','active','overweight'])
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6), sharey=True)
    # Boucle pour chaque valeur de cardio
    for i, cardio_value in enumerate([0, 1]):
    # Filtrer le DataFrame pour la valeur de cardio
        filtered_df = df[df['cardio'] == cardio_value]

    # Compter les occurrences pour chaque variable catégorielle
        counts = filtered_df[categorical_vars].apply(pd.Series.value_counts).fillna(0).T

    # Tracer les barres
        counts.plot(kind='bar', ax=axes[i], width=0.8, alpha=0.7, rot=0)
        axes[i].set_title(f'cardio = {cardio_value}')
        axes[i].set_xlabel('variable')
        axes[i].set_ylabel('total')
        axes[1].legend(title='value', bbox_to_anchor=(1, 0.5))

    # 5
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=categorical_vars, var_name='variable', value_name='value')

    # 6
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')

    cat_plot = sns.catplot(x='variable', y='total', hue='value', col='cardio', data=df_cat, kind='bar', height=6, aspect=1)

# Display the plot
    plt.show()

    # 7



    # 8



    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    df_heat = None

    # 12
    corr = None

    # 13
    mask = None



    # 14
    fig, ax = None

    # 15



    # 16
    fig.savefig('heatmap.png')
    return fig
