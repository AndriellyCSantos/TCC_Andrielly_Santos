import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')

# CONFIGURAÇÃO GLOBAL DOS GRÁFICOS

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.linewidth'] = 1.5

# DIRETÓRIO

os.chdir(r'C:\Users\victo\OneDrive\Área de Trabalho\TCC')

# 1 - CARREGAR OS DADOS

skills = pd.read_csv(
    'Employee_Skills_Datset.csv',
    sep=',',
    encoding='latin-1'
)

desig = pd.read_csv(
    'Employee_Designation.csv',
    sep=',',
    encoding='latin-1'
)

print("=== 1.1 CARREGAMENTO ===")
print("Skills:", skills.shape)
print("Designation:", desig.shape)

# 2 - QUALIDADE DOS DADOS

print("\n=== 1.2 QUALIDADE DOS DADOS ===")

print("\n-- Valores ausentes --")
print("Skills:", skills.isnull().sum().sum(), "nulos")
print("Designation:", desig.isnull().sum().sum(), "nulos")

print("\n-- Duplicatas --")
print("Skills:", skills.duplicated().sum(), "linhas duplicadas")
print("Designation:", desig.duplicated().sum(), "linhas duplicadas")

print("\n-- Verificação de duplicata VB.Net --")

colunas_vbnet = [
    col for col in skills.columns
    if col.lower() == 'vb.net'
]

print("Colunas encontradas:", colunas_vbnet)

print(
    "VB.Net - valores únicos:",
    sorted(skills['VB.Net'].unique())
)

print(
    "Vb.Net - valores únicos:",
    sorted(skills['Vb.Net'].unique())
)

print("\n-- Distribuição de cargos --")
print(desig['Designation'].value_counts())

# 3 - TRATAMENTO DA DUPLICATA VB.Net / Vb.Net

print("\n=== 1.3 TRATAMENTO DE DUPLICATA ===")

skills['VB.Net'] = skills['Vb.Net']

skills.drop(
    columns=['Vb.Net'],
    inplace=True
)

print(
    "Corrigido: valores reais de 'Vb.Net' mantidos como 'VB.Net'"
)

print(
    "VB.Net após correção:",
    sorted(skills['VB.Net'].unique())
)

# 4- LISTA OFICIAL DE 19 SKILLS

colunas_skills = [
    'Python',
    'Machine Learning',
    'Deep Learning',
    'Data Analysis',
    'Asp.Net',
    'Ado.Net',
    'VB.Net',
    'C#',
    'Java',
    'Spring Boot',
    'Hibernate',
    'NLP',
    'CV',
    'JS',
    'React',
    'Node',
    'Angular',
    'Dart',
    'Flutter'
]

print(f"Skills para análise: {len(colunas_skills)}")

# 5 - INTEGRAÇÃO DAS BASES VIA Eid

base = pd.merge(
    skills,
    desig,
    on='Eid',
    how='inner'
)

print("\n=== 1.4 INTEGRAÇÃO ===")

print("Base integrada:", base.shape)

print("Colunas:")
print(base.columns.tolist())

print(
    base[
        ['Eid', 'Ename', 'Designation']
        + colunas_skills[:3]
    ].head()
)

# 6 - ESTATÍSTICAS DESCRITIVAS

print("\n=== 1.5 ESTATÍSTICAS DESCRITIVAS ===")

desc = base[colunas_skills].describe().round(2)

print(desc)

# 7 - DISTRIBUIÇÃO DOS CARGOS REAIS

contagem = base['Designation'].value_counts()

fig, ax = plt.subplots(figsize=(10, 5))

ax.barh(
    contagem.index,
    contagem.values,
    color='white',
    edgecolor='black',
    linewidth=1.2
)

# Valores nas barras
for i, valor in enumerate(contagem.values):

    ax.text(
        valor + 3,
        i,
        str(valor),
        va='center',
        fontsize=10,
        color='black'
    )

ax.set_xlabel(
    'Número de Funcionários',
    fontsize=11
)

ax.set_ylabel(
    'Cargo',
    fontsize=11
)

ax.tick_params(
    axis='both',
    labelsize=10,
    colors='black'
)

# Remove grades
ax.grid(False)

# Remove bordas superiores/direitas
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Fundo branco
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

plt.tight_layout()
plt.show()

# 8 - DIAGRAMA DE CAIXA DAS HABILIDADES TÉCNICAS

ordem = base[colunas_skills].mean().sort_values().index

labels_formatados = [
    'Asp.Net',
    'Ado.Net',
    'VB.Net',
    'Dart',
    'Flutter',
    'C#',
    'Spring\nBoot',
    'Hibernate',
    'React',
    'Node',
    'Angular',
    'Java',
    'JS',
    'Machine\nLearning',
    'Deep\nLearning',
    'Data\nAnalysis',
    'NLP',
    'CV',
    'Python'
]

fig, ax = plt.subplots(figsize=(22, 8))

base[ordem].boxplot(
    ax=ax,
    patch_artist=False,
    widths=0.6,

    medianprops=dict(
        color='black',
        linewidth=1.5
    ),

    boxprops=dict(
        color='black',
        linewidth=1.2
    ),

    whiskerprops=dict(
        color='black',
        linewidth=1.2
    ),

    capprops=dict(
        color='black',
        linewidth=1.2
    ),

    flierprops=dict(
        marker='o',
        markerfacecolor='black',
        markeredgecolor='black',
        markersize=4,
        alpha=0.7
    )
)

ax.set_ylabel(
    'Nível de Proficiência',
    fontsize=11
)

ax.set_xlabel(
    'Habilidades Técnicas',
    fontsize=11
)

ax.set_xticklabels(
    labels_formatados,
    rotation=0,
    fontsize=16
)

ax.tick_params(
    axis='y',
    labelsize=16
)

# Linha média
ax.axhline(
    y=2,
    color='black',
    linestyle='--',
    linewidth=1.2
)

# Remove grades
ax.grid(False)

# Remove bordas
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Fundo branco
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

plt.subplots_adjust(bottom=0.20)

plt.tight_layout()
plt.show()

# 9 - HISTOGRAMAS DAS HABILIDADES TÉCNICAS

# FIGURA 1

skills_ds = [
    'Python',
    'Machine Learning',
    'Deep Learning',
    'Data Analysis',
    'NLP',
    'CV'
]

labels_ds = [
    'Python',
    'Machine\nLearning',
    'Deep\nLearning',
    'Data\nAnalysis',
    'NLP',
    'CV'
]

fig, axes = plt.subplots(
    2,
    3,
    figsize=(15, 9)
)

axes = axes.flatten()

for i, col in enumerate(skills_ds):

    axes[i].hist(
        base[col],
        bins=5,
        color='white',
        edgecolor='black',
        linewidth=1.2
    )

    # Título individual
    axes[i].set_title(
        labels_ds[i],
        fontsize=16,
        color='black'
    )

    # Eixo X
    axes[i].set_xlabel(
        'Proficiência',
        fontsize=14,
        color='black'
    )

    # Eixo Y
    axes[i].set_ylabel(
        'Frequência',
        fontsize=14,
        color='black'
    )

    # Escala eixo X
    axes[i].set_xticks([0, 1, 2, 3, 4])

    # Tamanho dos números dos eixos
    axes[i].tick_params(
        axis='both',
        labelsize=10,
        colors='black'
    )

    # Remove grades
    axes[i].grid(False)

    # Remove bordas superiores e direitas
    axes[i].spines['top'].set_visible(False)
    axes[i].spines['right'].set_visible(False)

    # Bordas principais
    axes[i].spines['left'].set_color('black')
    axes[i].spines['bottom'].set_color('black')

    axes[i].spines['left'].set_linewidth(1.2)
    axes[i].spines['bottom'].set_linewidth(1.2)

    # Fundo branco
    axes[i].set_facecolor('white')

fig.patch.set_facecolor('white')

# Espaçamento entre gráficos
plt.subplots_adjust(
    hspace=0.50,
    wspace=0.35
)

plt.show()

print("Histogramas de Data Science exibidos!")

# FIGURA 2

skills_dev = [
    'Asp.Net',
    'Ado.Net',
    'VB.Net',
    'C#',
    'Java',
    'Spring Boot',
    'Hibernate',
    'JS',
    'React',
    'Node',
    'Angular',
    'Dart',
    'Flutter'
]

labels_dev = [
    'Asp.Net',
    'Ado.Net',
    'VB.Net',
    'C#',
    'Java',
    'Spring\nBoot',
    'Hibernate',
    'JS',
    'React',
    'Node',
    'Angular',
    'Dart',
    'Flutter'
]

fig, axes = plt.subplots(
    5,
    3,
    figsize=(16, 18)
)

axes = axes.flatten()

for i, col in enumerate(skills_dev):

    axes[i].hist(
        base[col],
        bins=5,
        color='white',
        edgecolor='black',
        linewidth=1.2
    )

    # Título individual
    axes[i].set_title(
        labels_dev[i],
        fontsize=16,
        color='black'
    )

    # Eixo X
    axes[i].set_xlabel(
        'Proficiência',
        fontsize=14,
        color='black'
    )

    # Eixo Y
    axes[i].set_ylabel(
        'Frequência',
        fontsize=14,
        color='black'
    )

    # Escala eixo X
    axes[i].set_xticks([0, 1, 2, 3, 4])

    # Tamanho dos números dos eixos
    axes[i].tick_params(
        axis='both',
        labelsize=10,
        colors='black'
    )

    # Remove grades
    axes[i].grid(False)

    # Remove bordas superiores e direitas
    axes[i].spines['top'].set_visible(False)
    axes[i].spines['right'].set_visible(False)

    # Mantém apenas eixos principais
    axes[i].spines['left'].set_color('black')
    axes[i].spines['bottom'].set_color('black')

    axes[i].spines['left'].set_linewidth(1.2)
    axes[i].spines['bottom'].set_linewidth(1.2)

    # Fundo branco
    axes[i].set_facecolor('white')

# Remove espaços vazios
for j in range(len(skills_dev), len(axes)):
    fig.delaxes(axes[j])

fig.patch.set_facecolor('white')

# Espaçamento entre gráficos
plt.subplots_adjust(
    hspace=0.65,
    wspace=0.35
)

plt.show()

print("Histogramas de desenvolvimento exibidos!")

# 10 - TABELA DE ESTATÍSTICAS DESCRITIVAS

desc_custom = pd.DataFrame({

    'Moda':
        base[colunas_skills].mode().iloc[0],

    'Média':
        base[colunas_skills].mean().round(2),

    'Desvio-Padrão':
        base[colunas_skills].std().round(2),

    'Amplitude':
        (
            base[colunas_skills].max()
            - base[colunas_skills].min()
        )

}).reset_index()

# Renomeia colunas
desc_custom.columns = [
    'Skill',
    'Moda',
    'Média',
    'Desvio-Padrão',
    'Amplitude'
]

# Configuração para exibir tudo no console
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

print("\n=== TABELA DE ESTATÍSTICAS DESCRITIVAS ===\n")

print(desc_custom.to_string(index=False))

# 10- ANÁLISE DAS RELAÇÕES ENTRE VARIÁVEIS
# MATRIZ DE CORRELAÇÃO DAS HABILIDADES TÉCNICAS

print("\n=== 1.9 MATRIZ DE CORRELAÇÃO ===\n")

# 11- Cálculo da matriz de correlação


matriz_correlacao = (
    base[colunas_skills]
    .corr(method='pearson')
    .round(2)
)

# 12- Configuração para exibir tabela completa no console

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.max_colwidth', None)

# 13- Exibe matriz de correlação


print(matriz_correlacao.to_string())

# 14- IDENTIFICAÇÃO DAS MAIORES CORRELAÇÕES


print("\n=== MAIORES CORRELAÇÕES ENTRE SKILLS ===\n")

# 15- Transforma matriz em formato longo
corr_long = (
    matriz_correlacao
    .where(~np.eye(matriz_correlacao.shape[0], dtype=bool))
    .stack()
    .reset_index()
)

corr_long.columns = ['Skill 1', 'Skill 2', 'Correlação']

# 16- Remove duplicidades
corr_long['Par'] = corr_long.apply(
    lambda x: tuple(sorted([x['Skill 1'], x['Skill 2']])),
    axis=1
)

corr_long = corr_long.drop_duplicates(subset='Par')

# 17- Ordena da maior para menor correlação
corr_long = corr_long.sort_values(
    by='Correlação',
    ascending=False
)

# 18- Remove coluna auxiliar
corr_long.drop(columns='Par', inplace=True)

# 19 - Exibe TOP 15 correlações
print(corr_long.head(15).to_string(index=False))

# 20 - INTERPRETAÇÃO AUTOMÁTICA DAS CORRELAÇÕES

print("\n=== INTERPRETAÇÃO DAS CORRELAÇÕES ===\n")

fortes = corr_long[corr_long['Correlação'] >= 0.70]

moderadas = corr_long[
    (corr_long['Correlação'] >= 0.40) &
    (corr_long['Correlação'] < 0.70)
]

fracas = corr_long[corr_long['Correlação'] < 0.40]

print(f"Correlações fortes (>= 0.70): {len(fortes)}")
print(f"Correlações moderadas (0.40 a 0.69): {len(moderadas)}")
print(f"Correlações fracas (< 0.40): {len(fracas)}")

# 21- TABELA RESUMIDA DAS CORRELAÇÕES FORTES


print("\n=== TABELA DE CORRELAÇÕES FORTES ===\n")

if len(fortes) > 0:
    print(
        fortes[
            ['Skill 1', 'Skill 2', 'Correlação']
        ].to_string(index=False)
    )
else:
    print("Nenhuma correlação forte encontrada.")
    

# 22 - K-MEANS (ADICIONADO AO FINAL DO SEU CÓDIGO)

from sklearn.cluster import KMeans
from sklearn.metrics import euclidean_distances
from sklearn.decomposition import PCA

k_final = 5

kmeans = KMeans(
    n_clusters=k_final,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(base[colunas_skills])

base['Cluster'] = clusters

print("\n=== K-MEANS APLICADO ===")
print(base['Cluster'].value_counts().sort_index())

# 23 - CENTRÓIDES

centroids = pd.DataFrame(
    kmeans.cluster_centers_,
    columns=colunas_skills
)

print("\n=== CENTRÓIDES ===")
print(centroids.round(2))

# 23 - DISTÂNCIA AO CENTRÓIDE

distancias = []

for i in range(len(base)):
    cluster = clusters[i]

    distancia = euclidean_distances(
        [base[colunas_skills].iloc[i]],
        [centroids.iloc[cluster]]
    )[0][0]

    distancias.append(distancia)

base['Distancia_Centroide'] = distancias

print("\n=== DISTÂNCIA AO CENTRÓIDE ===")
print(base[['Cluster', 'Distancia_Centroide']].head())

# 24 - PERFIL DOS CLUSTERS

analise_clusters = base.groupby('Cluster')[colunas_skills].mean().round(2)

print("\n=== PERFIL MÉDIO POR CLUSTER ===")
print(analise_clusters)

# 25 - GAP MÉDIO POR CLUSTER

gap_cluster = base.groupby('Cluster')['Distancia_Centroide'].mean().round(3)

print("\n=== GAP MÉDIO POR CLUSTER ===")
print(gap_cluster.sort_values(ascending=False))

# 26 - FRONTEIRA (PROFISSIONAIS ENTRE CLUSTERS)

for c in range(k_final):
    base[f'Dist_Cluster_{c}'] = euclidean_distances(
        base[colunas_skills],
        [centroids.iloc[c]]
    ).flatten()

colunas_dist = [f'Dist_Cluster_{c}' for c in range(k_final)]

base['Dist_Atual'] = base.apply(
    lambda r: r[f'Dist_Cluster_{int(r["Cluster"])}'], axis=1
)

def segundo_cluster(row):
    d = row[colunas_dist].copy()
    d[f'Dist_Cluster_{int(row["Cluster"])}'] = np.inf
    return d.idxmin()

base['Cluster_Destino'] = base.apply(segundo_cluster, axis=1)
base['Cluster_Destino'] = base['Cluster_Destino'].str.extract(r'(\d+)').astype(int)

base['Dist_Destino'] = base.apply(
    lambda r: r[f'Dist_Cluster_{int(r["Cluster_Destino"])}'], axis=1
)

base['Razao_Proximidade'] = base['Dist_Destino'] / base['Dist_Atual']

base['Na_Fronteira'] = base['Razao_Proximidade'] <= 1.5

print("\n=== NA FRONTEIRA ===")
print(base['Na_Fronteira'].value_counts())


# 27 - PLOT FINAL DO AGRUPAMENTO (SEU RESULTADO)

pca = PCA(n_components=2)

X_pca = pca.fit_transform(base[colunas_skills])
centroids_pca = pca.transform(centroids)

plt.figure(figsize=(10,7))

plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=base['Cluster'],
    cmap='tab10',
    alpha=0.7
)

plt.scatter(
    centroids_pca[:,0],
    centroids_pca[:,1],
    c='black',
    s=200,
    marker='X'
)

plt.title("Clusters K-Means (PCA 2D)")
plt.xlabel("PC1")
plt.ylabel("PC2")

plt.grid(False)
plt.show()