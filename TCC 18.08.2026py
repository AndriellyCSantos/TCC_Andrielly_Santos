import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

from sklearn.cluster import KMeans
from sklearn.metrics import euclidean_distances, silhouette_score, silhouette_samples
from sklearn.decomposition import PCA

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

colunas_vbnet = [col for col in skills.columns if col.lower() == 'vb.net']
print("Colunas encontradas:", colunas_vbnet)
print("VB.Net - valores únicos:", sorted(skills['VB.Net'].unique()))
print("Vb.Net - valores únicos:", sorted(skills['Vb.Net'].unique()))

print("\n-- Distribuição de cargos --")
print(desig['Designation'].value_counts())

# 3 - TRATAMENTO DA DUPLICATA VB.Net / Vb.Net


print("\n=== 1.3 TRATAMENTO DE DUPLICATA ===")

skills['VB.Net'] = skills['Vb.Net']
skills.drop(columns=['Vb.Net'], inplace=True)

print("Corrigido: valores reais de 'Vb.Net' mantidos como 'VB.Net'")
print("VB.Net após correção:", sorted(skills['VB.Net'].unique()))

# 4 - LISTA OFICIAL DE 19 SKILLS

colunas_skills = [
    'Python', 'Machine Learning', 'Deep Learning', 'Data Analysis',
    'Asp.Net', 'Ado.Net', 'VB.Net', 'C#',
    'Java', 'Spring Boot', 'Hibernate',
    'NLP', 'CV',
    'JS', 'React', 'Node', 'Angular',
    'Dart', 'Flutter'
]

print(f"Skills para análise: {len(colunas_skills)}")

# 5 - INTEGRAÇÃO DAS BASES VIA Eid

base = pd.merge(skills, desig, on='Eid', how='inner')

print("\n=== 1.4 INTEGRAÇÃO ===")
print("Base integrada:", base.shape)

# 6 - ESTATÍSTICAS DESCRITIVAS

print("\n=== 1.5 ESTATÍSTICAS DESCRITIVAS ===")
print(base[colunas_skills].describe().round(2))

# FIGURA 1 - DISTRIBUIÇÃO DOS CARGOS

contagem = base['Designation'].value_counts()

fig, ax = plt.subplots(figsize=(10, 5))

ax.barh(
    contagem.index,
    contagem.values,
    color='white',
    edgecolor='black',
    linewidth=1.2
)

for i, valor in enumerate(contagem.values):
    ax.text(valor + 3, i, str(valor), va='center', fontsize=10, color='black')

ax.set_xlabel('Número de Funcionários', fontsize=11)
ax.set_ylabel('Cargo', fontsize=11)
ax.tick_params(axis='both', labelsize=10, colors='black')
ax.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
plt.tight_layout()
plt.show()

# FIGURA 2a - DIAGRAMA DE CAIXA — DATA SCIENCE E IA

skills_ds  = ['Python', 'Machine Learning', 'Deep Learning', 'Data Analysis', 'NLP', 'CV']

labels_ds_box = ['Python', 'Machine\nLearning', 'Deep\nLearning', 'Data\nAnalysis', 'NLP', 'CV']

ordem_ds = base[skills_ds].mean().sort_values().index

fig, ax = plt.subplots(figsize=(12, 6))

base[ordem_ds].boxplot(
    ax=ax,
    patch_artist=False,
    widths=0.5,
    medianprops=dict(color='black', linewidth=1.5),
    boxprops=dict(color='black', linewidth=1.2),
    whiskerprops=dict(color='black', linewidth=1.2),
    capprops=dict(color='black', linewidth=1.2),
    flierprops=dict(
        marker='o', markerfacecolor='black',
        markeredgecolor='black', markersize=4, alpha=0.7
    )
)

ax.set_ylabel('Nível de Proficiência', fontsize=11)
ax.set_xlabel('Habilidades Técnicas', fontsize=11)

# Reordena labels conforme ordem das médias
labels_ds_ordenados = [
    l for s in ordem_ds
    for l, sk in zip(labels_ds_box, skills_ds) if sk == s
]
ax.set_xticklabels(labels_ds_ordenados, rotation=0, fontsize=13)
ax.tick_params(axis='y', labelsize=12)
ax.axhline(y=2, color='black', linestyle='--', linewidth=1.2)
ax.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
plt.tight_layout()
plt.show()

# FIGURA 2b - DIAGRAMA DE CAIXA — DESENVOLVIMENTO E FRAMEWORKS

skills_dev_box = [
    'Asp.Net', 'Ado.Net', 'VB.Net', 'C#',
    'Java', 'Spring Boot', 'Hibernate',
    'JS', 'React', 'Node', 'Angular',
    'Dart', 'Flutter'
]

labels_dev_box = [
    'Asp.Net', 'Ado.Net', 'VB.Net', 'C#',
    'Java', 'Spring\nBoot', 'Hibernate',
    'JS', 'React', 'Node', 'Angular',
    'Dart', 'Flutter'
]

ordem_dev = base[skills_dev_box].mean().sort_values().index

fig, ax = plt.subplots(figsize=(18, 6))

base[ordem_dev].boxplot(
    ax=ax,
    patch_artist=False,
    widths=0.5,
    medianprops=dict(color='black', linewidth=1.5),
    boxprops=dict(color='black', linewidth=1.2),
    whiskerprops=dict(color='black', linewidth=1.2),
    capprops=dict(color='black', linewidth=1.2),
    flierprops=dict(
        marker='o', markerfacecolor='black',
        markeredgecolor='black', markersize=4, alpha=0.7
    )
)

ax.set_ylabel('Nível de Proficiência', fontsize=11)
ax.set_xlabel('Habilidades Técnicas', fontsize=11)

labels_dev_ordenados = [
    l for s in ordem_dev
    for l, sk in zip(labels_dev_box, skills_dev_box) if sk == s
]
ax.set_xticklabels(labels_dev_ordenados, rotation=0, fontsize=13)
ax.tick_params(axis='y', labelsize=12)
ax.axhline(y=2, color='black', linestyle='--', linewidth=1.2)
ax.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
plt.tight_layout()
plt.show()

# FIGURA 3 - HISTOGRAMAS DATA SCIENCE

skills_ds = ['Python', 'Machine Learning', 'Deep Learning', 'Data Analysis', 'NLP', 'CV']
labels_ds = ['Python', 'Machine\nLearning', 'Deep\nLearning', 'Data\nAnalysis', 'NLP', 'CV']

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

for i, col in enumerate(skills_ds):
    axes[i].hist(base[col], bins=5, color='white', edgecolor='black', linewidth=1.2)
    axes[i].set_title(labels_ds[i], fontsize=16, color='black')
    axes[i].set_xlabel('Proficiência', fontsize=14, color='black')
    axes[i].set_ylabel('Frequência', fontsize=14, color='black')
    axes[i].set_xticks([0, 1, 2, 3, 4])
    axes[i].tick_params(axis='both', labelsize=10, colors='black')
    axes[i].grid(False)
    axes[i].spines['top'].set_visible(False)
    axes[i].spines['right'].set_visible(False)
    axes[i].spines['left'].set_color('black')
    axes[i].spines['bottom'].set_color('black')
    axes[i].spines['left'].set_linewidth(1.2)
    axes[i].spines['bottom'].set_linewidth(1.2)
    axes[i].set_facecolor('white')

fig.patch.set_facecolor('white')
plt.subplots_adjust(hspace=0.50, wspace=0.35)
plt.show()

# FIGURA 4 - HISTOGRAMAS DESENVOLVIMENTO

skills_dev = [
    'Asp.Net', 'Ado.Net', 'VB.Net', 'C#', 'Java',
    'Spring Boot', 'Hibernate', 'JS', 'React',
    'Node', 'Angular', 'Dart', 'Flutter'
]

labels_dev = [
    'Asp.Net', 'Ado.Net', 'VB.Net', 'C#', 'Java',
    'Spring\nBoot', 'Hibernate', 'JS', 'React',
    'Node', 'Angular', 'Dart', 'Flutter'
]

fig, axes = plt.subplots(5, 3, figsize=(16, 18))
axes = axes.flatten()

for i, col in enumerate(skills_dev):
    axes[i].hist(base[col], bins=5, color='white', edgecolor='black', linewidth=1.2)
    axes[i].set_title(labels_dev[i], fontsize=16, color='black')
    axes[i].set_xlabel('Proficiência', fontsize=14, color='black')
    axes[i].set_ylabel('Frequência', fontsize=14, color='black')
    axes[i].set_xticks([0, 1, 2, 3, 4])
    axes[i].tick_params(axis='both', labelsize=10, colors='black')
    axes[i].grid(False)
    axes[i].spines['top'].set_visible(False)
    axes[i].spines['right'].set_visible(False)
    axes[i].spines['left'].set_color('black')
    axes[i].spines['bottom'].set_color('black')
    axes[i].spines['left'].set_linewidth(1.2)
    axes[i].spines['bottom'].set_linewidth(1.2)
    axes[i].set_facecolor('white')

for j in range(len(skills_dev), len(axes)):
    fig.delaxes(axes[j])

fig.patch.set_facecolor('white')
plt.subplots_adjust(hspace=0.65, wspace=0.35)
plt.show()

# TABELA 2 - ESTATÍSTICAS DESCRITIVAS CUSTOMIZADAS

desc_custom = pd.DataFrame({
    'Moda':          base[colunas_skills].mode().iloc[0],
    'Média':         base[colunas_skills].mean().round(2),
    'Desvio-Padrão': base[colunas_skills].std().round(2),
    'Amplitude':     (base[colunas_skills].max() - base[colunas_skills].min())
}).reset_index()

desc_custom.columns = ['Skill', 'Moda', 'Média', 'Desvio-Padrão', 'Amplitude']

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

print("\n=== TABELA 2 - ESTATÍSTICAS DESCRITIVAS ===\n")
print(desc_custom.to_string(index=False))

# TABELA 3 - MATRIZ DE CORRELAÇÃO E PRINCIPAIS CORRELAÇÕES
print("\n=== MATRIZ DE CORRELAÇÃO ===\n")

matriz_correlacao = base[colunas_skills].corr(method='pearson').round(2)

pd.set_option('display.width', 200)
print(matriz_correlacao.to_string())

corr_long = (
    matriz_correlacao
    .where(~np.eye(matriz_correlacao.shape[0], dtype=bool))
    .stack()
    .reset_index()
)

corr_long.columns = ['Skill 1', 'Skill 2', 'Correlação']

corr_long['Par'] = corr_long.apply(
    lambda x: tuple(sorted([x['Skill 1'], x['Skill 2']])), axis=1
)

corr_long = corr_long.drop_duplicates(subset='Par')
corr_long = corr_long.sort_values(by='Correlação', ascending=False)
corr_long.drop(columns='Par', inplace=True)

print("\n=== TABELA 3 - PRINCIPAIS CORRELAÇÕES ===\n")
print(corr_long.head(23).to_string(index=False))

fortes    = corr_long[corr_long['Correlação'] >= 0.70]
moderadas = corr_long[(corr_long['Correlação'] >= 0.40) & (corr_long['Correlação'] < 0.70)]
fracas    = corr_long[corr_long['Correlação'] < 0.40]

print(f"\nCorrelações fortes (>= 0.70): {len(fortes)}")
print(f"Correlações moderadas (0.40 a 0.69): {len(moderadas)}")
print(f"Correlações fracas (< 0.40): {len(fracas)}")

# FIGURA 5 - ELBOW METHOD (MÉTODO DO COTOVELO)

print("\n=== FIGURA 5 - ELBOW METHOD ===")

inercias_elbow = []
k_range = range(2, 12)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(base[colunas_skills])
    inercias_elbow.append(km.inertia_)

fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(
    list(k_range),
    inercias_elbow,
    color='black',
    linewidth=1.5,
    marker='o',
    markerfacecolor='white',
    markeredgecolor='black',
    markeredgewidth=1.5,
    markersize=7
)

ax.set_xlabel('Número de Clusters (K)', fontsize=11)
ax.set_ylabel('Inércia', fontsize=11)
ax.set_xticks(list(k_range))
ax.tick_params(axis='both', labelsize=10, colors='black')
ax.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
plt.tight_layout()
plt.show()

# FIGURA 6 - SILHOUETTE POR K (MÉTODO SILHUETA)

print("\n=== FIGURA 6 - SILHOUETTE POR K ===")

silhuetas_k = []

for k in k_range:
    km     = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(base[colunas_skills])
    silhuetas_k.append(silhouette_score(base[colunas_skills], labels))

fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(
    list(k_range),
    silhuetas_k,
    color='black',
    linewidth=1.5,
    marker='o',
    markerfacecolor='white',
    markeredgecolor='black',
    markeredgewidth=1.5,
    markersize=7
)

ax.set_xlabel('Número de Clusters (K)', fontsize=11)
ax.set_ylabel('Índice de Silhueta Médio', fontsize=11)
ax.set_xticks(list(k_range))
ax.tick_params(axis='both', labelsize=10, colors='black')
ax.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
plt.tight_layout()
plt.show()

# FIGURA 9 - COMPARAÇÃO K=5 vs K=6 (INÉRCIA + SILHUETA)

print("\n=== FIGURA 9 - COMPARAÇÃO K=5 vs K=6 ===\n")

resultados_k = {}

for k in [5, 6]:
    km     = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(base[colunas_skills])
    resultados_k[k] = {
        'modelo':    km,
        'labels':    labels,
        'inercia':   km.inertia_,
        'silhueta':  silhouette_score(base[colunas_skills], labels),
        'centroids': pd.DataFrame(km.cluster_centers_, columns=colunas_skills)
    }
    print(f"K={k} | Inércia: {resultados_k[k]['inercia']:.2f} | Silhueta: {resultados_k[k]['silhueta']:.4f}")

ks        = [5, 6]
inercias  = [resultados_k[k]['inercia']  for k in ks]
silhuetas = [resultados_k[k]['silhueta'] for k in ks]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].bar(
    ['K=5', 'K=6'], inercias,
    color='white', edgecolor='black', linewidth=1.2, width=0.4
)
for i, v in enumerate(inercias):
    axes[0].text(i, v + max(inercias) * 0.01, f'{v:.1f}', ha='center', fontsize=11)
axes[0].set_title('Inércia por número de clusters', fontsize=12)
axes[0].set_ylabel('Inércia', fontsize=11)
axes[0].grid(False)
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
axes[0].set_facecolor('white')

axes[1].bar(
    ['K=5', 'K=6'], silhuetas,
    color='white', edgecolor='black', linewidth=1.2, width=0.4
)
for i, v in enumerate(silhuetas):
    axes[1].text(i, v + 0.002, f'{v:.4f}', ha='center', fontsize=11)
axes[1].set_title('Índice de silhueta por número de clusters', fontsize=12)
axes[1].set_ylabel('Silhueta média', fontsize=11)
axes[1].set_ylim(0, max(silhuetas) * 1.2)
axes[1].grid(False)
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
axes[1].set_facecolor('white')

fig.patch.set_facecolor('white')
plt.tight_layout()
plt.show()

# DEFINIÇÃO DO K FINAL

k_final   = 5  # altere para 6 se preferir

kmeans    = resultados_k[k_final]['modelo']
clusters  = resultados_k[k_final]['labels']
centroids = resultados_k[k_final]['centroids']

base['Cluster'] = clusters

print(f"\n=== K-MEANS APLICADO (K={k_final}) ===")
print(base['Cluster'].value_counts().sort_index())

# CENTRÓIDES E DISTÂNCIAS

print("\n=== CENTRÓIDES ===")
print(centroids.round(2))

distancias = []
for i in range(len(base)):
    cluster   = clusters[i]
    distancia = euclidean_distances(
        [base[colunas_skills].iloc[i]],
        [centroids.iloc[cluster]]
    )[0][0]
    distancias.append(distancia)

base['Distancia_Centroide'] = distancias

# FIGURAS 7 e 8 - PCA 2D PARA K=5 E K=6

for k_plot in [5, 6]:
    km_plot       = resultados_k[k_plot]['modelo']
    labels_plot   = resultados_k[k_plot]['labels']
    cents_plot    = resultados_k[k_plot]['centroids']

    pca_plot      = PCA(n_components=2)
    X_pca_plot    = pca_plot.fit_transform(base[colunas_skills])
    cents_pca     = pca_plot.transform(cents_plot)
    var_exp       = pca_plot.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(10, 7))

    scatter = ax.scatter(
        X_pca_plot[:, 0],
        X_pca_plot[:, 1],
        c=labels_plot,
        cmap='tab10',
        alpha=0.7,
        s=40
    )

    ax.scatter(
        cents_pca[:, 0],
        cents_pca[:, 1],
        c='black',
        s=200,
        marker='X',
        label='Centróide',
        zorder=5
    )

    ax.set_title(f'Clusters K-Means — PCA 2D (K={k_plot})', fontsize=13)
    ax.set_xlabel(f'PC1 ({var_exp[0]*100:.1f}% da variância)', fontsize=11)
    ax.set_ylabel(f'PC2 ({var_exp[1]*100:.1f}% da variância)', fontsize=11)

    legend1 = ax.legend(
        *scatter.legend_elements(),
        title='Cluster',
        loc='upper right',
        fontsize=9
    )
    ax.add_artist(legend1)
    ax.legend(loc='upper left', fontsize=9)

    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    plt.tight_layout()
    plt.show()

# FIGURA 10a - HEATMAP PERFIL MÉDIO — DATA SCIENCE E IA

analise_clusters = base.groupby('Cluster')[colunas_skills].mean().round(2)

fig, ax = plt.subplots(figsize=(10, 5))

sns.heatmap(
    analise_clusters[skills_ds],
    ax=ax,
    annot=True,
    fmt='.2f',
    cmap='Greys',
    linewidths=0.5,
    linecolor='white',
    cbar_kws={'label': 'Proficiência média'}
)

ax.set_title('')
ax.set_xlabel('Habilidades Técnicas', fontsize=11)
ax.set_ylabel('Agrupamento', fontsize=11)
ax.set_yticks([0, 1, 2, 3, 4])
ax.set_yticklabels(['1', '2', '3', '4', '5'], rotation=0, fontsize=10)
ax.tick_params(axis='x', rotation=0, labelsize=10)
fig.patch.set_facecolor('white')
plt.tight_layout()
plt.show()

# FIGURA 10b - HEATMAP PERFIL MÉDIO — DESENVOLVIMENTO E FRAMEWORKS

fig, ax = plt.subplots(figsize=(16, 5))

sns.heatmap(
    analise_clusters[skills_dev_box],
    ax=ax,
    annot=True,
    fmt='.2f',
    cmap='Greys',
    linewidths=0.5,
    linecolor='white',
    cbar_kws={'label': 'Proficiência média'}
)

ax.set_title('')
ax.set_xlabel('Habilidades Técnicas', fontsize=11)
ax.set_ylabel('Agrupamento', fontsize=11)
ax.set_yticks([0, 1, 2, 3, 4])
ax.set_yticklabels(['1', '2', '3', '4', '5'], rotation=0, fontsize=10)
ax.tick_params(axis='x', rotation=45, labelsize=10)
fig.patch.set_facecolor('white')
plt.tight_layout()
plt.show()


# CARACTERIZAÇÃO DESCRITIVA DOS CLUSTERS

print(f"\n=== CARACTERIZAÇÃO DESCRITIVA DOS CLUSTERS (K={k_final}) ===\n")

for c in range(k_final):
    perfil     = analise_clusters.loc[c]
    top3       = perfil.nlargest(3)
    tamanho    = (base['Cluster'] == c).sum()
    percentual = tamanho / len(base) * 100
    print(f"--- Agrupamento {c+1} ({tamanho} funcionários | {percentual:.1f}% da base) ---")
    for skill, valor in top3.items():
        print(f"    • {skill}: {valor:.2f}")
    print()

# ÍNDICE DE SILHUETA DETALHADO

silhueta_media    = silhouette_score(base[colunas_skills], clusters)
silhueta_amostras = silhouette_samples(base[colunas_skills], clusters)
base['Silhueta']  = silhueta_amostras

print(f"\n=== ÍNDICE DE SILHUETA (K={k_final}) ===")
print(f"Silhueta média geral: {silhueta_media:.4f}")
print("\nSilhueta média por cluster:")
print(base.groupby('Cluster')['Silhueta'].mean().round(4).to_string())

if silhueta_media >= 0.50:
    print("\nInterpretação: estrutura FORTE (>= 0.50)")
elif silhueta_media >= 0.25:
    print("\nInterpretação: estrutura RAZOÁVEL (0.25 a 0.50)")
else:
    print("\nInterpretação: estrutura FRACA (< 0.25)")

# GAP E FRONTEIRA

gap_cluster = base.groupby('Cluster')['Distancia_Centroide'].mean().round(3)
print("\n=== GAP MÉDIO POR CLUSTER ===")
print(gap_cluster.sort_values(ascending=False))

for c in range(k_final):
    base[f'Dist_Cluster_{c}'] = euclidean_distances(
        base[colunas_skills], [centroids.iloc[c]]
    ).flatten()

colunas_dist = [f'Dist_Cluster_{c}' for c in range(k_final)]
base['Dist_Atual'] = base.apply(
    lambda r: r[f'Dist_Cluster_{int(r["Cluster"])}'], axis=1
)

def segundo_cluster(row):
    d = row[colunas_dist].copy()
    d[f'Dist_Cluster_{int(row["Cluster"])}'] = np.inf
    return d.idxmin()

base['Cluster_Destino']   = base.apply(segundo_cluster, axis=1)
base['Cluster_Destino']   = base['Cluster_Destino'].str.extract(r'(\d+)').astype(int)
base['Dist_Destino']      = base.apply(lambda r: r[f'Dist_Cluster_{int(r["Cluster_Destino"])}'], axis=1)
base['Razao_Proximidade'] = base['Dist_Destino'] / base['Dist_Atual']
base['Na_Fronteira']      = base['Razao_Proximidade'] <= 1.5

print("\n=== NA FRONTEIRA ===")
print(base['Na_Fronteira'].value_counts())

# FIGURA 11 - HEATMAP CARGOS POR CLUSTER (QUANTIDADE ABSOLUTA)

tabela_cruzada = pd.crosstab(base['Cluster'], base['Designation'])
tabela_cruzada.index = [1, 2, 3, 4, 5]
tabela_cruzada.index.name = 'Agrupamento'

print("\n=== DISTRIBUIÇÃO DE CARGOS POR CLUSTER ===\n")
print(tabela_cruzada.to_string())

fig, ax = plt.subplots(figsize=(12, 5))

sns.heatmap(
    tabela_cruzada,
    ax=ax,
    annot=True,
    fmt='d',
    cmap='Greys',
    linewidths=0.5,
    linecolor='white',
    cbar_kws={'label': 'Quantidade de funcionários'}
)

ax.set_title('')
ax.set_xlabel('Cargo (Designation)', fontsize=11)
ax.set_ylabel('Agrupamento', fontsize=11)
ax.tick_params(axis='x', rotation=45, labelsize=10)
ax.tick_params(axis='y', rotation=0, labelsize=10)
fig.patch.set_facecolor('white')
plt.tight_layout()
plt.show()

# FIGURA 12 - HEATMAP CARGOS POR CLUSTER (PERCENTUAL)

tabela_pct = pd.crosstab(
    base['Cluster'], base['Designation'], normalize='index'
).round(3) * 100

tabela_pct.index = [1, 2, 3, 4, 5]
tabela_pct.index.name = 'Agrupamento'

print("\n=== DISTRIBUIÇÃO PERCENTUAL DE CARGOS POR CLUSTER ===\n")
print(tabela_pct.to_string())

fig, ax = plt.subplots(figsize=(12, 5))

sns.heatmap(
    tabela_pct,
    ax=ax,
    annot=True,
    fmt='.1f',
    cmap='Greys',
    linewidths=0.5,
    linecolor='white',
    cbar_kws={'label': '% dentro do cluster'}
)

ax.set_title('')
ax.set_xlabel('Cargo (Designation)', fontsize=11)
ax.set_ylabel('Agrupamento', fontsize=11)
ax.tick_params(axis='x', rotation=45, labelsize=10)
ax.tick_params(axis='y', rotation=0, labelsize=10)
fig.patch.set_facecolor('white')
plt.tight_layout()
plt.show()

print("\n=== ANÁLISE CONCLUÍDA ===")
print("Figuras geradas: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12")
print("Tabelas geradas: 2, 3")
