import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
import os

from sklearn.cluster import KMeans
from sklearn.metrics import euclidean_distances, silhouette_score, silhouette_samples
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')


# ============================================================
# CONFIGURAÇÃO GLOBAL DOS GRÁFICOS
# ============================================================

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.linewidth'] = 1.5


# ============================================================
# PADRONIZAÇÃO GLOBAL — cores, nomes, casas decimais, escalas
# ============================================================
# Tudo que precisa ser IDÊNTICO em todas as figuras do TCC fica
# definido uma única vez aqui. Qualquer ajuste de estilo deve ser
# feito nesse bloco, nunca dentro de cada gráfico individualmente.

# --- Cores fixas por cluster (1 a 6), iguais em TODOS os gráficos
#     (PCA, perfil médio, cargos por cluster) e em todos os K
#     testados (K=4, K=5, K=6) ---
PALETA_CLUSTERS = {
    1: '#1f77b4',   # azul
    2: '#d62728',   # vermelho
    3: '#2ca02c',   # verde
    4: '#ff7f0e',   # laranja
    5: '#9467bd',   # roxo
    6: '#8c564b',   # marrom
}

# --- Nomes padronizados das 19 habilidades (mesma grafia e mesma
#     quebra de linha em todos os gráficos e tabelas) ---
LABELS_HABILIDADES = {
    'Python':           'Python',
    'Machine Learning': 'Machine\nLearning',
    'Deep Learning':    'Deep\nLearning',
    'Data Analysis':    'Data\nAnalysis',
    'Asp.Net':          'Asp.Net',
    'Ado.Net':          'Ado.Net',
    'VB.Net':           'VB.Net',
    'C#':               'C#',
    'Java':             'Java',
    'Spring Boot':      'Spring\nBoot',
    'Hibernate':        'Hibernate',
    'NLP':              'NLP',
    'CV':               'CV',
    'JS':               'JS',
    'React':            'React',
    'Node':             'Node',
    'Angular':          'Angular',
    'Dart':             'Dart',
    'Flutter':          'Flutter',
}


def rotulos(lista_skills):
    """Converte uma lista de nomes de habilidades para os rótulos padronizados."""
    return [LABELS_HABILIDADES[s] for s in lista_skills]


# --- Casas decimais padronizadas ---
FMT_PROFICIENCIA = '.2f'   # médias de proficiência (escala 0-4)
FMT_PERCENTUAL = '.2f'     # percentuais (% da amostra, % por cargo)
FMT_INERCIA = '.1f'        # inércia do K-Means
FMT_SILHUETA = '.4f'       # índice de silhueta (diferenças pequenas exigem mais casas)

# --- Escalas fixas para permitir comparação direta entre gráficos ---
VMIN_PROF, VMAX_PROF = 0, 4        # heatmaps de proficiência (escala original dos dados)
VMIN_PCT, VMAX_PCT = 0, 100        # heatmaps percentuais de cargos por cluster

# --- Padronização de fontes e legendas ---
FONTSIZE_TITULO = 13
FONTSIZE_EIXO = 11
FONTSIZE_TICK = 10
FONTSIZE_LEGENDA = 10
CMAP_HEATMAP = 'Greys'


def legenda_clusters(k, ax, loc='upper right', titulo='Cluster'):
    """Desenha uma legenda com as cores fixas de PALETA_CLUSTERS para os
    clusters 1..k presentes na figura."""
    handles = [
        mpatches.Patch(color=PALETA_CLUSTERS[c], label=f'Cluster {c}')
        for c in range(1, k + 1)
    ]
    return ax.legend(
        handles=handles, title=titulo, loc=loc, fontsize=FONTSIZE_LEGENDA
    )


# ============================================================
# DIRETÓRIO
# ============================================================

os.chdir(r'C:\Users\victo\OneDrive\Área de Trabalho\TCC')


# ============================================================
# 1 - CARREGAR OS DADOS
# ============================================================

skills = pd.read_csv('Employee_Skills_Datset.csv', sep=',', encoding='latin-1')
desig = pd.read_csv('Employee_Designation.csv', sep=',', encoding='latin-1')

print("=== 1.1 CARREGAMENTO ===")
print("Skills:", skills.shape)
print("Designation:", desig.shape)


# ============================================================
# 2 - QUALIDADE DOS DADOS
# ============================================================

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


# ============================================================
# 3 - TRATAMENTO DA DUPLICATA VB.Net / Vb.Net
# ============================================================

print("\n=== 1.3 TRATAMENTO DE DUPLICATA ===")
skills['VB.Net'] = skills['Vb.Net']
skills.drop(columns=['Vb.Net'], inplace=True)
print("Corrigido: valores reais de 'Vb.Net' mantidos como 'VB.Net'")
print("VB.Net após correção:", sorted(skills['VB.Net'].unique()))


# ============================================================
# 4 - LISTA OFICIAL DE 19 SKILLS
# ============================================================

colunas_skills = [
    'Python', 'Machine Learning', 'Deep Learning', 'Data Analysis',
    'Asp.Net', 'Ado.Net', 'VB.Net', 'C#',
    'Java', 'Spring Boot', 'Hibernate',
    'NLP', 'CV',
    'JS', 'React', 'Node', 'Angular',
    'Dart', 'Flutter'
]

skills_ds = ['Python', 'Machine Learning', 'Deep Learning', 'Data Analysis', 'NLP', 'CV']
skills_dev_box = [
    'Asp.Net', 'Ado.Net', 'VB.Net', 'C#', 'Java', 'Spring Boot', 'Hibernate',
    'JS', 'React', 'Node', 'Angular', 'Dart', 'Flutter'
]

print(f"Skills para análise: {len(colunas_skills)}")


# ============================================================
# 5 - INTEGRAÇÃO DAS BASES VIA Eid
# ============================================================

base = pd.merge(skills, desig, on='Eid', how='inner')
print("\n=== 1.4 INTEGRAÇÃO ===")
print("Base integrada:", base.shape)


# ============================================================
# 6 - ESTATÍSTICAS DESCRITIVAS
# ============================================================

print("\n=== 1.5 ESTATÍSTICAS DESCRITIVAS ===")
print(base[colunas_skills].describe().round(2))


# ============================================================
# FIGURA 1 - DISTRIBUIÇÃO DOS CARGOS
# ============================================================

contagem = base['Designation'].value_counts()

fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(contagem.index, contagem.values, color='white', edgecolor='black', linewidth=1.2)

for i, valor in enumerate(contagem.values):
    ax.text(valor + 3, i, str(valor), va='center', fontsize=FONTSIZE_TICK, color='black')

ax.set_title('Distribuição dos Cargos (Designations)', fontsize=FONTSIZE_TITULO)
ax.set_xlabel('Número de Funcionários', fontsize=FONTSIZE_EIXO)
ax.set_ylabel('Cargo', fontsize=FONTSIZE_EIXO)
ax.tick_params(axis='both', labelsize=FONTSIZE_TICK, colors='black')
ax.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
plt.tight_layout()
plt.show()


# ============================================================
# FIGURA 2a - DIAGRAMA DE CAIXA — DATA SCIENCE E IA
# ============================================================

ordem_ds = base[skills_ds].mean().sort_values().index

fig, ax = plt.subplots(figsize=(12, 6))

base[ordem_ds].boxplot(
    ax=ax, patch_artist=False, widths=0.5,
    medianprops=dict(color='black', linewidth=1.5),
    boxprops=dict(color='black', linewidth=1.2),
    whiskerprops=dict(color='black', linewidth=1.2),
    capprops=dict(color='black', linewidth=1.2),
    flierprops=dict(marker='o', markerfacecolor='black',
                     markeredgecolor='black', markersize=4, alpha=0.7)
)

ax.set_title('Distribuição das Habilidades Técnicas — Ciência de Dados e IA',
              fontsize=FONTSIZE_TITULO)
ax.set_ylabel('Nível de Proficiência (0 a 4)', fontsize=FONTSIZE_EIXO)
ax.set_xlabel('Habilidades Técnicas', fontsize=FONTSIZE_EIXO)
ax.set_xticklabels(rotulos(ordem_ds), rotation=0, fontsize=FONTSIZE_TICK + 3)
ax.tick_params(axis='y', labelsize=FONTSIZE_TICK + 2)
ax.set_ylim(VMIN_PROF, VMAX_PROF)
ax.axhline(y=2, color='black', linestyle='--', linewidth=1.2)
ax.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
plt.tight_layout()
plt.show()


# ============================================================
# FIGURA 2b - DIAGRAMA DE CAIXA — DESENVOLVIMENTO E FRAMEWORKS
# ============================================================

ordem_dev = base[skills_dev_box].mean().sort_values().index

fig, ax = plt.subplots(figsize=(18, 6))

base[ordem_dev].boxplot(
    ax=ax, patch_artist=False, widths=0.5,
    medianprops=dict(color='black', linewidth=1.5),
    boxprops=dict(color='black', linewidth=1.2),
    whiskerprops=dict(color='black', linewidth=1.2),
    capprops=dict(color='black', linewidth=1.2),
    flierprops=dict(marker='o', markerfacecolor='black',
                     markeredgecolor='black', markersize=4, alpha=0.7)
)

ax.set_title('Distribuição das Habilidades Técnicas — Desenvolvimento e Frameworks',
              fontsize=FONTSIZE_TITULO)
ax.set_ylabel('Nível de Proficiência (0 a 4)', fontsize=FONTSIZE_EIXO)
ax.set_xlabel('Habilidades Técnicas', fontsize=FONTSIZE_EIXO)
ax.set_xticklabels(rotulos(ordem_dev), rotation=0, fontsize=FONTSIZE_TICK + 3)
ax.tick_params(axis='y', labelsize=FONTSIZE_TICK + 2)
ax.set_ylim(VMIN_PROF, VMAX_PROF)
ax.axhline(y=2, color='black', linestyle='--', linewidth=1.2)
ax.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
plt.tight_layout()
plt.show()


# ============================================================
# FIGURA 3 - HISTOGRAMAS DATA SCIENCE
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

for i, col in enumerate(skills_ds):
    axes[i].hist(base[col], bins=5, color='white', edgecolor='black', linewidth=1.2)
    axes[i].set_title(LABELS_HABILIDADES[col].replace('\n', ' '),
                       fontsize=FONTSIZE_TITULO + 3, color='black')
    axes[i].set_xlabel('Proficiência (0 a 4)', fontsize=FONTSIZE_EIXO + 3, color='black')
    axes[i].set_ylabel('Frequência', fontsize=FONTSIZE_EIXO + 3, color='black')
    axes[i].set_xticks([0, 1, 2, 3, 4])
    axes[i].tick_params(axis='both', labelsize=FONTSIZE_TICK, colors='black')
    axes[i].grid(False)
    axes[i].spines['top'].set_visible(False)
    axes[i].spines['right'].set_visible(False)
    axes[i].spines['left'].set_color('black')
    axes[i].spines['bottom'].set_color('black')
    axes[i].spines['left'].set_linewidth(1.2)
    axes[i].spines['bottom'].set_linewidth(1.2)
    axes[i].set_facecolor('white')

fig.suptitle('Distribuição de Proficiência — Ciência de Dados e IA', fontsize=FONTSIZE_TITULO + 4)
fig.patch.set_facecolor('white')
plt.subplots_adjust(hspace=0.55, wspace=0.35)
plt.show()


# ============================================================
# FIGURA 4 - HISTOGRAMAS DESENVOLVIMENTO
# ============================================================

fig, axes = plt.subplots(5, 3, figsize=(16, 18))
axes = axes.flatten()

for i, col in enumerate(skills_dev_box):
    axes[i].hist(base[col], bins=5, color='white', edgecolor='black', linewidth=1.2)
    axes[i].set_title(LABELS_HABILIDADES[col].replace('\n', ' '),
                       fontsize=FONTSIZE_TITULO + 3, color='black')
    axes[i].set_xlabel('Proficiência (0 a 4)', fontsize=FONTSIZE_EIXO + 3, color='black')
    axes[i].set_ylabel('Frequência', fontsize=FONTSIZE_EIXO + 3, color='black')
    axes[i].set_xticks([0, 1, 2, 3, 4])
    axes[i].tick_params(axis='both', labelsize=FONTSIZE_TICK, colors='black')
    axes[i].grid(False)
    axes[i].spines['top'].set_visible(False)
    axes[i].spines['right'].set_visible(False)
    axes[i].spines['left'].set_color('black')
    axes[i].spines['bottom'].set_color('black')
    axes[i].spines['left'].set_linewidth(1.2)
    axes[i].spines['bottom'].set_linewidth(1.2)
    axes[i].set_facecolor('white')

for j in range(len(skills_dev_box), len(axes)):
    fig.delaxes(axes[j])

fig.suptitle('Distribuição de Proficiência — Desenvolvimento e Frameworks', fontsize=FONTSIZE_TITULO + 4)
fig.patch.set_facecolor('white')
plt.subplots_adjust(hspace=0.65, wspace=0.35)
plt.show()


# ============================================================
# TABELA 2 - ESTATÍSTICAS DESCRITIVAS CUSTOMIZADAS
# ============================================================

desc_custom = pd.DataFrame({
    'Moda': base[colunas_skills].mode().iloc[0],
    'Média': base[colunas_skills].mean().round(2),
    'Desvio-Padrão': base[colunas_skills].std().round(2),
    'Amplitude': (base[colunas_skills].max() - base[colunas_skills].min())
}).reset_index()

desc_custom.columns = ['Skill', 'Moda', 'Média', 'Desvio-Padrão', 'Amplitude']
desc_custom['Skill'] = desc_custom['Skill'].map(lambda s: s)  # mantém nome "cheio" na tabela

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.float_format', lambda x: f'{x:.2f}')

print("\n=== TABELA 2 - ESTATÍSTICAS DESCRITIVAS ===\n")
print(desc_custom.to_string(index=False))


# ============================================================
# TABELA 3 - MATRIZ DE CORRELAÇÃO
# ============================================================

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

fortes = corr_long[corr_long['Correlação'] >= 0.70]
moderadas = corr_long[(corr_long['Correlação'] >= 0.40) & (corr_long['Correlação'] < 0.70)]
fracas = corr_long[corr_long['Correlação'] < 0.40]

print(f"\nCorrelações fortes (>= 0,70): {len(fortes)}")
print(f"Correlações moderadas (0,40 a 0,69): {len(moderadas)}")
print(f"Correlações fracas (< 0,40): {len(fracas)}")


# ============================================================
# FIGURA 5 - ELBOW METHOD
# ============================================================

print("\n=== FIGURA 5 - ELBOW METHOD ===")

inercias_elbow = []
k_range = range(2, 12)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(base[colunas_skills])
    inercias_elbow.append(km.inertia_)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(list(k_range), inercias_elbow, color='black', linewidth=1.5,
        marker='o', markerfacecolor='white', markeredgecolor='black',
        markeredgewidth=1.5, markersize=7)
ax.set_title('Método do Cotovelo (Elbow Method)', fontsize=FONTSIZE_TITULO)
ax.set_xlabel('Número de Clusters (K)', fontsize=FONTSIZE_EIXO)
ax.set_ylabel('Inércia', fontsize=FONTSIZE_EIXO)
ax.set_xticks(list(k_range))
ax.tick_params(axis='both', labelsize=FONTSIZE_TICK, colors='black')
ax.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
plt.tight_layout()
plt.show()


# ============================================================
# FIGURA 6 - SILHOUETTE POR K
# ============================================================

print("\n=== FIGURA 6 - SILHOUETTE POR K ===")

silhuetas_k = []
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(base[colunas_skills])
    silhuetas_k.append(silhouette_score(base[colunas_skills], labels))

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(list(k_range), silhuetas_k, color='black', linewidth=1.5,
        marker='o', markerfacecolor='white', markeredgecolor='black',
        markeredgewidth=1.5, markersize=7)
ax.set_title('Índice de Silhouette por Número de Clusters', fontsize=FONTSIZE_TITULO)
ax.set_xlabel('Número de Clusters (K)', fontsize=FONTSIZE_EIXO)
ax.set_ylabel('Índice de Silhouette Médio', fontsize=FONTSIZE_EIXO)
ax.set_xticks(list(k_range))
ax.tick_params(axis='both', labelsize=FONTSIZE_TICK, colors='black')
ax.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
plt.tight_layout()
plt.show()


# ============================================================
# FIGURA 9 - COMPARAÇÃO K=4, K=5 E K=6 (inércia e silhouette)
# ============================================================

print("\n=== FIGURA 9 - COMPARAÇÃO K=4, K=5 E K=6 ===\n")

resultados_k = {}

for k in [4, 5, 6]:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(base[colunas_skills])
    resultados_k[k] = {
        'modelo': km,
        'labels': labels,
        'inercia': km.inertia_,
        'silhueta': silhouette_score(base[colunas_skills], labels),
        'centroids': pd.DataFrame(km.cluster_centers_, columns=colunas_skills)
    }
    print(
        f"K={k} | Inércia: {resultados_k[k]['inercia']:{FMT_INERCIA}} | "
        f"Silhueta: {resultados_k[k]['silhueta']:{FMT_SILHUETA}}"
    )

ks = [4, 5, 6]
inercias = [resultados_k[k]['inercia'] for k in ks]
silhuetas = [resultados_k[k]['silhueta'] for k in ks]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].bar(['K=4', 'K=5', 'K=6'], inercias, color='white',
            edgecolor='black', linewidth=1.2, width=0.5)
for i, v in enumerate(inercias):
    axes[0].text(i, v + max(inercias) * 0.01, f'{v:{FMT_INERCIA}}',
                 ha='center', fontsize=FONTSIZE_LEGENDA + 1)
axes[0].set_title('Inércia por Número de Clusters', fontsize=FONTSIZE_TITULO)
axes[0].set_ylabel('Inércia', fontsize=FONTSIZE_EIXO)
axes[0].tick_params(axis='both', labelsize=FONTSIZE_TICK)
axes[0].grid(False)
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
axes[0].set_facecolor('white')

axes[1].bar(['K=4', 'K=5', 'K=6'], silhuetas, color='white',
            edgecolor='black', linewidth=1.2, width=0.5)
for i, v in enumerate(silhuetas):
    axes[1].text(i, v + 0.002, f'{v:{FMT_SILHUETA}}',
                 ha='center', fontsize=FONTSIZE_LEGENDA + 1)
axes[1].set_title('Índice de Silhouette por Número de Clusters', fontsize=FONTSIZE_TITULO)
axes[1].set_ylabel('Silhouette Médio', fontsize=FONTSIZE_EIXO)
axes[1].tick_params(axis='both', labelsize=FONTSIZE_TICK)
axes[1].set_ylim(0, max(silhuetas) * 1.2)
axes[1].grid(False)
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
axes[1].set_facecolor('white')

fig.patch.set_facecolor('white')
plt.tight_layout()
plt.show()


# ============================================================
# DEFINIÇÃO DO K FINAL (usado nas análises detalhadas do TCC)
# ============================================================

k_final = 5
kmeans = resultados_k[k_final]['modelo']
clusters = resultados_k[k_final]['labels']
centroids = resultados_k[k_final]['centroids']
base['Cluster'] = clusters

print(f"\n=== K-MEANS APLICADO (K={k_final}) ===")
print(base['Cluster'].value_counts().sort_index())


# ============================================================
# CENTRÓIDES E DISTÂNCIAS (K final)
# ============================================================

print("\n=== CENTRÓIDES ===")
print(centroids.round(2))

distancias = []
for i in range(len(base)):
    cluster = clusters[i]
    distancia = euclidean_distances(
        [base[colunas_skills].iloc[i]], [centroids.iloc[cluster]]
    )[0][0]
    distancias.append(distancia)

base['Distancia_Centroide'] = distancias


# ============================================================
# FIGURAS 7, 8 e 8b - PCA 2D PARA K=4, K=5 E K=6
# Cores fixas por cluster (PALETA_CLUSTERS), legenda padronizada
# ============================================================

for k_plot in [4, 5, 6]:

    labels_plot = resultados_k[k_plot]['labels']
    cents_plot = resultados_k[k_plot]['centroids']

    pca_plot = PCA(n_components=2)
    X_pca_plot = pca_plot.fit_transform(base[colunas_skills])
    cents_pca = pca_plot.transform(cents_plot)
    var_exp = pca_plot.explained_variance_ratio_

    # cluster salvo como 0-based; +1 para exibir/colorir como 1..k
    cores_pontos = [PALETA_CLUSTERS[label + 1] for label in labels_plot]

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.scatter(
        X_pca_plot[:, 0], X_pca_plot[:, 1],
        c=cores_pontos, alpha=0.7, s=40
    )

    ax.scatter(
        cents_pca[:, 0], cents_pca[:, 1],
        c='black', s=200, marker='X', zorder=5
    )

    ax.set_title(f'Clusters K-Means — Projeção PCA (K={k_plot})', fontsize=FONTSIZE_TITULO)
    ax.set_xlabel(f'PC1 ({var_exp[0]*100:{FMT_PERCENTUAL}}% da variância)', fontsize=FONTSIZE_EIXO)
    ax.set_ylabel(f'PC2 ({var_exp[1]*100:{FMT_PERCENTUAL}}% da variância)', fontsize=FONTSIZE_EIXO)
    ax.tick_params(axis='both', labelsize=FONTSIZE_TICK)

    legenda_clusters(k_plot, ax, loc='upper right')

    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    plt.tight_layout()
    plt.show()


# ============================================================
# HEATMAPS DE PERFIL MÉDIO POR CLUSTER — K=4, K=5 E K=6
# Escala de cor fixa (0 a 4) e nomes de habilidades padronizados,
# para permitir comparação direta entre os mapas de calor.
# ============================================================

perfis_por_k = {}


def plot_heatmap_perfil(k, skills_subset, figsize):
    labels_k = resultados_k[k]['labels']
    base_temp = base.copy()
    base_temp['Cluster_temp'] = labels_k

    perfil_k = base_temp.groupby('Cluster_temp')[skills_subset].mean().round(2)
    perfil_k.index = range(1, k + 1)
    perfil_k.index.name = 'Cluster'
    perfil_plot = perfil_k.rename(columns=lambda s: LABELS_HABILIDADES[s])

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        perfil_plot, ax=ax, annot=True, fmt=FMT_PROFICIENCIA,
        cmap=CMAP_HEATMAP, vmin=VMIN_PROF, vmax=VMAX_PROF,
        linewidths=0.5, linecolor='white',
        cbar_kws={'label': 'Proficiência média (escala 0 a 4)'}
    )

    ax.set_title(f'Perfil Médio de Proficiência por Cluster (K={k})', fontsize=FONTSIZE_TITULO)
    ax.set_xlabel('Habilidades Técnicas', fontsize=FONTSIZE_EIXO)
    ax.set_ylabel('Cluster', fontsize=FONTSIZE_EIXO)
    ax.tick_params(axis='x', rotation=45, labelsize=FONTSIZE_TICK)
    ax.tick_params(axis='y', rotation=0, labelsize=FONTSIZE_TICK)

    fig.patch.set_facecolor('white')
    plt.tight_layout()
    plt.show()

    return perfil_k  # devolve com nomes originais (sem quebra de linha) p/ uso posterior


for k in [4, 5, 6]:
    print(f"\n=== PERFIL MÉDIO POR CLUSTER (K={k}) — CIÊNCIA DE DADOS/IA ===")
    perfil_ds_k = plot_heatmap_perfil(k, skills_ds, figsize=(10, 0.9 * k + 2))
    print(perfil_ds_k)

    print(f"\n=== PERFIL MÉDIO POR CLUSTER (K={k}) — DESENVOLVIMENTO/FRAMEWORKS ===")
    perfil_dev_k = plot_heatmap_perfil(k, skills_dev_box, figsize=(16, 0.9 * k + 2))
    print(perfil_dev_k)

    base_temp = base.copy()
    base_temp['Cluster_temp'] = resultados_k[k]['labels']
    perfis_por_k[k] = base_temp.groupby('Cluster_temp')[colunas_skills].mean().round(2)


# ============================================================
# CARACTERIZAÇÃO DESCRITIVA DOS CLUSTERS — K=4, K=5 E K=6
# ============================================================

for k in [4, 5, 6]:
    print(f"\n=== CARACTERIZAÇÃO DESCRITIVA DOS CLUSTERS (K={k}) ===\n")

    base_temp = base.copy()
    base_temp['Cluster_temp'] = resultados_k[k]['labels']
    perfil_k = perfis_por_k[k]

    for c in range(k):
        perfil_c = perfil_k.loc[c]
        top3 = perfil_c.nlargest(3)
        tamanho = (base_temp['Cluster_temp'] == c).sum()
        percentual = tamanho / len(base_temp) * 100

        print(f"--- Cluster {c+1} ({tamanho} funcionários | {percentual:{FMT_PERCENTUAL}}% da base) ---")
        for skill, valor in top3.items():
            print(f"    • {LABELS_HABILIDADES[skill].replace(chr(10), ' ')}: {valor:{FMT_PROFICIENCIA}}")
        print()


# ============================================================
# ÍNDICE DE SILHUETA DETALHADO (K final)
# ============================================================

silhueta_media = silhouette_score(base[colunas_skills], clusters)
silhueta_amostras = silhouette_samples(base[colunas_skills], clusters)
base['Silhueta'] = silhueta_amostras

print(f"\n=== ÍNDICE DE SILHUETA (K={k_final}) ===")
print(f"Silhueta média geral: {silhueta_media:{FMT_SILHUETA}}")
print("\nSilhueta média por cluster:")
print(base.groupby('Cluster')['Silhueta'].mean().round(4).to_string())

if silhueta_media >= 0.50:
    print("\nInterpretação: estrutura FORTE (>= 0,50)")
elif silhueta_media >= 0.25:
    print("\nInterpretação: estrutura RAZOÁVEL (0,25 a 0,50)")
else:
    print("\nInterpretação: estrutura FRACA (< 0,25)")


# ============================================================
# GAP E FRONTEIRA (K final)
# ============================================================

gap_cluster = base.groupby('Cluster')['Distancia_Centroide'].mean().round(3)
print("\n=== GAP MÉDIO POR CLUSTER ===")
print(gap_cluster.sort_values(ascending=False))

for c in range(k_final):
    base[f'Dist_Cluster_{c}'] = euclidean_distances(
        base[colunas_skills], [centroids.iloc[c]]
    ).flatten()

colunas_dist = [f'Dist_Cluster_{c}' for c in range(k_final)]
base['Dist_Atual'] = base.apply(lambda r: r[f'Dist_Cluster_{int(r["Cluster"])}'], axis=1)


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


# ============================================================
# HEATMAPS CARGOS POR CLUSTER — K=4, K=5 E K=6
# Percentual com escala fixa 0-100% (comparável entre K);
# absoluto com escala fixa pelo maior valor entre os três K
# (também comparável entre K=4, K=5 e K=6).
# ============================================================

# calcula o valor máximo absoluto entre as três tabelas, para
# fixar uma única escala de cor nos heatmaps em números absolutos
_max_abs_cargos = 0
for k in [4, 5, 6]:
    base_temp = base.copy()
    base_temp['Cluster_temp'] = resultados_k[k]['labels']
    tabela_tmp = pd.crosstab(base_temp['Cluster_temp'], base_temp['Designation'])
    _max_abs_cargos = max(_max_abs_cargos, tabela_tmp.values.max())


def plot_heatmap_cargos(k, normalizar=False):
    base_temp = base.copy()
    base_temp['Cluster_temp'] = resultados_k[k]['labels']

    if normalizar:
        tabela = (
            pd.crosstab(base_temp['Cluster_temp'], base_temp['Designation'], normalize='index') * 100
        ).round(2)
        fmt = FMT_PERCENTUAL
        label_cbar = '% de funcionários dentro do cluster'
        vmin, vmax = VMIN_PCT, VMAX_PCT
    else:
        tabela = pd.crosstab(base_temp['Cluster_temp'], base_temp['Designation'])
        fmt = 'd'
        label_cbar = 'Quantidade de funcionários'
        vmin, vmax = 0, _max_abs_cargos

    tabela.index = range(1, k + 1)
    tabela.index.name = 'Cluster'

    tipo = 'Percentual' if normalizar else 'Absoluta'
    print(f"\n=== DISTRIBUIÇÃO {tipo.upper()} DE CARGOS POR CLUSTER (K={k}) ===\n")
    print(tabela.to_string())

    fig, ax = plt.subplots(figsize=(12, 0.9 * k + 2))

    sns.heatmap(
        tabela, ax=ax, annot=True, fmt=fmt, cmap=CMAP_HEATMAP,
        vmin=vmin, vmax=vmax,
        linewidths=0.5, linecolor='white', cbar_kws={'label': label_cbar}
    )

    ax.set_title(f'Composição {tipo} de Clusters por Cargo (K={k})', fontsize=FONTSIZE_TITULO)
    ax.set_xlabel('Cargo (Designation)', fontsize=FONTSIZE_EIXO)
    ax.set_ylabel('Cluster', fontsize=FONTSIZE_EIXO)
    ax.tick_params(axis='x', rotation=45, labelsize=FONTSIZE_TICK)
    ax.tick_params(axis='y', rotation=0, labelsize=FONTSIZE_TICK)

    fig.patch.set_facecolor('white')
    plt.tight_layout()
    plt.show()

    return tabela


tabelas_cargos_abs = {}
tabelas_cargos_pct = {}

for k in [4, 5, 6]:
    tabelas_cargos_abs[k] = plot_heatmap_cargos(k, normalizar=False)
    tabelas_cargos_pct[k] = plot_heatmap_cargos(k, normalizar=True)


# ============================================================
# TABELA — TAMANHO, PERCENTUAL, HABILIDADES PREDOMINANTES
# E LIMITAÇÕES POR CLUSTER (pedida no e-mail 1 do professor)
# ============================================================
# A coluna "Pontos de atenção" é só um alerta automático de partida;
# revise/complete manualmente antes de colar no TCC.

def montar_tabela_resumo(k):
    base_temp = base.copy()
    base_temp['Cluster_temp'] = resultados_k[k]['labels']
    perfil_k = perfis_por_k[k]

    linhas = []
    for c in range(k):
        perfil_c = perfil_k.loc[c]
        top3 = perfil_c.nlargest(3)
        tamanho = (base_temp['Cluster_temp'] == c).sum()
        percentual = tamanho / len(base_temp) * 100
        media_geral_cluster = perfil_c.mean()

        habilidades_predominantes = '; '.join(
            f'{LABELS_HABILIDADES[skill].replace(chr(10), " ")} '
            f'({valor:{FMT_PROFICIENCIA}})'
            for skill, valor in top3.items()
        )

        alerta = []
        if perfil_c.max() < 2.5:
            alerta.append('nenhuma habilidade com média >= 2,5 no cluster')
        if percentual < 5:
            alerta.append('cluster pequeno (< 5% da amostra)')
        if not alerta:
            alerta.append('sem alerta automático — revisar manualmente')

        linhas.append({
            'Cluster': c + 1,
            'Tamanho (n)': tamanho,
            '% da amostra': round(percentual, 1),
            'Habilidades predominantes (top 3)': habilidades_predominantes,
            'Proficiência média do cluster': round(media_geral_cluster, 2),
            'Pontos de atenção (revisar manualmente)': '; '.join(alerta)
        })

    return pd.DataFrame(linhas)


print("\n" + "=" * 70)
print("TABELAS-RESUMO POR CLUSTER (tamanho, % amostra, habilidades)")
print("=" * 70)

tabelas_resumo = {}
for k in [4, 5, 6]:
    tabela_resumo_k = montar_tabela_resumo(k)
    tabelas_resumo[k] = tabela_resumo_k
    print(f"\n--- Tabela-resumo dos clusters (K={k}) ---\n")
    print(tabela_resumo_k.to_string(index=False))

tabela_resumo_final = tabelas_resumo[k_final]
print(f"\n\n>>> Tabela a inserir no TCC (K={k_final}, solução final usada no trabalho) <<<\n")
print(tabela_resumo_final.to_string(index=False))

tabela_resumo_final.to_csv('tabela_resumo_clusters_K5.csv', index=False, encoding='utf-8-sig')
for k in [4, 5, 6]:
    tabelas_resumo[k].to_csv(f'tabela_resumo_clusters_K{k}.csv', index=False, encoding='utf-8-sig')


# ============================================================
# FINALIZAÇÃO
# ============================================================

print("\n=== ANÁLISE CONCLUÍDA ===")
print("Padronização aplicada: títulos, nomes de habilidades, casas decimais,")
print("escalas de cor (0-4 para proficiência; 0-100% para cargos) e cores")
print("fixas por cluster (1 a 6) em todos os gráficos, para K=4, K=5 e K=6.")
print(f"K final utilizado nas análises detalhadas do TCC: K={k_final}")
