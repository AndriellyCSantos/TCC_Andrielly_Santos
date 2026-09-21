import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.metrics import euclidean_distances, silhouette_score, silhouette_samples
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')

PASTA_DADOS = None          

PASTA_FIGURAS = 'figuras'   
DPI_SAIDA = 400             
SALVAR_SVG = True           
MOSTRAR_FIGURAS = True     

LARGURA_CM = 16.0          
LARGURA_POL = LARGURA_CM / 2.54

if PASTA_DADOS is None:
    try:
        PASTA_DADOS = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        PASTA_DADOS = os.getcwd()
os.chdir(PASTA_DADOS)

os.makedirs(os.path.join(PASTA_FIGURAS, 'suplementares'), exist_ok=True)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
    'axes.edgecolor': 'black',
    'axes.linewidth': 1.5,
    'axes.labelcolor': 'black',
    'axes.grid': False,
    'text.color': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black',
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'svg.fonttype': 'path',
})


PALETA_CLUSTERS = {
    1: '#1f77b4',   # azul
    2: '#d62728',   # vermelho
    3: '#2ca02c',   # verde
    4: '#ff7f0e',   # laranja
    5: '#9467bd',   # roxo
    6: '#8c564b',   # marrom
}

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
    """Converte nomes de habilidades para os rótulos padronizados."""
    return [LABELS_HABILIDADES[s] for s in lista_skills]


def nome_linha(skill):
    """Rótulo padronizado em uma única linha."""
    return LABELS_HABILIDADES[skill].replace('\n', ' ')


CASAS_PROFICIENCIA = 2      
CASAS_PERCENTUAL = 2        
CASAS_INERCIA = 2           
CASAS_SILHUETA = 4         

VMIN_PROF, VMAX_PROF = 0, 4
VMIN_PCT, VMAX_PCT = 0, 100
CMAP_HEATMAP = 'Greys'

FS_EIXO = 9
FS_TICK = 8
FS_ANOT = 8
FS_LEG = 8
FS_PAINEL = 11


def fmt_br(valor, casas=2):
    """Número no padrão brasileiro: vírgula decimal, ponto de milhar."""
    s = f'{valor:,.{casas}f}'
    return s.replace(',', '§').replace('.', ',').replace('§', '.')


def formatador_br(casas):
    """Formatador de eixo/barra de cores no padrão brasileiro."""
    return FuncFormatter(lambda x, pos: fmt_br(x, casas))


def nova_figura(altura_cm, nrows=1, ncols=1, **kwargs):
    """Cria figura com 16 cm de largura (tamanho real de impressão)."""
    return plt.subplots(
        nrows, ncols,
        figsize=(LARGURA_POL, altura_cm / 2.54),
        layout='constrained',
        **kwargs
    )


def estilo_eixos(ax):
    """Sem grade e sem bordas superior/direita; ticks pretos."""
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', labelsize=FS_TICK, colors='black', length=3.5)


def rotulo_painel(ax, letra):
    """Letra maiúscula, sem pontuação, no canto superior esquerdo do painel."""
    ax.text(0.0, 1.03, letra, transform=ax.transAxes, fontsize=FS_PAINEL,
            fontweight='bold', ha='left', va='bottom')


def salvar_figura(fig, nome, suplementar=False):
    """Salva PNG (DPI_SAIDA) e SVG; exibe e fecha a figura."""
    pasta = os.path.join(PASTA_FIGURAS, 'suplementares') if suplementar else PASTA_FIGURAS
    caminho = os.path.join(pasta, nome)
    fig.savefig(caminho + '.png', dpi=DPI_SAIDA)
    if SALVAR_SVG:
        fig.savefig(caminho + '.svg')
    print(f'   figura salva: {caminho}.png')
    if MOSTRAR_FIGURAS:
        plt.show()
    plt.close(fig)


def legenda_clusters(k, ax, loc='upper right', titulo='Cluster'):
    """Legenda com as cores fixas de PALETA_CLUSTERS (clusters 1..k)."""
    handles = [
        mpatches.Patch(color=PALETA_CLUSTERS[c], label=f'Cluster {c}')
        for c in range(1, k + 1)
    ]
    return ax.legend(handles=handles, title=titulo, loc=loc, fontsize=FS_LEG,
                     title_fontsize=FS_LEG, frameon=False)

skills = pd.read_csv('Employee_Skills_Datset.csv', sep=',', encoding='latin-1')
desig = pd.read_csv('Employee_Designation.csv', sep=',', encoding='latin-1')

print("=== 1.1 CARREGAMENTO ===")
print("Skills:", skills.shape)
print("Designation:", desig.shape)

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

print("\n=== 1.3 TRATAMENTO DE DUPLICATA ===")
skills['VB.Net'] = skills['Vb.Net']
skills.drop(columns=['Vb.Net'], inplace=True)
print("Corrigido: valores reais de 'Vb.Net' mantidos como 'VB.Net'")
print("VB.Net após correção:", sorted(skills['VB.Net'].unique()))


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


base = pd.merge(skills, desig, on='Eid', how='inner')
print("\n=== 1.4 INTEGRAÇÃO ===")
print("Base integrada:", base.shape)

print("\n=== 1.5 ESTATÍSTICAS DESCRITIVAS ===")
print(base[colunas_skills].describe().round(2))

print("\n=== FIGURA 1 - DISTRIBUIÇÃO DOS CARGOS ===")

contagem = base['Designation'].value_counts()

fig, ax = nova_figura(7.5)
ax.barh(contagem.index, contagem.values, color='white', edgecolor='black', linewidth=1.2)

for i, valor in enumerate(contagem.values):
    ax.text(valor + contagem.values.max() * 0.012, i, fmt_br(valor, 0),
            va='center', fontsize=FS_ANOT, color='black')

ax.set_xlabel('Número de funcionários', fontsize=FS_EIXO)
ax.set_ylabel('Cargo', fontsize=FS_EIXO)
ax.set_xlim(0, contagem.values.max() * 1.10)
ax.xaxis.set_major_formatter(formatador_br(0))
estilo_eixos(ax)
salvar_figura(fig, 'fig01_cargos')

def figura_caixas(lista, nome_arquivo, altura_cm, rotacao):
    ordem = base[lista].mean().sort_values().index

    fig, ax = nova_figura(altura_cm)
    base[ordem].boxplot(
        ax=ax, patch_artist=False, widths=0.5,
        medianprops=dict(color='black', linewidth=1.5),
        boxprops=dict(color='black', linewidth=1.2),
        whiskerprops=dict(color='black', linewidth=1.2),
        capprops=dict(color='black', linewidth=1.2),
        flierprops=dict(marker='o', markerfacecolor='black',
                        markeredgecolor='black', markersize=3, alpha=0.7)
    )
    if rotacao:
        ax.set_xticklabels([nome_linha(s) for s in ordem], rotation=rotacao,
                           ha='right', rotation_mode='anchor', fontsize=FS_TICK)
    else:
        ax.set_xticklabels(rotulos(ordem), rotation=0, fontsize=FS_TICK)
    ax.set_ylabel('Nível de proficiência (0 a 4)', fontsize=FS_EIXO)
    ax.set_xlabel('Habilidades técnicas', fontsize=FS_EIXO)
    ax.set_ylim(VMIN_PROF, VMAX_PROF)
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.yaxis.set_major_formatter(formatador_br(0))
    ax.axhline(y=2, color='black', linestyle='--', linewidth=1.2)
    estilo_eixos(ax)
    salvar_figura(fig, nome_arquivo)


print("\n=== FIGURA 2 - CAIXAS: CIÊNCIA DE DADOS E IA ===")
figura_caixas(skills_ds, 'fig02_caixas_ciencia_dados', altura_cm=7.5, rotacao=0)

print("\n=== FIGURA 3 - CAIXAS: DESENVOLVIMENTO E FRAMEWORKS ===")
figura_caixas(skills_dev_box, 'fig03_caixas_desenvolvimento', altura_cm=8.5, rotacao=45)

def figura_histogramas(lista, nrows, ncols, altura_cm, nome_arquivo):
    fig, axes = nova_figura(altura_cm, nrows, ncols, sharey=True)
    axes = np.atleast_1d(axes).flatten()

    for i, col in enumerate(lista):
        ax = axes[i]
        ax.hist(base[col], bins=np.arange(-0.5, 5, 1), color='white',
                edgecolor='black', linewidth=1.0)
        ax.set_title(nome_linha(col), fontsize=FS_EIXO, color='black', pad=3)
        ax.set_xticks([0, 1, 2, 3, 4])
        ax.set_xlim(-0.6, 4.6)
        ax.yaxis.set_major_formatter(formatador_br(0))
        estilo_eixos(ax)

    for j in range(len(lista), len(axes)):
        axes[j].set_visible(False)

    fig.supxlabel('Nível de proficiência (0 a 4)', fontsize=FS_EIXO)
    fig.supylabel('Frequência', fontsize=FS_EIXO)
    salvar_figura(fig, nome_arquivo)


print("\n=== FIGURA 4 - HISTOGRAMAS: CIÊNCIA DE DADOS E IA ===")
figura_histogramas(skills_ds, 2, 3, 9.0, 'fig04_histogramas_ciencia_dados')

print("\n=== FIGURA 5 - HISTOGRAMAS: DESENVOLVIMENTO E FRAMEWORKS ===")
figura_histogramas(skills_dev_box, 4, 4, 14.5, 'fig05_histogramas_desenvolvimento')


desc_custom = pd.DataFrame({
    'Moda': base[colunas_skills].mode().iloc[0],
    'Média': base[colunas_skills].mean().round(2),
    'Desvio-Padrão': base[colunas_skills].std().round(2),
    'Amplitude': (base[colunas_skills].max() - base[colunas_skills].min())
}).reset_index()

desc_custom.columns = ['Skill', 'Moda', 'Média', 'Desvio-Padrão', 'Amplitude']

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.float_format', lambda x: f'{x:.2f}')

print("\n=== TABELA 2 - ESTATÍSTICAS DESCRITIVAS ===\n")
print(desc_custom.to_string(index=False))

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


print("\n=== FIGURA 6 - ELBOW METHOD ===")

inercias_elbow = []
k_range = range(2, 12)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(base[colunas_skills])
    inercias_elbow.append(km.inertia_)

fig, ax = nova_figura(7.5)
ax.plot(list(k_range), inercias_elbow, color='black', linewidth=1.5,
        marker='o', markerfacecolor='white', markeredgecolor='black',
        markeredgewidth=1.5, markersize=6)
ax.set_xlabel('Número de clusters (K)', fontsize=FS_EIXO)
ax.set_ylabel('Inércia', fontsize=FS_EIXO)
ax.set_xticks(list(k_range))
ax.yaxis.set_major_formatter(formatador_br(0))
estilo_eixos(ax)
salvar_figura(fig, 'fig06_cotovelo')


print("\n=== FIGURA 7 - SILHOUETTE POR K ===")

silhuetas_k = []
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(base[colunas_skills])
    silhuetas_k.append(silhouette_score(base[colunas_skills], labels))

fig, ax = nova_figura(7.5)
ax.plot(list(k_range), silhuetas_k, color='black', linewidth=1.5,
        marker='o', markerfacecolor='white', markeredgecolor='black',
        markeredgewidth=1.5, markersize=6)
ax.set_xlabel('Número de clusters (K)', fontsize=FS_EIXO)
ax.set_ylabel('Índice de Silhouette médio', fontsize=FS_EIXO)
ax.set_xticks(list(k_range))
ax.yaxis.set_major_formatter(formatador_br(2))
estilo_eixos(ax)
salvar_figura(fig, 'fig07_silhouette')


print("\n=== FIGURA 8 - COMPARAÇÃO K=4, K=5 E K=6 ===\n")

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
        f"K={k} | Inércia: {fmt_br(resultados_k[k]['inercia'], CASAS_INERCIA)} | "
        f"Silhouette: {fmt_br(resultados_k[k]['silhueta'], CASAS_SILHUETA)}"
    )

ks = [4, 5, 6]
inercias = [resultados_k[k]['inercia'] for k in ks]
silhuetas = [resultados_k[k]['silhueta'] for k in ks]
rotulos_k = ['K=4', 'K=5', 'K=6']

fig, axes = nova_figura(7.0, 1, 2)

axes[0].bar(rotulos_k, inercias, color='white', edgecolor='black',
            linewidth=1.2, width=0.5)
for i, v in enumerate(inercias):
    axes[0].text(i, v + max(inercias) * 0.02, fmt_br(v, CASAS_INERCIA),
                 ha='center', fontsize=FS_ANOT)
axes[0].set_ylabel('Inércia', fontsize=FS_EIXO)
axes[0].set_ylim(0, max(inercias) * 1.15)
axes[0].yaxis.set_major_formatter(formatador_br(0))
estilo_eixos(axes[0])
rotulo_painel(axes[0], 'A')

axes[1].bar(rotulos_k, silhuetas, color='white', edgecolor='black',
            linewidth=1.2, width=0.5)
for i, v in enumerate(silhuetas):
    axes[1].text(i, v + max(silhuetas) * 0.02, fmt_br(v, CASAS_SILHUETA),
                 ha='center', fontsize=FS_ANOT)
axes[1].set_ylabel('Índice de Silhouette médio', fontsize=FS_EIXO)
axes[1].set_ylim(0, max(silhuetas) * 1.2)
axes[1].set_yticks(np.arange(0, max(silhuetas) * 1.2, 0.05))
axes[1].yaxis.set_major_formatter(formatador_br(2))
estilo_eixos(axes[1])
rotulo_painel(axes[1], 'B')

salvar_figura(fig, 'fig08_comparacao_k4_k5_k6')

k_final = 5
kmeans = resultados_k[k_final]['modelo']
clusters = resultados_k[k_final]['labels']
centroids = resultados_k[k_final]['centroids']
base['Cluster'] = clusters

print(f"\n=== K-MEANS APLICADO (K={k_final}) ===")
print(base['Cluster'].value_counts().sort_index())


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

print("\n=== FIGURA 9 - PCA (K=4, K=5, K=6) ===")

pca_plot = PCA(n_components=2)
X_pca_plot = pca_plot.fit_transform(base[colunas_skills])
var_exp = pca_plot.explained_variance_ratio_

print(f"Variância explicada PC1: {fmt_br(var_exp[0] * 100, 2)}%")
print(f"Variância explicada PC2: {fmt_br(var_exp[1] * 100, 2)}%")
print(f"Variância explicada PC1+PC2: {fmt_br(var_exp[:2].sum() * 100, 2)}%  "
      f"(informar no texto: a PCA serve apenas para visualização)")

fig, axes = plt.subplots(
    1, 3, figsize=(LARGURA_POL, 7.8 / 2.54), sharex=True, sharey=True
)
fig.subplots_adjust(left=0.085, right=0.995, top=0.91, bottom=0.27, wspace=0.06)

for ax, k_plot, letra in zip(axes, [4, 5, 6], ['A', 'B', 'C']):
    labels_plot = resultados_k[k_plot]['labels']
    cents_pca = pca_plot.transform(resultados_k[k_plot]['centroids'])
    cores_pontos = [PALETA_CLUSTERS[label + 1] for label in labels_plot]

    ax.scatter(X_pca_plot[:, 0], X_pca_plot[:, 1], c=cores_pontos,
               alpha=0.7, s=7, linewidths=0)
    ax.scatter(cents_pca[:, 0], cents_pca[:, 1], c='black', s=45,
               marker='X', zorder=5)
    estilo_eixos(ax)
    ax.xaxis.set_major_formatter(formatador_br(0))
    ax.yaxis.set_major_formatter(formatador_br(0))
    rotulo_painel(ax, letra)

axes[0].set_ylabel(f'PC2 ({fmt_br(var_exp[1] * 100, 2)}% da variância)',
                   fontsize=FS_EIXO)
fig.supxlabel(f'PC1 ({fmt_br(var_exp[0] * 100, 2)}% da variância)',
              fontsize=FS_EIXO, y=0.115)

# Legenda única e completa (clusters 1 a 6 + centroide)
handles = [mpatches.Patch(color=PALETA_CLUSTERS[c], label=f'Cluster {c}')
           for c in range(1, 7)]
handles.append(Line2D([0], [0], marker='X', color='w', markerfacecolor='black',
                      markersize=7, label='Centroide'))
fig.legend(handles=handles, loc='lower center', ncol=7, fontsize=FS_LEG,
           frameon=False, handletextpad=0.4, columnspacing=1.0,
           bbox_to_anchor=(0.5, 0.0))

salvar_figura(fig, 'fig09_pca_k4_k5_k6')

perfis_por_k = {}


def anotacoes_br(valores, casas):
    """Matriz de textos com vírgula decimal para anotar o heatmap."""
    return np.array([[fmt_br(v, casas) for v in linha] for linha in valores])


def estilizar_heatmap(ax, rot_x):
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(axis='both', length=0, labelsize=FS_TICK)
    if rot_x:
        plt.setp(ax.get_xticklabels(), rotation=rot_x, ha='right',
                 rotation_mode='anchor')
    else:
        plt.setp(ax.get_xticklabels(), rotation=0)
    plt.setp(ax.get_yticklabels(), rotation=0)
    cbar = ax.collections[0].colorbar
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=FS_TICK, length=3)
    cbar.ax.yaxis.label.set_size(FS_EIXO)


def plot_heatmap_perfil(k, skills_subset, nome_base, altura_cm, rot_x):
    labels_k = resultados_k[k]['labels']
    base_temp = base.copy()
    base_temp['Cluster_temp'] = labels_k

    perfil_k = base_temp.groupby('Cluster_temp')[skills_subset].mean().round(2)
    perfil_k.index = range(1, k + 1)
    perfil_k.index.name = 'Cluster'
    perfil_plot = perfil_k.rename(columns=lambda s: LABELS_HABILIDADES[s])

    fig, ax = nova_figura(altura_cm)
    sns.heatmap(
        perfil_plot, ax=ax, annot=anotacoes_br(perfil_plot.values, CASAS_PROFICIENCIA),
        fmt='', annot_kws={'size': FS_ANOT},
        cmap=CMAP_HEATMAP, vmin=VMIN_PROF, vmax=VMAX_PROF,
        linewidths=0.5, linecolor='white',
        cbar_kws={'label': 'Proficiência média (escala 0 a 4)',
                  'format': formatador_br(1)}
    )
    ax.set_xlabel('Habilidades técnicas', fontsize=FS_EIXO)
    ax.set_ylabel('Cluster', fontsize=FS_EIXO)
    estilizar_heatmap(ax, rot_x)

    suplementar = (k != k_final)
    nome = f'{nome_base}_k{k}'
    salvar_figura(fig, nome, suplementar=suplementar)
    return perfil_k


for k in [4, 5, 6]:
    print(f"\n=== PERFIL MÉDIO POR CLUSTER (K={k}) — CIÊNCIA DE DADOS/IA ===")
    perfil_ds_k = plot_heatmap_perfil(
        k, skills_ds,
        'fig10_perfil_ciencia_dados' if k == k_final else 'perfil_ciencia_dados',
        altura_cm=0.75 * k + 2.6, rot_x=0)
    print(perfil_ds_k)

    print(f"\n=== PERFIL MÉDIO POR CLUSTER (K={k}) — DESENVOLVIMENTO/FRAMEWORKS ===")
    perfil_dev_k = plot_heatmap_perfil(
        k, skills_dev_box,
        'fig11_perfil_desenvolvimento' if k == k_final else 'perfil_desenvolvimento',
        altura_cm=0.75 * k + 3.4, rot_x=45)
    print(perfil_dev_k)

    base_temp = base.copy()
    base_temp['Cluster_temp'] = resultados_k[k]['labels']
    perfis_por_k[k] = base_temp.groupby('Cluster_temp')[colunas_skills].mean().round(2)


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

        print(f"--- Cluster {c+1} ({tamanho} funcionários | "
              f"{fmt_br(percentual, CASAS_PERCENTUAL)}% da base) ---")
        for skill, valor in top3.items():
            print(f"    • {nome_linha(skill)}: {fmt_br(valor, CASAS_PROFICIENCIA)}")
        print()


silhueta_media = silhouette_score(base[colunas_skills], clusters)
silhueta_amostras = silhouette_samples(base[colunas_skills], clusters)
base['Silhueta'] = silhueta_amostras

print(f"\n=== ÍNDICE DE SILHUETA (K={k_final}) ===")
print(f"Silhueta média geral: {fmt_br(silhueta_media, CASAS_SILHUETA)}")
print("\nSilhueta média por cluster:")
print(base.groupby('Cluster')['Silhueta'].mean().round(4).to_string())

if silhueta_media >= 0.50:
    print("\nInterpretação: estrutura FORTE (>= 0,50)")
elif silhueta_media >= 0.25:
    print("\nInterpretação: estrutura RAZOÁVEL (0,25 a 0,50)")
else:
    print("\nInterpretação: estrutura FRACA (< 0,25)")


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
            pd.crosstab(base_temp['Cluster_temp'], base_temp['Designation'],
                        normalize='index') * 100
        ).round(2)
        anot = anotacoes_br(tabela.values, CASAS_PERCENTUAL)
        label_cbar = '% de funcionários dentro do cluster'
        vmin, vmax = VMIN_PCT, VMAX_PCT
        fmt_cbar = formatador_br(0)
        nome_base = 'fig13_cargos_cluster_percentual'
    else:
        tabela = pd.crosstab(base_temp['Cluster_temp'], base_temp['Designation'])
        anot = anotacoes_br(tabela.values, 0)
        label_cbar = 'Quantidade de funcionários'
        vmin, vmax = 0, _max_abs_cargos
        fmt_cbar = formatador_br(0)
        nome_base = 'fig12_cargos_cluster_absoluto'

    tabela.index = range(1, k + 1)
    tabela.index.name = 'Cluster'

    tipo = 'Percentual' if normalizar else 'Absoluta'
    print(f"\n=== DISTRIBUIÇÃO {tipo.upper()} DE CARGOS POR CLUSTER (K={k}) ===\n")
    print(tabela.to_string())

    fig, ax = nova_figura(0.75 * k + 4.6)
    sns.heatmap(
        tabela, ax=ax, annot=anot, fmt='', annot_kws={'size': FS_ANOT},
        cmap=CMAP_HEATMAP, vmin=vmin, vmax=vmax,
        linewidths=0.5, linecolor='white',
        cbar_kws={'label': label_cbar, 'format': fmt_cbar}
    )
    ax.set_xlabel('Cargo', fontsize=FS_EIXO)
    ax.set_ylabel('Cluster', fontsize=FS_EIXO)
    estilizar_heatmap(ax, rot_x=45)

    suplementar = (k != k_final)
    nome = nome_base + f'_k{k}' if not suplementar else nome_base.replace('fig12_', '').replace('fig13_', '') + f'_k{k}'
    salvar_figura(fig, nome, suplementar=suplementar)

    return tabela


tabelas_cargos_abs = {}
tabelas_cargos_pct = {}

for k in [4, 5, 6]:
    tabelas_cargos_abs[k] = plot_heatmap_cargos(k, normalizar=False)
    tabelas_cargos_pct[k] = plot_heatmap_cargos(k, normalizar=True)

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
            f'{nome_linha(skill)} ({fmt_br(valor, CASAS_PROFICIENCIA)})'
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


print("\n=== ANÁLISE CONCLUÍDA ===")
print(f'Figuras salvas em "{PASTA_FIGURAS}" (PNG {DPI_SAIDA} dpi'
      f'{" + SVG" if SALVAR_SVG else ""}).')
print("Sem títulos nas imagens; vírgula decimal; rótulos e cores padronizados;")
print("escalas de cor fixas (0-4 para proficiência; 0-100% para cargos).")
