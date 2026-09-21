import os
import numpy as np
import pandas as pd

from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)

PASTA_DADOS = None      

K = 5                   
RANDOM_STATE = 42       
N_INIT = 10             

TCC_INERCIA_K5 = 12294.77
TCC_SILHUETA_K5 = 0.2554
TCC_TAMANHOS_K5 = [224, 174, 206, 219, 177]   # Clusters 1 a 5

LIMIAR_PREDOMINANTE = 3.0   # centroide >= 3,0 => habilidade predominante

if PASTA_DADOS is None:
    try:
        PASTA_DADOS = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        PASTA_DADOS = os.getcwd()
os.chdir(PASTA_DADOS)

PASTA_SAIDA = 'resultados_sensibilidade'
os.makedirs(PASTA_SAIDA, exist_ok=True)


def br(valor, casas=2):
    """Formata número no padrão brasileiro (vírgula decimal, ponto de milhar)."""
    s = f"{valor:,.{casas}f}"
    return s.replace(',', '§').replace('.', ',').replace('§', '.')

skills = pd.read_csv('Employee_Skills_Datset.csv', sep=',', encoding='latin-1')
desig = pd.read_csv('Employee_Designation.csv', sep=',', encoding='latin-1')

# Correção VB.Net / Vb.Net (mantém os valores da coluna 'Vb.Net')
skills['VB.Net'] = skills['Vb.Net']
skills.drop(columns=['Vb.Net'], inplace=True)

colunas_skills = [
    'Python', 'Machine Learning', 'Deep Learning', 'Data Analysis',
    'Asp.Net', 'Ado.Net', 'VB.Net', 'C#',
    'Java', 'Spring Boot', 'Hibernate',
    'NLP', 'CV',
    'JS', 'React', 'Node', 'Angular',
    'Dart', 'Flutter'
]

base = pd.merge(skills, desig, on='Eid', how='inner')

print('=' * 80)
print('ANÁLISE DE SENSIBILIDADE - REMOÇÃO DE COLUNAS IDÊNTICAS')
print('=' * 80)
print(f'Registros: {len(base)} | Habilidades: {len(colunas_skills)}')

grupos = []
restantes = list(colunas_skills)
while restantes:
    ref = restantes.pop(0)
    identicas = [c for c in restantes if base[ref].equals(base[c])]
    for c in identicas:
        restantes.remove(c)
    grupos.append([ref] + identicas)

grupos_redundantes = [g for g in grupos if len(g) > 1]
colunas_reduzidas = [g[0] for g in grupos]

print('\n--- Grupos de colunas idênticas (todos os registros) ---')
for g in grupos_redundantes:
    print(f'  {", ".join(g)}  -> mantida: {g[0]}')
print(f'\nGrupos redundantes: {len(grupos_redundantes)} '
      f'(esperado no TCC: 6)')
print(f'Variáveis mantidas: {len(colunas_reduzidas)} '
      f'(esperado no TCC: 10)')
print('Variáveis mantidas:', ', '.join(colunas_reduzidas))

if len(grupos_redundantes) != 6 or len(colunas_reduzidas) != 10:
    print('\n*** ATENÇÃO: o resultado difere do descrito no TCC '
          '(6 grupos / 10 variáveis). Revise antes de usar o texto. ***')

pd.DataFrame({
    'Grupo': range(1, len(grupos_redundantes) + 1),
    'Habilidades idênticas': ['; '.join(g) for g in grupos_redundantes],
    'Habilidade mantida': [g[0] for g in grupos_redundantes],
}).to_csv(f'{PASTA_SAIDA}/tabela_sensibilidade_grupos_identicos.csv',
          index=False, encoding='utf-8-sig')

X_orig = base[colunas_skills]
X_red = base[colunas_reduzidas]

km_orig = KMeans(n_clusters=K, random_state=RANDOM_STATE, n_init=N_INIT)
lab_orig = km_orig.fit_predict(X_orig)
sil_orig = silhouette_score(X_orig, lab_orig)

km_red = KMeans(n_clusters=K, random_state=RANDOM_STATE, n_init=N_INIT)
lab_red = km_red.fit_predict(X_red)
sil_red = silhouette_score(X_red, lab_red)

tam_orig = np.bincount(lab_orig, minlength=K)
tam_red = np.bincount(lab_red, minlength=K)

print('\n' + '=' * 80)
print('CONFERÊNCIA DA ANÁLISE PRINCIPAL (19 habilidades, K=5)')
print('=' * 80)
print(f'Inércia:   {br(km_orig.inertia_)}   (TCC: {br(TCC_INERCIA_K5)})')
print(f'Silhouette: {br(sil_orig, 4)}   (TCC: {br(TCC_SILHUETA_K5, 4)})')
print(f'Tamanhos:  {[int(x) for x in tam_orig]}   (TCC: {TCC_TAMANHOS_K5})')

confere = (
    abs(km_orig.inertia_ - TCC_INERCIA_K5) < 0.01
    and abs(sil_orig - TCC_SILHUETA_K5) < 0.00005
    and [int(x) for x in tam_orig] == TCC_TAMANHOS_K5
)
print('Reproduz o TCC:', 'SIM' if confere else
      'NÃO -> a numeração/tamanhos podem diferir do texto; revise')

def parear_clusters(rot_a, rot_b, k):
    """Devolve (contingência, dict {cluster_a: cluster_b}) maximizando
    o número de registros em comum."""
    cont = np.zeros((k, k), dtype=int)
    for a, b in zip(rot_a, rot_b):
        cont[a, b] += 1
    linhas, colunas = linear_sum_assignment(-cont)
    return cont, {int(a): int(b) for a, b in zip(linhas, colunas)}


contingencia, pares = parear_clusters(lab_orig, lab_red, K)

_perm = np.random.default_rng(0).permutation(K)
_c, _p = parear_clusters(lab_orig, _perm[lab_orig], K)
assert sum(_c[a, b] for a, b in _p.items()) == len(lab_orig), \
    'Falha no teste interno de pareamento'

concordantes = sum(contingencia[a, b] for a, b in pares.items())
pct_concordancia = concordantes / len(base) * 100
ari = adjusted_rand_score(lab_orig, lab_red)
nmi = normalized_mutual_info_score(lab_orig, lab_red)

print('\n' + '=' * 80)
print('COMPOSIÇÃO: COMPARAÇÃO ENTRE AS DUAS SOLUÇÕES')
print('=' * 80)
print(f'Adjusted Rand Index (ARI):              {br(ari, 4)}')
print(f'Normalized Mutual Information (NMI):    {br(nmi, 4)}')
print(f'Registros no cluster equivalente:       {concordantes} de {len(base)} '
      f'({br(pct_concordancia, 1)}%)')

ordem_red = [pares[a] for a in range(K)]
cont_df = pd.DataFrame(
    contingencia[:, ordem_red],
    index=[f'Principal - Cluster {a + 1}' for a in range(K)],
    columns=[f'Reduzida equiv. ao Cluster {a + 1}' for a in range(K)],
)
print('\nContingência (linhas: análise principal; colunas: análise reduzida, '
      'reordenada pelo pareamento):')
print(cont_df.to_string())
cont_df.to_csv(f'{PASTA_SAIDA}/tabela_sensibilidade_contingencia.csv',
               encoding='utf-8-sig')

linhas_tam = []
for a in range(K):
    b = pares[a]
    em_comum = int(contingencia[a, b])
    linhas_tam.append({
        'Cluster (análise principal)': a + 1,
        'n (principal)': int(tam_orig[a]),
        '% (principal)': round(tam_orig[a] / len(base) * 100, 1),
        'n (reduzida, cluster equivalente)': int(tam_red[b]),
        '% (reduzida)': round(tam_red[b] / len(base) * 100, 1),
        'Diferença de n': int(tam_red[b] - tam_orig[a]),
        'Registros em comum': em_comum,
        '% do cluster principal mantido': round(em_comum / tam_orig[a] * 100, 1),
    })
tab_tam = pd.DataFrame(linhas_tam)
print('\n--- Tamanho dos clusters ---')
print(tab_tam.to_string(index=False))
tab_tam.to_csv(f'{PASTA_SAIDA}/tabela_sensibilidade_tamanhos.csv',
               index=False, encoding='utf-8-sig')


cent_orig = (base.assign(_c=lab_orig)
             .groupby('_c')[colunas_reduzidas].mean())
cent_red = (base.assign(_c=lab_red)
            .groupby('_c')[colunas_reduzidas].mean())

cent_orig.index = [f'Cluster {i + 1}' for i in cent_orig.index]
cent_red_par = cent_red.loc[ordem_red].copy()
cent_red_par.index = [f'Cluster {a + 1}' for a in range(K)]

dif = (cent_red_par - cent_orig).abs()

print('\n' + '=' * 80)
print('PERFIS: CENTROIDES (10 habilidades mantidas)')
print('=' * 80)
print('\nAnálise principal:')
print(cent_orig.round(2).to_string())
print('\nAnálise reduzida (clusters pareados):')
print(cent_red_par.round(2).to_string())
print('\nDiferença absoluta:')
print(dif.round(2).to_string())
print(f'\nDiferença absoluta máxima: {br(dif.values.max(), 2)}')
print(f'Diferença absoluta média:  {br(dif.values.mean(), 2)}')

cent_orig.round(4).to_csv(
    f'{PASTA_SAIDA}/tabela_sensibilidade_centroides_principal.csv',
    encoding='utf-8-sig')
cent_red_par.round(4).to_csv(
    f'{PASTA_SAIDA}/tabela_sensibilidade_centroides_reduzida.csv',
    encoding='utf-8-sig')
dif.round(4).to_csv(
    f'{PASTA_SAIDA}/tabela_sensibilidade_diferenca_centroides.csv',
    encoding='utf-8-sig')

print(f'\n--- Habilidades predominantes (centroide >= '
      f'{br(LIMIAR_PREDOMINANTE, 1)}) ---')
linhas_pred = []
todas_iguais = True
for a in range(K):
    nome = f'Cluster {a + 1}'
    p_o = [c for c in colunas_reduzidas if cent_orig.loc[nome, c] >= LIMIAR_PREDOMINANTE]
    p_r = [c for c in colunas_reduzidas if cent_red_par.loc[nome, c] >= LIMIAR_PREDOMINANTE]
    iguais = set(p_o) == set(p_r)
    todas_iguais &= iguais
    linhas_pred.append({
        'Cluster (principal)': a + 1,
        'Predominantes - principal': '; '.join(p_o) if p_o else '(nenhuma)',
        'Predominantes - reduzida': '; '.join(p_r) if p_r else '(nenhuma)',
        'Coincidem': 'sim' if iguais else 'não',
    })
tab_pred = pd.DataFrame(linhas_pred)
print(tab_pred.to_string(index=False))
tab_pred.to_csv(f'{PASTA_SAIDA}/tabela_sensibilidade_predominantes.csv',
                index=False, encoding='utf-8-sig')

print('\n' + '=' * 80)
print('K = 4, 5 e 6 NA BASE REDUZIDA (10 variáveis)')
print('=' * 80)
linhas_k = []
for k in (4, 5, 6):
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT)
    rot = km.fit_predict(X_red)
    linhas_k.append({'K': k, 'Inércia (10 variáveis)': round(km.inertia_, 2),
                     'Silhouette (10 variáveis)': round(silhouette_score(X_red, rot), 4)})
tab_k = pd.DataFrame(linhas_k)
print(tab_k.to_string(index=False))
print('\nAtenção: inércia e Silhouette calculados em espaços de dimensões '
      'diferentes (19 x 10) não são diretamente comparáveis em valor absoluto.')
tab_k.to_csv(f'{PASTA_SAIDA}/tabela_sensibilidade_k4_k5_k6_reduzida.csv',
             index=False, encoding='utf-8-sig')

resumo = pd.DataFrame({
    'Indicador': [
        'Grupos de colunas idênticas', 'Variáveis mantidas',
        'ARI', 'NMI', 'Registros no cluster equivalente (n)',
        'Registros no cluster equivalente (%)',
        'Diferença absoluta máxima entre centroides',
        'Diferença absoluta média entre centroides',
        'Silhouette - principal (19)', 'Silhouette - reduzida (10)',
        'Habilidades predominantes coincidem em todos os clusters',
    ],
    'Resultado': [
        len(grupos_redundantes), len(colunas_reduzidas),
        round(ari, 4), round(nmi, 4), int(concordantes),
        round(pct_concordancia, 1),
        round(float(dif.values.max()), 2), round(float(dif.values.mean()), 2),
        round(sil_orig, 4), round(sil_red, 4),
        'sim' if todas_iguais else 'não',
    ],
})
resumo.to_csv(f'{PASTA_SAIDA}/tabela_sensibilidade_resumo.csv',
              index=False, encoding='utf-8-sig')

n_min, n_max = int(tam_red.min()), int(tam_red.max())
p_min, p_max = n_min / len(base) * 100, n_max / len(base) * 100
o_min, o_max = int(tam_orig.min()), int(tam_orig.max())

if todas_iguais:
    frase_pred = ('Os clusters equivalentes mantiveram as mesmas habilidades '
                  'predominantes')
else:
    frase_pred = ('As habilidades predominantes diferiram em pelo menos um '
                  'cluster equivalente (ver tabela de predominantes)')

paragrafo = (
    f'Na análise de sensibilidade com dez variáveis, o K-Means com K=5 '
    f'resultou em clusters com {n_min} a {n_max} registros '
    f'({br(p_min, 1)}% a {br(p_max, 1)}% da amostra), contra {o_min} a {o_max} '
    f'na análise principal. A concordância entre as duas soluções, medida '
    f'pelo ARI, foi de {br(ari, 2)}, e {br(pct_concordancia, 1)}% dos registros '
    f'permaneceram no cluster equivalente. {frase_pred}, com diferença '
    f'absoluta máxima de {br(float(dif.values.max()), 2)} pontos entre os '
    f'centroides das dez habilidades comparadas. O Silhouette da solução '
    f'reduzida foi de {br(sil_red, 4)}, valor que não é diretamente comparável '
    f'ao da análise principal ({br(sil_orig, 4)}) por ter sido calculado em '
    f'outro conjunto de variáveis.'
)

print('\n' + '=' * 80)
print('PARÁGRAFO-BASE PARA O TCC (revisar a interpretação antes de usar)')
print('=' * 80)
print(paragrafo)

with open(f'{PASTA_SAIDA}/texto_sensibilidade_para_TCC.txt', 'w',
          encoding='utf-8') as f:
    f.write(paragrafo + '\n\n')
    f.write('Valores para preencher os campos destacados no TCC:\n')
    f.write(f'  Tamanho mínimo/máximo (reduzida): {n_min} / {n_max}\n')
    f.write(f'  ARI: {br(ari, 2)}\n')
    f.write(f'  % de registros no cluster equivalente: {br(pct_concordancia, 1)}\n')
    f.write(f'  Diferença absoluta máxima entre centroides: '
            f'{br(float(dif.values.max()), 2)}\n')
    f.write(f'  Silhouette (reduzida): {br(sil_red, 4)}\n')
    f.write(f'  Predominantes coincidem em todos os clusters: '
            f'{"sim" if todas_iguais else "não"}\n')

print(f'\nArquivos gerados na pasta "{PASTA_SAIDA}".')
print('=' * 80)
