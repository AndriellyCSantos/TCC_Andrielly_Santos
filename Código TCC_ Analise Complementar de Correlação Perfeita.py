# ============================================================
# TCC - ANÁLISE DE CORRELAÇÕES E REDUNDÂNCIA ENTRE SKILLS
# ============================================================
#
# Objetivo:
#   1. Carregar as bases de Skills e Designation
#   2. Integrar as bases pelo Eid
#   3. Calcular as correlações entre as 19 skills
#   4. Identificar pares com correlação >= 0,95
#   5. Identificar colunas exatamente idênticas
#   6. Verificar se os valores são iguais para os mesmos Eid
#   7. Gerar tabelas para utilização no TCC
#
# NÃO realiza:
#   - K-Means
#   - PCA
#   - Elbow
#   - Silhueta
#   - Histogramas
#   - Boxplots
#   - Heatmaps
#   - Outras análises do código original
# ============================================================


import pandas as pd
import numpy as np
import os


# ============================================================
# 1. DIRETÓRIO
# ============================================================

os.chdir(
    r'C:\Users\victo\OneDrive\Área de Trabalho\TCC'
)


# ============================================================
# 2. CARREGAMENTO DOS DADOS
# ============================================================

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


print("=" * 80)
print("ANÁLISE DE CORRELAÇÕES E REDUNDÂNCIA ENTRE SKILLS")
print("=" * 80)

print("\nBases carregadas:")
print(f"Skills:       {skills.shape}")
print(f"Designation:  {desig.shape}")


# ============================================================
# 3. TRATAMENTO DA DUPLICATA VB.Net / Vb.Net
# ============================================================

print("\n" + "=" * 80)
print("TRATAMENTO DAS COLUNAS VB.Net / Vb.Net")
print("=" * 80)


if 'Vb.Net' in skills.columns:

    # Mantém os valores da coluna Vb.Net
    # sob o nome padronizado VB.Net

    skills['VB.Net'] = skills['Vb.Net']

    skills.drop(
        columns=['Vb.Net'],
        inplace=True
    )

    print(
        "A coluna 'Vb.Net' foi incorporada como 'VB.Net'."
    )


elif 'VB.Net' in skills.columns:

    print(
        "A coluna 'VB.Net' já está no formato correto."
    )

else:

    print(
        "ATENÇÃO: nenhuma coluna VB.Net/Vb.Net foi encontrada."
    )


# ============================================================
# 4. LISTA DAS 19 SKILLS
# ============================================================

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


print("\n" + "=" * 80)
print("SKILLS ANALISADAS")
print("=" * 80)

print(
    f"Quantidade de skills: {len(colunas_skills)}"
)

for i, skill in enumerate(
    colunas_skills,
    start=1
):

    print(
        f"{i:02d} - {skill}"
    )


# ============================================================
# 5. VERIFICAÇÃO DAS COLUNAS
# ============================================================

print("\n" + "=" * 80)
print("VERIFICAÇÃO DAS COLUNAS")
print("=" * 80)


colunas_faltantes = [

    col
    for col in colunas_skills
    if col not in skills.columns

]


if len(colunas_faltantes) > 0:

    print(
        "\nATENÇÃO! As seguintes colunas não foram encontradas:"
    )

    for col in colunas_faltantes:
        print(
            f" - {col}"
        )

    raise ValueError(
        "Existem skills ausentes na base."
    )


print(
    "Todas as 19 skills foram encontradas."
)


# ============================================================
# 6. INTEGRAÇÃO DAS BASES PELO Eid
# ============================================================

base = pd.merge(

    skills,

    desig,

    on='Eid',

    how='inner'

)


print("\n" + "=" * 80)
print("INTEGRAÇÃO DAS BASES")
print("=" * 80)

print(
    f"Registros da base integrada: {len(base)}"
)

print(
    f"Colunas da base integrada: {base.shape[1]}"
)


# ============================================================
# 7. VERIFICAÇÃO DOS VALORES DAS SKILLS
# ============================================================

print("\n" + "=" * 80)
print("VERIFICAÇÃO DOS VALORES DAS SKILLS")
print("=" * 80)


valores_unicos = {}

for col in colunas_skills:

    valores = sorted(
        base[col]
        .dropna()
        .unique()
    )

    valores_unicos[col] = valores

    print(
        f"{col:<20} -> {valores}"
    )


# ============================================================
# 8. MATRIZ DE CORRELAÇÃO
# ============================================================

print("\n" + "=" * 80)
print("MATRIZ DE CORRELAÇÃO DE PEARSON")
print("=" * 80)


matriz_correlacao = (

    base[colunas_skills]

    .corr(
        method='pearson'
    )

)


print(
    matriz_correlacao
    .round(4)
    .to_string()
)


# ============================================================
# 9. TODAS AS CORRELAÇÕES ENTRE PARES
# ============================================================

corr_long = (

    matriz_correlacao

    .where(
        ~np.eye(
            matriz_correlacao.shape[0],
            dtype=bool
        )
    )

    .stack()

    .reset_index()

)


corr_long.columns = [

    'Skill 1',
    'Skill 2',
    'Correlação'

]


# Como a matriz possui A x B e B x A,
# criamos uma identificação única do par.

corr_long['Par'] = corr_long.apply(

    lambda linha:
        tuple(
            sorted(
                [
                    linha['Skill 1'],
                    linha['Skill 2']
                ]
            )
        ),

    axis=1

)


corr_long = (

    corr_long

    .drop_duplicates(
        subset='Par'
    )

    .drop(
        columns='Par'
    )

    .sort_values(
        by='Correlação',
        ascending=False
    )

    .reset_index(
        drop=True
    )

)


print("\n" + "=" * 80)
print("TODOS OS PARES DE SKILLS")
print("=" * 80)


print(
    corr_long
    .round(4)
    .to_string(index=False)
)


# ============================================================
# 10. PARES COM CORRELAÇÃO >= 0,95
# ============================================================

pares_095 = corr_long[
    corr_long['Correlação'] >= 0.95
].copy()


print("\n" + "=" * 80)
print("PARES COM CORRELAÇÃO >= 0,95")
print("=" * 80)


if len(pares_095) > 0:

    print(
        pares_095
        .round(6)
        .to_string(index=False)
    )

else:

    print(
        "Nenhum par apresentou correlação >= 0,95."
    )


# ============================================================
# 11. PARES COM CORRELAÇÃO EXATAMENTE IGUAL A 1
# ============================================================

pares_r1 = corr_long[
    np.isclose(
        corr_long['Correlação'],
        1.0
    )
].copy()


print("\n" + "=" * 80)
print("PARES COM CORRELAÇÃO EXATAMENTE IGUAL A 1,00")
print("=" * 80)


if len(pares_r1) > 0:

    print(
        pares_r1
        .round(6)
        .to_string(index=False)
    )

else:

    print(
        "Nenhum par apresentou correlação exatamente igual a 1,00."
    )


# ============================================================
# 12. VERIFICAÇÃO DE IDENTIDADE EXATA
# ============================================================
#
# Aqui está a parte mais importante para a sua pergunta:
#
# Se duas colunas possuem exatamente os mesmos valores para
# todos os Eid da base integrada, elas são consideradas
# exatamente idênticas.
#
# Isso é diferente de apenas possuir correlação alta.
#
# Exemplo:
#
# Python      Machine Learning
# 0           0
# 1           1
# 2           2
# 3           3
# 4           4
#
# Se isso ocorrer em TODAS as linhas, as duas colunas
# são exatamente idênticas.
#
# ============================================================

print("\n" + "=" * 80)
print("VERIFICAÇÃO DE COLUNAS EXATAMENTE IDÊNTICAS")
print("=" * 80)


pares_identicos = []


for i in range(
    len(colunas_skills)
):

    for j in range(
        i + 1,
        len(colunas_skills)
    ):

        col1 = colunas_skills[i]

        col2 = colunas_skills[j]


        serie1 = base[col1]

        serie2 = base[col2]


        identicas = serie1.equals(
            serie2
        )


        correlacao = (
            base[
                [col1, col2]
            ]
            .corr()
            .iloc[0, 1]
        )


        diferenca = (
            serie1
            -
            serie2
        )


        diferencas_unicas = (
            diferenca
            .dropna()
            .unique()
        )


        diferenca_constante = (
            len(diferencas_unicas) == 1
        )


        if identicas:

            pares_identicos.append({

                'Skill 1':
                    col1,

                'Skill 2':
                    col2,

                'Correlação':
                    correlacao,

                'Idênticas':
                    True,

                'Diferença constante':
                    diferenca_constante,

                'Diferença':
                    diferencas_unicas[0]
                    if len(diferencas_unicas) > 0
                    else 0

            })


pares_identicos_df = pd.DataFrame(
    pares_identicos
)


if len(pares_identicos_df) > 0:

    print(
        pares_identicos_df
        .round(6)
        .to_string(index=False)
    )

else:

    print(
        "Nenhuma coluna exatamente idêntica foi encontrada."
    )


# ============================================================
# 13. VERIFICAÇÃO DETALHADA DOS PARES COM CORRELAÇÃO >= 0,95
# ============================================================
#
# Esta tabela permite diferenciar:
#
#   - correlação muito alta
#   - correlação exatamente 1
#   - colunas realmente idênticas
#
# ============================================================

print("\n" + "=" * 80)
print("ANÁLISE DETALHADA DOS PARES COM CORRELAÇÃO >= 0,95")
print("=" * 80)


analise_altas = []


for _, linha in pares_095.iterrows():

    col1 = linha['Skill 1']

    col2 = linha['Skill 2']

    r = linha['Correlação']


    identicas = base[col1].equals(
        base[col2]
    )


    diferenca = (
        base[col1]
        -
        base[col2]
    )


    valores_diferenca = (
        diferenca
        .dropna()
        .unique()
    )


    if len(valores_diferenca) == 1:

        diferenca_constante = True

        diferenca_valor = (
            valores_diferenca[0]
        )

    else:

        diferenca_constante = False

        diferenca_valor = np.nan


    analise_altas.append({

        'Skill 1':
            col1,

        'Skill 2':
            col2,

        'Correlação':
            r,

        'Idênticas':
            identicas,

        'Diferença constante':
            diferenca_constante,

        'Diferença':
            diferenca_valor

    })


analise_altas_df = pd.DataFrame(
    analise_altas
)


if len(analise_altas_df) > 0:

    print(
        analise_altas_df
        .round(6)
        .to_string(index=False)
    )

else:

    print(
        "Nenhum par com correlação >= 0,95."
    )


# ============================================================
# 14. VERIFICAÇÃO DOS VALORES POR Eid
# ============================================================
#
# Para cada par altamente correlacionado, mostramos:
#
# Eid | Skill 1 | Skill 2 | Diferença
#
# Isso permite verificar diretamente se os valores são
# iguais para os mesmos funcionários.
#
# ============================================================

print("\n" + "=" * 80)
print("VERIFICAÇÃO DOS VALORES POR Eid")
print("=" * 80)


if len(pares_095) > 0:

    for _, linha in pares_095.iterrows():

        col1 = linha['Skill 1']

        col2 = linha['Skill 2']


        temp = base[
            [
                'Eid',
                col1,
                col2
            ]
        ].copy()


        temp['Diferença'] = (
            temp[col1]
            -
            temp[col2]
        )


        print("\n" + "-" * 80)

        print(
            f"{col1} × {col2}"
        )

        print(
            f"Correlação: "
            f"{linha['Correlação']:.6f}"
        )

        print("-" * 80)


        print(
            temp.to_string(
                index=False
            )
        )


# ============================================================
# 15. RESUMO QUANTITATIVO
# ============================================================

numero_skills = len(
    colunas_skills
)


numero_pares = (

    numero_skills
    *
    (numero_skills - 1)
    //
    2

)


numero_r95 = len(
    pares_095
)


numero_r1 = len(
    pares_r1
)


numero_identicos = len(
    pares_identicos_df
)


print("\n" + "=" * 80)
print("RESUMO DA ANÁLISE")
print("=" * 80)


print(
    f"Número de skills analisadas: "
    f"{numero_skills}"
)


print(
    f"Número de pares possíveis: "
    f"{numero_pares}"
)


print(
    f"Pares com correlação >= 0,95: "
    f"{numero_r95}"
)


print(
    f"Pares com correlação = 1,00: "
    f"{numero_r1}"
)


print(
    f"Pares de colunas exatamente idênticas: "
    f"{numero_identicos}"
)


# ============================================================
# 16. EXPORTAÇÃO DAS TABELAS
# ============================================================

print("\n" + "=" * 80)
print("EXPORTAÇÃO DOS RESULTADOS")
print("=" * 80)


# ------------------------------------------------------------
# Matriz completa de correlação
# ------------------------------------------------------------

matriz_correlacao.round(6).to_csv(

    'tabela_matriz_correlacao.csv',

    encoding='utf-8-sig'

)


# ------------------------------------------------------------
# Todos os pares
# ------------------------------------------------------------

corr_long.round(6).to_csv(

    'tabela_todas_correlacoes.csv',

    index=False,

    encoding='utf-8-sig'

)


# ------------------------------------------------------------
# Correlações >= 0,95
# ------------------------------------------------------------

pares_095.round(6).to_csv(

    'tabela_correlacoes_095.csv',

    index=False,

    encoding='utf-8-sig'

)


# ------------------------------------------------------------
# Correlações exatamente 1
# ------------------------------------------------------------

pares_r1.round(6).to_csv(

    'tabela_correlacoes_1.csv',

    index=False,

    encoding='utf-8-sig'

)


# ------------------------------------------------------------
# Colunas exatamente idênticas
# ------------------------------------------------------------

pares_identicos_df.round(6).to_csv(

    'tabela_colunas_identicas.csv',

    index=False,

    encoding='utf-8-sig'

)


# ------------------------------------------------------------
# Análise detalhada das correlações altas
# ------------------------------------------------------------

analise_altas_df.round(6).to_csv(

    'tabela_analise_correlacoes_altas.csv',

    index=False,

    encoding='utf-8-sig'

)


# ============================================================
# 17. TABELA RESUMIDA PARA O TCC
# ============================================================

resumo_tcc = pd.DataFrame({

    'Indicador': [

        'Número de skills',
        'Número de pares possíveis',
        'Pares com correlação >= 0,95',
        'Pares com correlação = 1,00',
        'Pares de colunas exatamente idênticas'

    ],

    'Resultado': [

        numero_skills,
        numero_pares,
        numero_r95,
        numero_r1,
        numero_identicos

    ]

})


print("\n" + "=" * 80)
print("TABELA RESUMIDA PARA O TCC")
print("=" * 80)


print(
    resumo_tcc.to_string(
        index=False
    )
)


resumo_tcc.to_csv(

    'tabela_resumo_correlacoes.csv',

    index=False,

    encoding='utf-8-sig'

)


# ============================================================
# 18. FINALIZAÇÃO
# ============================================================

print("\n" + "=" * 80)
print("ANÁLISE CONCLUÍDA")
print("=" * 80)

print("\nArquivos gerados:")

print(
    "1. tabela_matriz_correlacao.csv"
)

print(
    "2. tabela_todas_correlacoes.csv"
)

print(
    "3. tabela_correlacoes_095.csv"
)

print(
    "4. tabela_correlacoes_1.csv"
)

print(
    "5. tabela_colunas_identicas.csv"
)

print(
    "6. tabela_analise_correlacoes_altas.csv"
)

print(
    "7. tabela_resumo_correlacoes.csv"
)

print("\n")
print("=" * 80)
