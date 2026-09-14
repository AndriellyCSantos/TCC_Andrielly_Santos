import pandas as pd
import os
from sklearn.cluster import KMeans

# DIRETÓRIO

os.chdir(r'C:\Users\victo\OneDrive\Área de Trabalho\TCC')

# CARREGAR OS ARQUIVOS

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

# CORREÇÃO VB.Net / Vb.Net

skills['VB.Net'] = skills['Vb.Net']
skills.drop(columns=['Vb.Net'], inplace=True)

# 19 SKILLS

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

# INTEGRAÇÃO

base = pd.merge(
skills,
desig,
on='Eid',
how='inner'
)

# K = 1

km1 = KMeans(n_clusters=1, random_state=42, n_init=10)
km1.fit(base[colunas_skills])

# K = 2

km2 = KMeans(n_clusters=2, random_state=42, n_init=10)
km2.fit(base[colunas_skills])

# K = 3

km3 = KMeans(n_clusters=3, random_state=42, n_init=10)
km3.fit(base[colunas_skills])

# K = 4

km4 = KMeans(n_clusters=4, random_state=42, n_init=10)
km4.fit(base[colunas_skills])

# K = 5

km5 = KMeans(n_clusters=5, random_state=42, n_init=10)
km5.fit(base[colunas_skills])

# K = 6

km6 = KMeans(n_clusters=6, random_state=42, n_init=10)
km6.fit(base[colunas_skills])

# K = 7

km7 = KMeans(n_clusters=7, random_state=42, n_init=10)
km7.fit(base[colunas_skills])

# K = 8

km8 = KMeans(n_clusters=8, random_state=42, n_init=10)
km8.fit(base[colunas_skills])

# K = 9

km9 = KMeans(n_clusters=9, random_state=42, n_init=10)
km9.fit(base[colunas_skills])

# K = 10

km10 = KMeans(n_clusters=10, random_state=42, n_init=10)
km10.fit(base[colunas_skills])

# TABELA

tabela_inercia = pd.DataFrame({
'K': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
'Inércia': [
km1.inertia_,
km2.inertia_,
km3.inertia_,
km4.inertia_,
km5.inertia_,
km6.inertia_,
km7.inertia_,
km8.inertia_,
km9.inertia_,
km10.inertia_
]
})

tabela_inercia['Inércia'] = tabela_inercia['Inércia'].round(2)

print()
print('=== TABELA DE INÉRCIA — K=1 ATÉ K=10 ===')
print()
print(tabela_inercia.to_string(index=False))
