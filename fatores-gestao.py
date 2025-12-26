import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Carregar o arquivo de jogos
df = pd.read_csv("SPFC_JOGOS_COMPLETOS_CONSOLIDADO.csv", sep=';', encoding='utf-8-sig')

# 2. Criar a variável alvo numérica (Vitória = 1, Outros = 0)
df['Vitoria'] = df['Status_Jogo'].apply(lambda x: 1 if x == 'Vitória' else 0)

# 3. Gerar Variáveis Dummy
# Criamos todas primeiro para poder escolher qual dropar manualmente
dummies = pd.get_dummies(df['Gestao'], prefix='Pres')

# 4. EVITANDO A ARMADILHA DA VARIÁVEL DUMMY (The Dummy Variable Trap)
# Vamos dropar propositalmente o Juvenal Juvêncio para ele ser nossa referência (Baseline)
# Se todas as outras forem 0, significa que estamos falando da era Juvenal.
baseline = 'Pres_Juvenal Juvêncio (2006-2014)'

if baseline in dummies.columns:
    df_modelo = dummies.drop(columns=[baseline])
    print(f"--- Variável de Referência (Baseline): {baseline} ---")
else:
    # Caso o nome seja levemente diferente no CSV, pegamos o que contém Juvenal
    col_juvenal = [c for c in dummies.columns if 'Juvenal' in c][0]
    df_modelo = dummies.drop(columns=[col_juvenal])
    print(f"--- Variável de Referência (Baseline): {col_juvenal} ---")

# 5. Concatenar com a coluna Vitória para análise
df_analise = pd.concat([df_modelo, df['Vitoria']], axis=1)

# 6. Calcular a Correlação
# Isso mostra o impacto de cada presidente EM RELAÇÃO ao Juvenal
correlacao = df_analise.corr()['Vitoria'].drop('Vitoria').sort_values()

# 7. Visualização Profissional
plt.figure(figsize=(11, 6))
colors = ['red' if x < 0 else 'green' for x in correlacao]
correlacao.plot(kind='barh', color=colors, alpha=0.8)

plt.title('Impacto na Probabilidade de Vitória (Baseline: Era Juvenal Juvêncio)', fontsize=14, fontweight='bold')
plt.xlabel('Coeficiente de Correlação (Diferença em relação à Referência)', fontsize=11)
plt.ylabel('Gestão Analisada', fontsize=11)
plt.axvline(0, color='black', linestyle='-', linewidth=1.5) # Linha do Baseline
plt.grid(axis='x', linestyle='--', alpha=0.4)

# Adicionando anotação para o recrutador
plt.annotate('Baseline (Juvenal Juvêncio)', xy=(0, 0.5), xytext=(0.02, 0.5),
             arrowprops=dict(facecolor='black', shrink=0.05),
             fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()

print("\nResultados da Correlação (Relativos ao Baseline):")
print(correlacao)