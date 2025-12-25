import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURAÇÃO ---
ARQUIVO_FINANCEIRO = "spfc_financeiro_FINAL_MILHOES.csv"

# Carrega os dados
df = pd.read_csv(ARQUIVO_FINANCEIRO, sep=';')

# Configuração visual para gráficos profissionais
sns.set_theme(style="whitegrid")
plt.figure(figsize=(12, 6))

# --- 1. PREPARAÇÃO DOS DADOS ---
# Vamos separar Receitas e Despesas para comparar
# Nota: Ajuste os termos no .isin() se no seu CSV estiverem como "Receita", "Receitas", etc.
df_receitas = df[df['Tipo'].str.contains('Receita', case=False)].copy()
df_despesas = df[df['Tipo'].str.contains('Despesa|Saída', case=False)].copy()

# Agrupamento por Gestão (Soma Total)
receita_por_gestao = df_receitas.groupby('Gestao')['Valor_Milhoes'].sum().reset_index()
despesa_por_gestao = df_despesas.groupby('Gestao')['Valor_Milhoes'].sum().reset_index()

# Renomeia para facilitar o merge
receita_por_gestao.columns = ['Gestao', 'Total_Receita']
despesa_por_gestao.columns = ['Gestao', 'Total_Despesa']

# Junta tudo num dataframe resumo
resumo_financeiro = pd.merge(receita_por_gestao, despesa_por_gestao, on='Gestao', how='outer').fillna(0)

# Cria coluna de Saldo
resumo_financeiro['Saldo_Bruto'] = resumo_financeiro['Total_Receita'] - resumo_financeiro['Total_Despesa']

# Ordena por ano aproximado (manualmente para ficar cronológico no gráfico)
ordem_cronologica = [
    "Juvenal Juvêncio", 
    "Carlos Miguel Aidar (Transição JJ)", 
    "Carlos Miguel Aidar (Transição Leco)", 
    "Leco", 
    "Julio Casares"
]
# Transforma a coluna Gestao em categoria ordenada
resumo_financeiro['Gestao'] = pd.Categorical(resumo_financeiro['Gestao'], categories=ordem_cronologica, ordered=True)
resumo_financeiro = resumo_financeiro.sort_values('Gestao')

print("--- TABELA DESCRITIVA: TOTAIS ACUMULADOS (EM MILHÕES DE €) ---")
print(resumo_financeiro)
print("-" * 50)

# --- 2. ESTATÍSTICA DESCRITIVA DETALHADA (MÉDIA E DESVIO PADRÃO) ---
# Vamos ver a VOLATILIDADE das receitas (Desvio Padrão)
estatisticas = df_receitas.groupby('Gestao')['Valor_Milhoes'].agg(['mean', 'std', 'min', 'max']).reset_index()
print("\n--- ESTATÍSTICAS DE RECEITA ANUAL (CONSISTÊNCIA) ---")
print(estatisticas)

# --- 3. VISUALIZAÇÃO GRÁFICA (BAR PLOT AGRUPADO) ---
# Derreter (Melt) o dataframe para formato longo (necessário para o Seaborn fazer barras agrupadas)
df_melted = resumo_financeiro.melt(id_vars="Gestao", value_vars=["Total_Receita", "Total_Despesa"], var_name="Tipo", value_name="Valor_Milhoes")

plt.figure(figsize=(14, 7))
grafico = sns.barplot(
    data=df_melted, 
    x="Gestao", 
    y="Valor_Milhoes", 
    hue="Tipo", 
    palette=["#2ecc71", "#e74c3c"] # Verde para Receita, Vermelho para Despesa
)

plt.title('Comparativo Financeiro: Arrecadação vs. Gastos por Gestão', fontsize=16, fontweight='bold')
plt.ylabel('Valor Total (Milhões de Euros)', fontsize=12)
plt.xlabel('Gestão', fontsize=12)
plt.xticks(rotation=45)
plt.legend(title='Indicador')

# Adiciona valores no topo das barras
for container in grafico.containers:
    grafico.bar_label(container, fmt='%.1f', padding=3)

plt.tight_layout()
plt.show()