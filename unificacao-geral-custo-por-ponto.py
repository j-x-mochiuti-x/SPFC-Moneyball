import pandas as pd

# --- CONFIGURAÇÃO ---
ARQUIVO_JOGOS = "SPFC_JOGOS_COMPLETOS_CONSOLIDADO.csv"
ARQUIVO_FINANCEIRO = "spfc_financeiro_FINAL_MILHOES.csv"

print("--- GERAÇÃO FINAL COM CORREÇÃO MONETÁRIA (INFLAÇÃO EURO) ---\n")

try:
    # 1. CARREGAR DADOS
    df_jogos = pd.read_csv(ARQUIVO_JOGOS, sep=';', encoding='utf-8-sig')
    df_fin = pd.read_csv(ARQUIVO_FINANCEIRO, sep=';', encoding='utf-8-sig')

    # ---------------------------------------------------------
    # 2. APLICAÇÃO DA INFLAÇÃO (NOVA ETAPA)
    # ---------------------------------------------------------
    # Fatores multiplicadores aproximados para trazer valores ao poder de compra de 2024/2025
    # Fonte base: Eurostat HICP (Historical Inflation)
    # Ex: 1 Euro de 2006 equivale a aprox 1.42 Euros hoje.
    fator_inflacao_euro = {
        2006: 1.42, 2007: 1.39, 2008: 1.35, 2009: 1.34, 2010: 1.32,
        2011: 1.28, 2012: 1.25, 2013: 1.23, 2014: 1.22, 2015: 1.20,
        2016: 1.19, 2017: 1.17, 2018: 1.15, 2019: 1.14, 2020: 1.14,
        2021: 1.11, 2022: 1.05, 2023: 1.02, 2024: 1.00, 2025: 1.00
    }

    # Mapeia o fator baseado no Ano
    df_fin['Fator_Ajuste'] = df_fin['Ano'].map(fator_inflacao_euro).fillna(1.0)

    # Cria a coluna de Valor Corrigido (Valor Real)
    df_fin['Valor_Milhoes_Corrigido'] = df_fin['Valor_Milhoes'] * df_fin['Fator_Ajuste']
    
    print("Correção inflacionária aplicada com sucesso!")

    # ---------------------------------------------------------
    # 3. PADRONIZAÇÃO DE NOMES
    # ---------------------------------------------------------
    mapa_nomes_jogos = {
        "Julio Casares (2021-Presente)": "Julio Casares",
        "Leco (2015-2020)": "Leco",
        "Juvenal Juvêncio (2006-2014)": "Juvenal Juvêncio",
        "Carlos Miguel Aidar (2014-2015)": "Carlos Miguel Aidar" 
    }
    df_jogos['Gestao'] = df_jogos['Gestao'].replace(mapa_nomes_jogos)
    
    df_fin['Gestao'] = df_fin['Gestao'].replace({
        "Carlos Miguel Aidar (Transição JJ)": "Carlos Miguel Aidar",
        "Carlos Miguel Aidar (Transição Leco)": "Carlos Miguel Aidar"
    })

    # ---------------------------------------------------------
    # 4. CÁLCULOS ESPORTIVOS
    # ---------------------------------------------------------
    df_jogos['Vitoria_Num'] = df_jogos['Status_Jogo'].apply(lambda x: 1 if x == 'Vitória' else 0)
    df_jogos['Empate_Num'] = df_jogos['Status_Jogo'].apply(lambda x: 1 if x == 'Empate' else 0)
    
    kpi_esportivo = df_jogos.groupby('Gestao').agg({
        'Status_Jogo': 'count', 
        'Vitoria_Num': 'sum',
        'Empate_Num': 'sum',
        'Gols_Pro': 'sum',
        'Gols_Contra': 'sum'
    }).reset_index()
    
    kpi_esportivo.rename(columns={'Status_Jogo': 'Jogos_Totais'}, inplace=True)
    kpi_esportivo['Pontos_Ganhos'] = (kpi_esportivo['Vitoria_Num'] * 3) + kpi_esportivo['Empate_Num']
    kpi_esportivo['Aproveitamento_Pct'] = (kpi_esportivo['Pontos_Ganhos'] / (kpi_esportivo['Jogos_Totais'] * 3)) * 100

    # ---------------------------------------------------------
    # 5. CÁLCULOS FINANCEIROS
    # ---------------------------------------------------------
    df_fin_receita = df_fin[df_fin['Tipo'].str.contains('Receita', case=False)]
    df_fin_despesa = df_fin[df_fin['Tipo'].str.contains('Despesa|Saída', case=False)]

    # Note que agora somamos 'Valor_Milhoes_Corrigido'
    kpi_fin_rec = df_fin_receita.groupby('Gestao')['Valor_Milhoes_Corrigido'].sum().reset_index(name='Receita_Real_Mi')
    kpi_fin_des = df_fin_despesa.groupby('Gestao')['Valor_Milhoes_Corrigido'].sum().reset_index(name='Despesa_Real_Mi')
    
    kpi_financeiro = pd.merge(kpi_fin_rec, kpi_fin_des, on='Gestao', how='outer').fillna(0)

    # ---------------------------------------------------------
    # 6. MERGE FINAL E NOVOS KPIs
    # ---------------------------------------------------------
    df_final = pd.merge(kpi_esportivo, kpi_financeiro, on='Gestao', how='inner')
    
    # KPI 1: Custo Real por Ponto (Quanto custou cada ponto em valores de hoje?)
    df_final['Custo_Ponto_Real_Euro'] = df_final['Despesa_Real_Mi'] / df_final['Pontos_Ganhos']
    
    # KPI 2: Balanço Financeiro Real
    df_final['Saldo_Real_Mi'] = df_final['Receita_Real_Mi'] - df_final['Despesa_Real_Mi']

    # Arredondamento
    df_final = df_final.round(2)

    print("\n--- TABELA DEFINITIVA (VALORES AJUSTADOS PELA INFLAÇÃO) ---")
    colunas_vis = ['Gestao', 'Jogos_Totais', 'Despesa_Real_Mi', 'Pontos_Ganhos', 'Custo_Ponto_Real_Euro']
    print(df_final[colunas_vis])
    
    df_final.to_csv("SPFC_DATASET_MESTRE_AJUSTADO.csv", sep=';', index=False, encoding='utf-8-sig')
    print("\nArquivo salvo: SPFC_DATASET_MESTRE_AJUSTADO.csv")

except Exception as e:
    print(f"Erro: {e}")