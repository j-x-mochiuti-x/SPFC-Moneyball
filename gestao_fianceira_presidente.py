import pandas as pd
from datetime import datetime

# --- CONFIGURAÇÃO ---
ARQUIVO_ENTRADA = "spfc_financeiro_bruto_anual.csv"

# --- DEFINIÇÃO DOS PRESIDENTES ---
def definir_presidente(ano_referencia):
    try:
        ano = int(ano_referencia)
    except:
        return "Desconhecido"

    if 2006 <= ano <= 2013:
        return "Juvenal Juvêncio"
    elif ano == 2014:
        return "Carlos Miguel Aidar (Transição JJ)" 
    elif ano == 2015:
        return "Carlos Miguel Aidar (Transição Leco)"
    elif 2016 <= ano <= 2020:
        return "Leco"
    elif 2021 <= ano <= 2025:
        return "Julio Casares"
    else:
        return "Outros/Antigos"

# --- FUNÇÃO DE LIMPEZA DE VALOR ---
def limpar_valor(valor_str):
    if pd.isna(valor_str) or valor_str == '-':
        return 0.0
    
    s = str(valor_str).lower().strip()
    
    fator = 1.0
    if 'mi.' in s:
        fator = 1_000_000.0
        s = s.replace('mi.', '')
    elif 'mil' in s:
        fator = 1_000.0
        s = s.replace('mil', '')
        
    s = s.replace('€', '').replace('balanced', '').strip()
    s = s.replace('.', '') # Tira ponto de milhar
    s = s.replace(',', '.') # Põe ponto decimal
    
    try:
        return float(s) * fator
    except ValueError:
        return 0.0

print(f"--- CORRIGINDO E PROCESSANDO: {ARQUIVO_ENTRADA} ---\n")

try:
    # 1. Carregar arquivo
    df = pd.read_csv(ARQUIVO_ENTRADA, sep=';', encoding='utf-8-sig')
    
    print(f"Colunas encontradas originalmente: {len(df.columns)}")
    print(f"Nomes das colunas: {list(df.columns)}")

    # 2. SELEÇÃO CIRÚRGICA (A CORREÇÃO ESTÁ AQUI)
    df = df.iloc[:, [0, 1, -1]]
    df.columns = ['Tipo_Movimentacao', 'Valor_Texto', 'Ano']

    # 3. Aplicar as limpezas
    print("\nCalculando valores reais...")
    df['Valor_Numerico'] = df['Valor_Texto'].apply(limpar_valor)
    
    print("Identificando presidentes...")
    df['Gestao'] = df['Ano'].apply(definir_presidente)
    
    # 4. Filtragem Final
    filtro_validos = df['Tipo_Movimentacao'].astype(str).str.contains('Receita|Despesa|Balanço|Entrada|Saída', case=False, na=False)
    df_final = df[filtro_validos].copy()

    # Salvar
    nome_saida = "spfc_financeiro_FINAL.csv"
    df_final.to_csv(nome_saida, sep=';', index=False, encoding='utf-8-sig')

    print(f"\nSUCESSO! Arquivo salvo como: {nome_saida}")
    print("\n--- AMOSTRA DOS DADOS ---")
    print(df_final[['Ano', 'Gestao', 'Tipo_Movimentacao', 'Valor_Numerico']].head())

except FileNotFoundError:
    print(f"Erro: O arquivo {ARQUIVO_ENTRADA} não foi encontrado.")
except Exception as e:
    print(f"Erro inesperado: {e}")