import pandas as pd

# --- CONFIGURAÇÃO ---
ARQUIVO_ENTRADA = "spfc_financeiro_bruto_anual.csv"
ARQUIVO_SAIDA = "spfc_financeiro_FINAL_MILHOES.csv"

# --- 1. LÓGICA DE VALORES CORRIGIDA (CONDICIONAL) ---
def limpar_valor_milhoes(valor_str):
    if pd.isna(valor_str) or str(valor_str).strip() in ['-', '?', 'nan']:
        return 0.0
    
    # Padroniza para minúsculo e remove o símbolo de Euro
    s = str(valor_str).lower().replace('€', '').strip()
    
    fator = 1.0
    
    # Lógica: O objetivo é transformar tudo em MILHÕES
    if 'mi' in s:
        fator = 1.0 
        s = s.replace('mi.', '').replace('mi', '')
        
    elif 'mil' in s:
        fator = 0.001
        s = s.replace('mil', '')
    
    # Limpeza de texto extra
    s = s.replace('balanced', '').strip()
    
    # --- O PULO DO GATO (CORREÇÃO DO ERRO DE ESCALA) ---
    if ',' in s:
        s = s.replace('.', '')  # Remove ponto de milhar
        s = s.replace(',', '.') # Transforma vírgula em ponto decimal
    
    # Se não tem vírgula, ele não faz nada com os pontos, preservando o "14.98"
    
    try:
        valor_final = float(s) * fator
        return round(valor_final, 3)
    except ValueError:
        return 0.0

# --- 2. LÓGICA DE PRESIDENTES ---
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

print(f"--- PROCESSANDO VALORES (CORREÇÃO DE ESCALA) ---\n")

try:
    # Lê o arquivo bruto
    df = pd.read_csv(ARQUIVO_ENTRADA, sep=';', encoding='utf-8-sig')
    
    # CORREÇÃO DE COLUNAS
    # Seleciona: Tipo (0), Valor Original (2) e Ano (-1/Último)
    df = df.iloc[:, [0, 2, -1]]
    df.columns = ['Tipo', 'Valor_Original', 'Ano']
    
    # Aplica a limpeza de valores corrigida
    print("Convertendo valores para Milhões de Euros...")
    df['Valor_Milhoes'] = df['Valor_Original'].apply(limpar_valor_milhoes)
    
    # Aplica a definição de gestão
    print("Atribuindo Presidentes...")
    df['Gestao'] = df['Ano'].apply(definir_presidente)
    
    # Filtra apenas linhas úteis
    filtro = df['Tipo'].astype(str).str.contains('Receita|Despesa|Balanço|Entrada|Saída', case=False, na=False)
    df_final = df[filtro].copy()
    
    # Salva o arquivo final
    df_final.to_csv(ARQUIVO_SAIDA, sep=';', index=False, encoding='utf-8-sig')
    
    print(f"\nSUCESSO! Arquivo gerado: {ARQUIVO_SAIDA}")
    print("-" * 50)
    print("AMOSTRA DOS DADOS (Verifique se 14.98 mi virou 14.98):")
    # Mostra colunas específicas para checagem
    print(df_final[['Valor_Original', 'Valor_Milhoes']].head(10))
    
    # Teste Rápido
    val_teste = df_final['Valor_Milhoes'].iloc[0] # Pega o primeiro valor
    print(f"\nVerificação manual do primeiro valor: {val_teste}")
    if val_teste > 1000:
        print("ALERTA: O valor ainda parece alto demais! Verifique a lógica.")
    else:
        print("Parece correto (escala de milhões mantida).")

except FileNotFoundError:
    print(f"Erro: Arquivo {ARQUIVO_ENTRADA} não encontrado.")
except Exception as e:
    print(f"Erro: {e}")