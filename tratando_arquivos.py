import pandas as pd
import os

# --- CONFIGURAÇÃO MANUAL (Edite aqui para cada presidente) ---
# Coloque aqui o nome exato do arquivo que você quer processar agora
ARQUIVO_ALVO = "SPFC_Jogos_Julio_Casares__2021-Presente_.csv" 

# Lista de colunas que serão apagadas
COLUNAS_PARA_REMOVER = [
    'Rodada',
    'Data', 
    'Horário', 
    'Sistema de jogo', 
    'Treinadores', 
    'Público'
]
# ------------------------------------------------------------

print(f"--- INICIANDO TRATAMENTO MANUAL: {ARQUIVO_ALVO} ---\n")

try:
    # 1. Carregar o arquivo
    print("1. Lendo arquivo original...")
    df = pd.read_csv(ARQUIVO_ALVO, sep=';', encoding='utf-8-sig')
    
    # Mostra quantas colunas tinha antes
    cols_antes = list(df.columns)
    print(f"   Colunas encontradas ({len(cols_antes)}): {cols_antes}")

    # 2. Remover as colunas
    print("\n2. Removendo colunas indesejadas...")
    # errors='ignore' serve para o código não travar se uma coluna já tiver sido apagada antes
    df_limpo = df.drop(columns=COLUNAS_PARA_REMOVER, errors='ignore')

    # 3. Salvar o novo arquivo
    novo_nome = ARQUIVO_ALVO.replace(".csv", "_tratado.csv")
    df_limpo.to_csv(novo_nome, sep=';', index=False, encoding='utf-8-sig')
    
    print("\n3. Salvando resultado...")
    print(f"   Arquivo salvo com sucesso: {novo_nome}")
    
    # 4. Relatório final para você conferir antes de abrir o Excel
    print("\n--- RESUMO DO ARQUIVO GERADO ---")
    print(f"Colunas restantes: {list(df_limpo.columns)}")
    print("Primeira linha dos dados (Amostra):")
    # O .to_dict() imprime de um jeito fácil de ler no terminal
    print(df_limpo.iloc[0].to_dict()) 
    print("\n------------------------------------------------")
    print("Pode abrir o Excel para conferir!")

except FileNotFoundError:
    print(f"\nERRO: O arquivo '{ARQUIVO_ALVO}' não foi encontrado na pasta.")
    print("Verifique se o nome está escrito corretamente (incluindo o .csv).")
except Exception as e:
    print(f"\nERRO INESPERADO: {e}")