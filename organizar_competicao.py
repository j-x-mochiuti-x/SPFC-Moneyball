import pandas as pd
import numpy as np

# --- CONFIGURAÇÃO MANUAL ---
# Edite aqui o nome do arquivo que você quer processar
ARQUIVO_ALVO = "SPFC_Jogos_Julio_Casares__2021-Presente__tratado.csv"
# ---------------------------

print(f"--- PROCESSANDO E LIMPANDO NOMES: {ARQUIVO_ALVO} ---\n")

try:
    df = pd.read_csv(ARQUIVO_ALVO, sep=';', encoding='utf-8-sig')
    
    linhas_processadas = []
    competicao_atual = "Desconhecido" 
    
    # Função auxiliar para limpar o nome do time
    # Ela pega "São Paulo (5.)" e devolve "São Paulo"
    def limpar_nome(nome_sujo):
        if pd.isna(nome_sujo):
            return nome_sujo
        # Converte para texto, divide no " (" e pega a primeira parte
        return str(nome_sujo).split(' (')[0].strip()

    # Loop linha a linha
    for index, row in df.iterrows():
        
        # Lógica para identificar Título de Campeonato (igual ao anterior)
        valor_coluna_0 = row['Time da casa']
        
        if pd.notna(valor_coluna_0) and isinstance(valor_coluna_0, str) and len(valor_coluna_0) > 3:
            competicao_atual = valor_coluna_0
            
        else:
            # É LINHA DE JOGO
            if pd.notna(row['Resultado']):
                
                # --- AQUI ESTÁ A NOVIDADE ---
                # Aplicamos a limpeza antes de salvar
                mandante_limpo = limpar_nome(row['Time da casa.1'])
                visitante_limpo = limpar_nome(row['Time visitante.1'])
                
                novo_registro = {
                    'Competicao': competicao_atual,
                    'Mandante': mandante_limpo,    # Nome limpo
                    'Visitante': visitante_limpo,  # Nome limpo
                    'Resultado': row['Resultado'],
                    'Gestao': row['REF_GESTAO']
                }
                
                linhas_processadas.append(novo_registro)

    # Cria o DataFrame final
    df_final = pd.DataFrame(linhas_processadas)
    
    # Salva o arquivo final
    novo_nome = ARQUIVO_ALVO.replace("_tratado.csv", "_FINAL.csv")
    df_final.to_csv(novo_nome, sep=';', index=False, encoding='utf-8-sig')
    
    print("Processamento concluído!")
    print(f"Arquivo salvo como: {novo_nome}")
    print(f"Total de jogos processados: {len(df_final)}")
    
    print("\n--- AMOSTRA DOS NOMES LIMPOS ---")
    # Mostra só os times para você conferir se os parênteses sumiram
    print(df_final[['Mandante', 'Visitante']].head(10))

except FileNotFoundError:
    print(f"Erro: Arquivo '{ARQUIVO_ALVO}' não encontrado.")
except Exception as e:
    print(f"Erro inesperado: {e}")