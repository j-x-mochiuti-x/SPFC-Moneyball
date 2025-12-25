import pandas as pd
import os
import re

# 1. Carregar o CSV Bruto que criamos no passo anterior
# Certifique-se de que o nome do arquivo aqui é o mesmo que foi gerado antes
arquivo_origem = "sao_paulo_jogos_bruto.csv"

print(f"Lendo o arquivo mestre: {arquivo_origem}...")

try:
    # Lendo com o separador ';' e encoding utf-8-sig (para acentos)
    df = pd.read_csv(arquivo_origem, sep=';', encoding='utf-8-sig')
    
    # 2. Identificar a coluna que tem o nome do Presidente
    coluna_filtro = 'REF_GESTAO'
    
    if coluna_filtro not in df.columns:
        print(f"ERRO: Não encontrei a coluna '{coluna_filtro}'. As colunas disponíveis são:")
        print(df.columns)
    else:
        # Pega a lista única de presidentes que existem no arquivo (sem repetir)
        lista_presidentes = df[coluna_filtro].unique()
        
        print(f"\nPresidentes encontrados no arquivo: {len(lista_presidentes)}")
        
        # 3. Loop para criar um arquivo para cada um
        for presidente in lista_presidentes:
            
            # Filtra apenas as linhas daquele presidente
            df_presidente = df[df[coluna_filtro] == presidente]
            
            # Limpeza do nome do arquivo (para não ter parenteses ou espaços no nome do arquivo)
            nome_limpo = re.sub(r'[^\w\-]', '_', presidente) 
            nome_arquivo_final = f"SPFC_Jogos_{nome_limpo}.csv"
            
            # Salva o novo CSV
            df_presidente.to_csv(nome_arquivo_final, sep=';', index=False, encoding='utf-8-sig')
            
            print(f" -> Arquivo criado: {nome_arquivo_final} ({len(df_presidente)} jogos)")

        print("\nProcesso concluído! Verifique sua pasta.")

except FileNotFoundError:
    print(f"ERRO: O arquivo '{arquivo_origem}' não foi encontrado na pasta. Verifique o nome.")
except Exception as e:
    print(f"Ocorreu um erro inesperado: {e}")