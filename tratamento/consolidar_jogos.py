import pandas as pd
import glob
import os

# --- CONFIGURAÇÃO ---
# O código vai procurar qualquer arquivo que termine com "FINAL.csv" na pasta
# Certifique-se de que os 4 arquivos dos presidentes estão na pasta com esse sufixo!!!!
PADRAO_ARQUIVOS = "*_FINAL.csv" 
ARQUIVO_SAIDA = "SPFC_JOGOS_COMPLETOS_CONSOLIDADO.csv"

# Lista para guardar os dataframes processados
lista_dfs = []

# Busca os arquivos na pasta
arquivos_encontrados = glob.glob(PADRAO_ARQUIVOS)

print(f"--- INICIANDO CONSOLIDAÇÃO DE {len(arquivos_encontrados)} ARQUIVOS ---\n")

for arquivo in arquivos_encontrados:
    # Pula o arquivo financeiro se ele cair na busca por engano
    if "financeiro" in arquivo.lower():
        continue
        
    print(f"Processando: {arquivo}...")
    
    try:
        # Lê o arquivo individual
        df = pd.read_csv(arquivo, sep=';', encoding='utf-8-sig')
        
        # --- APLICAÇÃO DA LÓGICA DE VITÓRIA/DERROTA (Para cada arquivo) ---
        resultados_processados = []
        
        for index, row in df.iterrows():
            # Tratamento do placar (ex: "2:1")
            try:
                placar = str(row['Resultado']).strip()
                if ':' not in placar:
                    continue # Pula jogos sem resultado (adiados/cancelados)
                    
                gols_mandante = int(placar.split(':')[0])
                gols_visitante = int(placar.split(':')[1])
            except:
                continue

            mandante = str(row['Mandante'])
            visitante = str(row['Visitante'])
            
            status = "Erro"
            gols_pro = 0
            gols_contra = 0

            # Lógica SPFC
            if "São Paulo" in mandante:
                gols_pro = gols_mandante
                gols_contra = gols_visitante
                if gols_mandante > gols_visitante: status = "Vitória"
                elif gols_mandante < gols_visitante: status = "Derrota"
                else: status = "Empate"

            elif "São Paulo" in visitante:
                gols_pro = gols_visitante
                gols_contra = gols_mandante
                if gols_visitante > gols_mandante: status = "Vitória"
                elif gols_visitante < gols_mandante: status = "Derrota"
                else: status = "Empate"
            
            else:
                status = "Não Identificado"

            # Atualiza a linha
            row['Gols_Pro'] = gols_pro
            row['Gols_Contra'] = gols_contra
            row['Status_Jogo'] = status
            
            resultados_processados.append(row)
        
        # Cria o DF processado deste presidente específico
        df_temp = pd.DataFrame(resultados_processados)
        
        # Adiciona na lista geral
        lista_dfs.append(df_temp)
        print(f"   -> {len(df_temp)} jogos adicionados.")

    except Exception as e:
        print(f"   -> ERRO ao ler {arquivo}: {e}")

# --- CONSOLIDAÇÃO FINAL ---
print("\n" + "="*40)
if lista_dfs:
    # O comando concat junta tudo, um embaixo do outro
    df_geral = pd.concat(lista_dfs, ignore_index=True)
    
    # Salva o arquivo mestre
    df_geral.to_csv(ARQUIVO_SAIDA, sep=';', index=False, encoding='utf-8-sig')
    
    print(f"SUCESSO! Base consolidada criada: {ARQUIVO_SAIDA}")
    print(f"Total de Jogos na Base Histórica: {len(df_geral)}")
    
    print("\n--- RESUMO GERAL (TODAS AS GESTÕES) ---")
    print(df_geral['Status_Jogo'].value_counts())
    print("\nDivisão por Gestão:")
    print(df_geral['Gestao'].value_counts())
    
else:
    print("Nenhum arquivo encontrado. Verifique se os nomes terminam em '_FINAL.csv'")