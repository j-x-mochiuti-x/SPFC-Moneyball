import pandas as pd

# --- CONFIGURAÇÃO MANUAL ---
# Use o arquivo que saiu do passo anterior (o _FINAL)
ARQUIVO_ENTRADA = "SPFC_Jogos_Julio_Casares__2021-Presente__FINAL.csv"
# ---------------------------

print(f"--- CALCULANDO RESULTADOS: {ARQUIVO_ENTRADA} ---\n")

try:
    df = pd.read_csv(ARQUIVO_ENTRADA, sep=';', encoding='utf-8-sig')
    
    resultados_processados = []

    for index, row in df.iterrows():
        # 1. Quebra o placar "2:1" em dois números inteiros
        # Esquerda = Gols do Mandante | Direita = Gols do Visitante
        placar = str(row['Resultado']).strip()
        
        try:
            gols_mandante = int(placar.split(':')[0])
            gols_visitante = int(placar.split(':')[1])
        except:
            # Se o placar estiver errado ou for texto, pula
            continue

        mandante = str(row['Mandante'])
        visitante = str(row['Visitante'])
        
        status = "Erro"
        gols_pro = 0
        gols_contra = 0

        # --- LÓGICA DO SÃO PAULO ---
        
        # CASO 1: SÃO PAULO É O MANDANTE (CASA)
        # Procuramos "São Paulo" no nome do mandante
        if "São Paulo" in mandante:
            gols_pro = gols_mandante
            gols_contra = gols_visitante
            
            if gols_mandante > gols_visitante:
                status = "Vitória"
            elif gols_mandante < gols_visitante:
                status = "Derrota"
            else:
                status = "Empate"

        # CASO 2: SÃO PAULO É O VISITANTE (FORA)
        elif "São Paulo" in visitante:
            gols_pro = gols_visitante      # O gol do SPFC é o da direita
            gols_contra = gols_mandante    # O gol do adversário é o da esquerda
            
            if gols_visitante > gols_mandante:
                status = "Vitória"
            elif gols_visitante < gols_mandante:
                status = "Derrota"
            else:
                status = "Empate"
        
        else:
            # Caso raro onde o SPFC não está no nome (ex: erro de digitação no site)
            status = "Não Identificado"

        # Cria a nova linha com os dados calculados
        row['Gols_Pro'] = gols_pro
        row['Gols_Contra'] = gols_contra
        row['Status_Jogo'] = status
        
        resultados_processados.append(row)

    # Cria DataFrame final
    df_calculado = pd.DataFrame(resultados_processados)

    # Salva
    novo_nome = ARQUIVO_ENTRADA.replace("_FINAL.csv", "_CALCULADO.csv")
    df_calculado.to_csv(novo_nome, sep=';', index=False, encoding='utf-8-sig')

    print("Cálculos finalizados!")
    print(f"Arquivo salvo como: {novo_nome}")
    
    # --- RESUMO RÁPIDO PARA VOCÊ CONFERIR NO TERMINAL ---
    print("\n--- RESUMO DA GESTÃO ---")
    contagem = df_calculado['Status_Jogo'].value_counts()
    print(contagem)
    
    # Exemplo de cálculo de aproveitamento simples
    vitorias = contagem.get('Vitória', 0)
    empates = contagem.get('Empate', 0)
    derrotas = contagem.get('Derrota', 0)
    total = vitorias + empates + derrotas
    
    if total > 0:
        aproveitamento = ((vitorias * 3) + empates) / (total * 3) * 100
        print(f"\nAproveitamento Estimado: {aproveitamento:.2f}%")
        print(f"Jogos Totais: {total}")

except FileNotFoundError:
    print(f"ERRO: Não achei o arquivo {ARQUIVO_ENTRADA}")
except Exception as e:
    print(f"Erro inesperado: {e}")