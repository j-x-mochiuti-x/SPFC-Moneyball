import requests
import pandas as pd
import time
from io import StringIO

# 1. Configuração (Headers para não ser bloqueado)
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# 2. Períodos de Gestão (Formato exigido pela URL: DD.MM.YYYY)
mandatos = [
    {"Presidente": "Juvenal Juvêncio (2006-2014)", "inicio": "01.01.2006", "fim": "16.04.2014"},
    {"Presidente": "Carlos Miguel Aidar (2014-2015)", "inicio": "17.04.2014", "fim": "13.10.2015"},
    {"Presidente": "Leco (2015-2020)", "inicio": "14.10.2015", "fim": "31.12.2020"},
    {"Presidente": "Julio Casares (2021-Presente)", "inicio": "01.01.2021", "fim": "31.12.2025"}
]

lista_dfs = []

print("--- INICIANDO EXTRAÇÃO BRUTA (RAW DATA) ---\n")

for mandato in mandatos:
    # A URL 
    url = f"https://www.transfermarkt.com.br/sao-paulo-fc/spielplandatum/verein/585/plus/1?saison_id=2024&wettbewerb_id=&day=&heim_gast=&punkte=&datum_von={mandato['inicio']}&datum_bis={mandato['fim']}"
    
    print(f"Extraindo: {mandato['Presidente']}...")
    
    try:
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            html = StringIO(response.text)
            # Lê TODAS as tabelas da página
            dfs = pd.read_html(html, flavor='html5lib')
            
            tabela_alvo = None
            
            # Lógica para encontrar a tabela certa baseada nas colunas que você quer
            # Procuramos uma tabela que tenha 'Data', 'Horário' ou 'Público' no cabeçalho
            for df in dfs:
                colunas_str = [str(c).lower() for c in df.columns]
                if any("data" in c for c in colunas_str) and any("público" in c for c in colunas_str):
                    tabela_alvo = df
                    break
            
            # Se não achar por nome, pega a maior tabela da página (fallback de segurança)
            if tabela_alvo is None and len(dfs) > 0:
                tabela_alvo = max(dfs, key=len)
            
            if tabela_alvo is not None:
                # Adiciona coluna identificadora da gestão (Metadado essencial)
                tabela_alvo['REF_GESTAO'] = mandato['Presidente']
                
                lista_dfs.append(tabela_alvo)
                print(f" -> OK: {len(tabela_alvo)} linhas coletadas.")
            else:
                print(" -> AVISO: Nenhuma tabela compatível encontrada.")
                
        else:
            print(f" -> ERRO HTTP: {response.status_code}")
            
    except Exception as e:
        print(f" -> ERRO CRÍTICO: {e}")
    
    # Pausa para não bloquear o IP
    time.sleep(3)

# 3. Consolidação e Salvamento
if lista_dfs:
    print("\nConsolidando arquivo final...")
    df_raw = pd.concat(lista_dfs, ignore_index=True)
    
    nome_arquivo = "sao_paulo_jogos_bruto.csv"
    
    # Salva com encoding 'utf-8-sig' para o Excel abrir acentos corretamente
    # Usa separador ';' para o Excel dividir as colunas automaticamente
    df_raw.to_csv(nome_arquivo, index=False, sep=';', encoding='utf-8-sig')
    
    print(f"\nSUCESSO! Arquivo gerado: {nome_arquivo}")
    print(f"Total de registros: {len(df_raw)}")
    print("Nota: Como é um CSV bruto, espere colunas como 'Unnamed' (logos) e dados sujos.")
else:
    print("\nFALHA: Nada foi coletado.")