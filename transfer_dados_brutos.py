import requests
import pandas as pd
from bs4 import BeautifulSoup
import time
from io import StringIO

# --- CONFIGURAÇÃO ---
ANO_INICIAL = 2006
ANO_FINAL = 2026
URL_BASE = "https://www.transfermarkt.com.br/sao-paulo-fc/transfers/verein/585/plus/?saison_id={}"

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

lista_balancos = []

print(f"--- INICIANDO COLETA FINANCEIRA ({ANO_INICIAL} a {ANO_FINAL}) ---\n")

for ano in range(ANO_INICIAL, ANO_FINAL + 1):
    url = URL_BASE.format(ano)
    print(f"Processando temporada: {ano}...", end="")
    
    try:
        response = requests.get(url, headers=HEADERS)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Estratégia: Procurar o H2 que tem o texto "Balanço de transferências"
            titulos = soup.find_all("h2", class_="content-box-headline")
            
            tabela_encontrada = False
            
            for h2 in titulos:
                texto_titulo = h2.get_text(strip=True)
                
                if "Balanço" in texto_titulo:
                    box_pai = h2.find_parent("div", class_="box")
                    
                    if box_pai:
                        tabela = box_pai.find("table")
                        
                        if tabela:
                            # Converte para DataFrame
                            df_temp = pd.read_html(StringIO(str(tabela)))[0]
                            
                            # Adiciona a coluna do Ano para não nos perdermos depois
                            df_temp['TEMPORADA_REF'] = ano
                            
                            lista_balancos.append(df_temp)
                            tabela_encontrada = True
                            print(" -> OK! Tabela capturada.")
                            break # Para de procurar outros H2 na mesma página
            
            if not tabela_encontrada:
                print(" -> Aviso: Tabela de balanço não encontrada nesta página.")
                
        else:
            print(f" -> Erro HTTP {response.status_code}")
            
    except Exception as e:
        print(f" -> Erro técnico: {e}")
    
    # Pausa respeitosa para o servidor
    time.sleep(2)

# --- CONSOLIDAÇÃO ---
print("\n" + "="*40)
if lista_balancos:
    df_financeiro = pd.concat(lista_balancos, ignore_index=True)
    
    nome_arquivo = "spfc_financeiro_bruto_anual.csv"
    df_financeiro.to_csv(nome_arquivo, sep=';', index=False, encoding='utf-8-sig')
    
    print(f"SUCESSO! Arquivo salvo: {nome_arquivo}")
    print(f"Linhas capturadas: {len(df_financeiro)}")
    print("\nAMOSTRA DOS DADOS:")
    print(df_financeiro.head())
else:
    print("Falha: Nenhuma tabela foi capturada.")