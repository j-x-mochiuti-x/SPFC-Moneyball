#  dados-SPFC
# ⚽ SPFC Moneyball: Análise de Eficiência Financeira e Esportiva (2006-2024)

👨‍💻 Desenvolvido por:
## João Otávio Mochiuti

[LinkedIn](www.linkedin.com/in/joao-otavio-mochiuti)


![Python](https://img.shields.io/badge/Python-3.10-blue)
![Pandas](https://img.shields.io/badge/Pandas-Data_Analysis-green)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-orange)
![Status](https://img.shields.io/badge/Status-Concluído-success)

> *"Dinheiro ganha jogo?"* — Uma análise de dados sobre 18 anos de gestão do São Paulo Futebol Clube, cruzando balanços financeiros auditados com performance em campo.

---

## 🎯 O Problema de Negócio
No futebol moderno, existe uma crença comum de que "gastar mais = ganhar mais". O objetivo deste projeto foi utilizar Ciência de Dados para auditar a eficiência das últimas gestões do SPFC, respondendo a perguntas cruciais:
1. Qual presidente teve o **pior Custo-Benefício** (custo por ponto ganho)?
2. A gestão atual (Casares) é financeiramente saudável?
3. Existe correlação matemática forte entre Despesa Anual e Aproveitamento (%)?

---

## 📊 Principais Insights & Visualizações

### 1. A Matriz de Eficiência (Scatter Plot)
Cruzamos o **Investimento Real (Eixo X)** com o **Aproveitamento de Pontos (Eixo Y)**. O tamanho das bolhas representa a Receita Total.

<img width="1365" height="655" alt="image" src="https://github.com/user-attachments/assets/058bf68d-873b-4bf0-93cb-2cd04f0f77d8" />


**Descobertas:**
* **A "Máquina de Queimar Dinheiro":** A gestão Leco (2015-2020) situa-se isolada no quadrante "Caro e Ineficiente". O clube gastou muito acima da média para obter apenas 51% de aproveitamento.
* **Retomada da Eficiência:** A gestão atual (Casares) conseguiu retornar ao **Custo por Ponto (€0.06M)** da Era de Ouro (Juvenal 2006-2008), gastando 3x menos por ponto que a gestão anterior. **`Levando em consideração que na gestão Casares é marcada por casos de corrupção escancaradas, incluindo novamente o Douglas Schwartzmann. Este mesmo que estava no escândalo de corrupção envolvendo o São Paulo Futebol Clube e a fornecedora de material esportivo Under Armour em 2015, onde o então presidente Carlos Miguel Aidar e sua namorada foram acusados de receber comissões ilegais (um suposto valor de R$ 18 milhões).`**
---

### 2. A Realidade Financeira com Compra e Venda de Jogadores (Linha do Tempo)
Análise temporal da Receita vs. Despesa, ajustada pela inflação (**IPCA Zona do Euro**).

<img width="1358" height="650" alt="image" src="https://github.com/user-attachments/assets/297b073b-7c76-4f50-9751-c77835f8519f" />


**Descobertas:**
* Visualiza-se um superávit consistente nas janelas de transferência (Receita de Vendas > Despesa de Compras). O fato de o clube gerar caixa com atletas e, paradoxalmente, acumular uma dívida global próxima a R$ 1 Bilhão, sugere que o dinheiro das vendas não é reinvestido no elenco, mas sim utilizado para cobrir "rombos" operacionais (salários inflacionados, juros bancários). A necessidade atual de recorrer a FIDC confirma que o modelo de "vender o almoço para pagar a janta" se tornou insustentável, apesar dos bons números no gráfico de transferências.

---

### 3. Correlação: Dinheiro x Resultado (Heatmap)
Matriz de correlação de Pearson para validar estatisticamente se gastar mais por jogo garante vitórias.

<img width="1187" height="643" alt="image" src="https://github.com/user-attachments/assets/66e4fca8-041e-4fb8-bd43-dce10e2a8209" />


**Descobertas:**
* A correlação entre **Gasto por Jogo** e **% de Vitórias** indicou  -0.25 uma correlação negativa fraca. Estatisticamente, isso demonstra que aumentar o investimento financeiro por jogo não resultou em mais vitórias no período analisado. Pelo contrário, houve momentos de alto gasto e baixo retorno. Isso refuta a hipótese de que "o time perdeu porque faltou investimento" e fortalece a tese de ineficiência na alocação de recursos. Gestão técnica pesou mais que o volume financeiro (Correlação não implica causalidade, mas aponta uma tendência clara de desperdício)

---

## 🛠️ Stack Tecnológico e Metodologia

* **Linguagem:** Python
* **Bibliotecas:**
    * `Pandas`: Limpeza de dados (Data Cleaning), manipulação de DataFrames, Regex para tratamento de valores monetários.
    * `Matplotlib` & `Seaborn`: Visualização de dados e storytelling visual.
    * `NumPy`: Cálculos vetoriais.

**Destaque Técnico (Data Engineering):**
Para garantir uma comparação justa entre 2006 e 2024, foi aplicado um **Deflator Econômico** baseado no índice oficial de inflação da Zona do Euro (HICP). Um milhão de euros gasto em 2006 tem um peso (poder de compra) diferente de um milhão em 2024. Sem esse ajuste, a análise seria enviesada.

---

## 📂 Estrutura do Projeto

* `data/`: Contém os CSVs brutos (Jogos e Financeiro) e o Dataset Mestre tratado.
* `fonte de dados/`: Todos os dados foram retirados do site https://www.transfermarkt.com.br/.
* `scripts/`: Scripts Python modulares para geração dos gráficos.
* `SPFC_DATASET_MESTRE_AJUSTADO.csv`: O arquivo final enriquecido com KPIs (Custo/Ponto, ROI Esportivo).

---

## 🤝 Agradecimentos e Créditos
Este projeto foi desenvolvido com suporte de ferramentas de **Inteligência Artificial Generativa (Google Gemini)**, atuando como "Pair Programmer" para:
* Refatoração e otimização de scripts Python.
* Brainstorming de hipóteses de negócio.
* Revisão de conceitos estatísticos (Correlação e Ajuste Inflacionário).

A estruturação da análise, a curadoria dos dados e a interpretação final dos insights de negócio são de autoria do cientista de dados responsável (Eu kkkkk)

---

## 🚀 Como Executar

1. Clone o repositório:
```bash
git clone [https://github.com/j-x-mochiuti-x/dados-SPFC/](https://github.com/j-x-mochiuti-x/dados-SPFC/)
