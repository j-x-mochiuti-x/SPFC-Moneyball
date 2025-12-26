#  dados-SPFC
# ⚽ SPFC Moneyball: Análise de Eficiência Financeira e Esportiva (2006-2024)

👨‍💻 Desenvolvido por:
## João Otávio Mochiuti

[LinkedIn](www.linkedin.com/in/joao-otavio-mochiuti)


![Python](https://img.shields.io/badge/Python-3.10-blue)
![Pandas](https://img.shields.io/badge/Pandas-Data_Analysis-green)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-orange)
![Status](https://img.shields.io/badge/Status-Concluído-success)

> *"Sentimento Calculado: Dinheiro ganha jogo?" — Uma análise de dados sobre 18 anos de gestão do São Paulo Futebol Clube, cruzando balanços financeiros com performance em campo.

---

## 🎯 O Problema de Negócio
No futebol moderno, existe uma crença comum de que "gastar mais = ganhar mais". O objetivo deste projeto foi utilizar Ciência de Dados para auditar a eficiência das últimas gestões do SPFC, respondendo a perguntas cruciais:
1. Qual presidente teve o **pior Custo-Benefício** (custo por ponto ganho)?
2. A gestão atual (Casares) é financeiramente eficiente ou apenas reflete ciclos passados?
3. Existe correlação matemática forte entre Despesa Anual e Aproveitamento (%)?

---

## 📊 Principais Insights & Visualizações

### 1. A Matriz de Eficiência (Scatter Plot)
Cruzamos o **Investimento Real (Eixo X)** com o **Aproveitamento de Pontos (Eixo Y)**. O tamanho das bolhas representa a Receita Total.

<img width="1365" height="655" alt="image" src="https://github.com/user-attachments/assets/058bf68d-873b-4bf0-93cb-2cd04f0f77d8" />


**Descobertas:**
* **A "Máquina de Queimar Dinheiro":** A gestão Leco (2015-2020) situa-se isolada no quadrante "Caro e Ineficiente". O clube gastou muito acima da média para obter apenas 51% de aproveitamento.
* **Retomada da Eficiência s. Governança:** A gestão atual (Casares) conseguiu retornar ao **Custo por Ponto (€0.06M)** da Era de Ouro (Juvenal 2006-2008), gastando 3x menos por ponto que a gestão anterior. 
* **Ressalva Institucional da era Casares:**`Levando em consideração que na gestão Casares é marcada por casos de corrupção escancaradas, incluindo novamente o Douglas Schwartzmann. Este mesmo que estava no escândalo de corrupção envolvendo o São Paulo Futebol Clube e a fornecedora de material esportivo Under Armour em 2015, onde o então presidente Carlos Miguel Aidar e sua namorada foram acusados de receber comissões ilegais (um suposto valor de R$ 18 milhões). O que levanta questões sobre a sustentabilidade ética e transparência a longo prazo, apesar do ROI esportivo atual.`**
---

### 2. Fluxo de Caixa de Transferências (Linha do Tempo)
Análise temporal da Receita vs. Despesa, ajustada pela inflação (**HICP Zona do Euro**).

<img width="1358" height="650" alt="image" src="https://github.com/user-attachments/assets/297b073b-7c76-4f50-9751-c77835f8519f" />


**Descobertas:**
* **O Paradoxo da Dívida:** Nota-se um superávit consistente em transferências. O fato de o clube ser um "exportador" e ainda assim acumular dívida próxima a **R$ 1 Bilhão** comprova que o lucro das vendas não é reinvestido no futebol, mas drenado por rombos operacionais e juros. O recurso recente a FIDCs é o sintoma final de um modelo de "vender o almoço para pagar a janta".

---

### 3. Correlação: Dinheiro x Resultado (Heatmap)
Matriz de correlação de Pearson para validar estatisticamente se gastar mais por jogo garante vitórias.

<img width="1187" height="643" alt="image" src="https://github.com/user-attachments/assets/66e4fca8-041e-4fb8-bd43-dce10e2a8209" />


**Descobertas:**
* A correlação entre **Gasto por Jogo** e **% de Vitórias** indicou  -0.25 uma correlação negativa fraca. Estatisticamente, isso demonstra que aumentar o investimento financeiro por jogo não resultou em mais vitórias no período analisado. Pelo contrário, houve momentos de alto gasto e baixo retorno. Isso refuta a hipótese de que "o time perdeu porque faltou investimento" e fortalece a tese de ineficiência na alocação de recursos. Gestão técnica pesou mais que o volume financeiro (Correlação não implica causalidade, mas aponta uma tendência clara de desperdício)

### 3. Impacto Estatístico por Era (Variáveis Dummy)

Utilização de One-Hot Encoding para medir o impacto de cada gestão nas vitórias, tendo a "Era de Ouro" (Juvenal Juvêncio) como baseline.

![alt text](image.png)

* **Efeito Gestão:** Ao isolar as gestões via variáveis dummy, quantificamos que o "fator político/técnico" pesou mais nos resultados do que o volume financeiro disponível.
---

## 🛠️ Stack Tecnológico e Metodologia

* **Linguagem:** Python
* **Bibliotecas:**
    * `Pandas`: Limpeza de dados (Data Cleaning), manipulação de DataFrames, Regex para tratamento de valores monetários.
    * `Matplotlib` & `Seaborn`: Visualização de dados e storytelling visual.
    * `NumPy`: Cálculos vetoriais.

**⚙️ Rigor Técnico e Engenharia de Dados**
* **Deflator Econômico:** Aplicação do índice HICP (Zona do Euro) para normalizar valores de 2006 a 2024, garantindo que a análise reflita o poder de compra real de cada época.
* **Tratamento de Dummies (Drop First):** Para evitar a Dummy Variable Trap e a multicolinearidade perfeita, utilizei *n-1* categorias de gestão, permitindo uma análise de regressão estável e comparativa.* **Regex & Data Cleaning:** Tratamento de strings e conversão de câmbio automatizada via Python.

---

## 📂 Estrutura do Projeto

* `data/`: Contém os CSVs brutos (Jogos e Financeiro) e o Dataset Mestre tratado.
* `fonte de dados/`: Todos os dados foram retirados do site https://www.transfermarkt.com.br/.
* `scripts/`: Scripts Python modulares para geração dos gráficos.


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
