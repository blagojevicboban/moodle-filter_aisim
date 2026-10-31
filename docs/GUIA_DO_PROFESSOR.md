# AI for VET – Guia pedagógico e técnico para professores
## Integração de simulações interativas de inteligência artificial e aprendizagem automática no Moodle

*Projeto: **AI for VET** (Artificial Intelligence for Vocational Education and Training)*  
*Programa: **Erasmus+***  
*Portal oficial:* [https://aiforvet.eu/](https://aiforvet.eu/)  
*Simulações sem Moodle:* [https://sim.aiforvet.eu/](https://sim.aiforvet.eu/)  
*Autores e equipa técnica do projeto AI for VET*  
*Versão do guia: 1.1 (outubro de 2026)*  
*Compatibilidade: Moodle 4.5 LTS, 5.0, 5.1, 5.2, 5.3+*

---

## Índice
1. [Introdução e enquadramento pedagógico](#1-introdução-e-enquadramento-pedagógico)
   - 1.1. O desafio: porquê IA no ensino profissional?
   - 1.2. Abordagem pedagógica: da intuição visual à compreensão técnica
   - 1.3. Arquitetura do pacote para Moodle (filtro, tipo de pergunta, editor TinyMCE)
2. [Guia pedagógico das 9 simulações](#2-guia-pedagógico-das-9-simulações)
   - 2.1. Módulo 1: Descida do gradiente e otimização do modelo (`gradient_descent`)
   - 2.2. Módulo 2: Matriz de confusão e limiar de decisão (`confusion_matrix`)
   - 2.3. Módulo 3: Regressão polinomial e regularização (`regression`)
   - 2.4. Módulo 4: Fronteiras de decisão não lineares – k-NN e árvore de decisão (`knn_tree`)
   - 2.5. Módulo 5: Aprendizagem não supervisionada e agrupamento K-Means (`kmeans`)
   - 2.6. Módulo 6: Perceptrão e rede neuronal (`neural_network`)
   - 2.7. Módulo 7: Convolução e filtros de imagem (`convolution`)
   - 2.8. Módulo 8: Aprendizagem por reforço – Q-learning (`q_learning`)
   - 2.9. Módulo 9: Divisão de dados e enviesamento (`data_bias`)
3. [Instruções técnicas para professores](#3-instruções-técnicas-para-professores)
   - 3.1. Inserir simulações com o editor visual TinyMCE (`tiny_aisim`)
   - 3.2. Criar tarefas com avaliação automática (`qtype_aisim`)
   - 3.3. Importar o banco de perguntas pronto (Moodle XML)
   - 3.4. AISim Lab: as simulações sem Moodle
4. [Recomendações metodológicas e organização da aula](#4-recomendações-metodológicas-e-organização-da-aula)
   - 4.1. Estrutura de uma aula prática de 90 minutos
   - 4.2. Avaliação formativa e aprendizagem com os erros
   - 4.3. Ensino diferenciado
   - 4.4. Ordem sugerida das simulações
5. [Anexo e resumo das fórmulas](#5-anexo-e-resumo-das-fórmulas)

---

## 1. Introdução e enquadramento pedagógico

### 1.1. O desafio: porquê IA no ensino profissional?
Na mecatrónica, eletrotecnia, automação industrial, tecnologias da saúde e informática, a inteligência artificial (IA) e a aprendizagem automática (ML) já não são apenas tema de investigação universitária. Os alunos do ensino profissional (VET – *Vocational Education and Training*) contactam hoje com:
- inspeção ótica automática da qualidade em linhas de produção;
- manutenção preditiva de máquinas com base em sensores de vibração e temperatura;
- gestão inteligente de energia e climatização de edifícios;
- classificação de sinais médicos e robôs móveis autónomos.

No entanto, o ensino tradicional da aprendizagem automática cai muitas vezes num de dois extremos:
1. **Abstração matemática excessiva:** cálculo multivariável, derivadas parciais e álgebra matricial que desmotivam os alunos do ensino profissional.
2. **Bibliotecas como "caixa negra":** chamadas a funções prontas em Python (`model.fit()`, `model.predict()`) em que os alunos veem o resultado, mas não percebem o que o algoritmo faz realmente.

### 1.2. Abordagem pedagógica: da intuição visual à compreensão técnica
O pacote **AI for VET** propõe a **aprendizagem ativa por investigação (Active Inquiry-Based Learning)**:
- **Experimentação em tempo real:** o aluno altera os hiperparâmetros com cursores e botões e vê de imediato como o algoritmo reage.
- **Execução no navegador, sem instalação:** não é preciso Python, controladores de GPU nem servidores – as simulações correm inteiramente no navegador, mesmo em computadores escolares modestos, tablets e telemóveis.
- **Ciclo pedagógico em duas fases:**
  1. *Fase de exploração (lição):* o aluno altera livremente os parâmetros e procura regularidades.
  2. *Fase de aplicação (teste):* o aluno resolve um problema prático em que tem de levar o algoritmo à tolerância pedida, e o Moodle avalia a solução automaticamente.

### 1.3. Arquitetura do pacote para Moodle
O pacote é composto por três plugins que trabalham em conjunto:
- **`moodle-filter_aisim`** (filtro de texto e simulações): mostra as 9 simulações interativas onde quer que surja a marca `data-aisim` no conteúdo.
- **`moodle-qtype_aisim`** (tipo de pergunta e avaliação automática): coloca tarefas interativas nos testes do Moodle e avalia os valores alcançados após a submissão.
- **`moodle-tiny_aisim`** (plugin do editor TinyMCE): uma caixa de diálogo em que o professor insere, configura e edita (com duplo clique) uma simulação sem escrever HTML.

As simulações seguem o idioma do utilizador no Moodle: incluem inglês, sérvio (latino e cirílico), eslovaco e português.

---

## 2. Guia pedagógico das 9 simulações

As simulações vão das mais simples às mais exigentes. As cinco primeiras cobrem os fundamentos (otimização, avaliação de modelos, sobreajuste, classificação, agrupamento); os módulos 6–9 acrescentam redes neuronais, processamento de imagem, aprendizagem por recompensas e o uso responsável dos dados.

### 2.1. Módulo 1: Descida do gradiente e otimização do modelo (`gradient_descent`)
- **Conceitos-chave:** função de custo J(w), taxa de aprendizagem α, funções convexas e não convexas, mínimos locais.
- **Metáfora pedagógica:** Um caminhante desce uma montanha no meio de um nevoeiro cerrado. Não vê o mapa, só a inclinação do terreno debaixo dos pés. Em cada passo procura a direção de maior descida e avança nessa direção. O tamanho do passo é a **taxa de aprendizagem (α)**.
- **Aplicação prática:** afinação de controladores em mecatrónica; treino de redes neuronais (retropropagação).
- **O que os alunos observam:**
  1. *Taxa pequena (α = 0,01):* o algoritmo avança muito devagar.
  2. *Taxa adequada (α = 0,10 – 0,20):* convergência rápida e estável para o mínimo.
  3. *Taxa demasiado grande (α ≥ 1,05):* oscilações e divergência – o algoritmo salta o fundo e "foge".
  4. *Função não convexa:* o ponto de partida (w₀) decide em que mínimo local o modelo termina.
- **Tarefa típica de teste:**
  > *"O mínimo da função de custo convexa é J(w) = 1,00. Ajuste a taxa de aprendizagem (α) e execute o algoritmo de modo que o custo desça para J(w) ≤ 1,01."*

---

### 2.2. Módulo 2: Matriz de confusão e limiar de decisão (`confusion_matrix`)
- **Conceitos-chave:** avaliação de um classificador binário, classes desequilibradas, limiar θ, VP, FP, FN, VN, precisão (Precision), sensibilidade (Recall), F1.
- **Porque é que a exatidão (Accuracy) engana:** se só 2% das peças forem defeituosas, um modelo que considera todas boas tem **exatidão de 98%** – e não deteta uma única peça defeituosa.
- **Significado das métricas:**
  - *Limiar (θ):* a probabilidade acima da qual a peça é considerada defeituosa.
  - *Precisão (Precision):* das peças assinaladas como defeituosas, quantas o são de facto (poucos alarmes falsos).
  - *Sensibilidade (Recall):* de todas as peças defeituosas, quantas o modelo detetou (nenhuma peça defeituosa chega ao cliente).
  - *F1:* média harmónica da precisão e da sensibilidade.
- **O que os alunos observam:** mover o limiar θ para a esquerda aumenta a sensibilidade à custa da precisão; para a direita aumenta a precisão, mas deixa passar defeitos.
- **Tarefa típica de teste:**
  > *"Ajuste o limiar θ para que o classificador detete a maioria das peças defeituosas sem demasiados alarmes falsos: sensibilidade ≥ 0,80 e precisão ≥ 0,70."*

---

### 2.3. Módulo 3: Regressão polinomial e regularização (`regression`)
- **Conceitos-chave:** relações não lineares, grau do polinómio d, subajuste (*underfitting*), sobreajuste (*overfitting*), erro de treino e de teste, regularização L₂ (λ).
- **Metáfora pedagógica:** O subajuste é ler só os títulos dos capítulos. O sobreajuste é decorar cada letra do livro de exercícios, incluindo as gralhas – e não saber nada quando muda um número no enunciado.
- **Aplicação prática:** calibração de sensores de temperatura e pressão; previsão da carga de transformadores consoante a hora e a temperatura.
- **O que os alunos observam:**
  - d = 1: uma reta não consegue seguir a onda (erro alto no treino e no teste).
  - d = 3: uma curva suave descreve bem o sinal real.
  - d = 5 e λ = 0: a curva passa por cada ponto com ruído e oscila violentamente nas extremidades.
  - λ = 0,01 – 0,10: penalizar pesos grandes alisa a curva e recupera a generalização.
- **Tarefa típica de teste:**
  > *"Para um sinal periódico com ruído, escolha um grau de polinómio no máximo 7 de modo que o modelo capte a tendência sem sobreajuste, com um erro de validação MSE ≤ 0,19."*

---

### 2.4. Módulo 4: Fronteiras de decisão não lineares – k-NN e árvore de decisão (`knn_tree`)
- **Conceitos-chave:** aprendizagem supervisionada, k vizinhos mais próximos, árvore de decisão, fronteira de decisão, hiperparâmetros k e profundidade máxima da árvore.
- **Comparação:**
  - *k-NN:* "Diz-me com quem andas, dir-te-ei quem és." Um ponto novo recebe a classe mais frequente entre os k vizinhos mais próximos – a fronteira é arredondada.
  - *Árvore de decisão:* uma série de perguntas "se – então – senão" que dividem o plano em retângulos paralelos aos eixos.
- **Efeito dos hiperparâmetros:** k = 1 é sensível ao ruído (ilhas à volta de pontos isolados); k = 15 simplifica demasiado. Uma árvore pouco profunda não chega; uma muito profunda fragmenta o plano em pequenos retângulos à volta de cada ponto.

---

### 2.5. Módulo 5: Aprendizagem não supervisionada e agrupamento K-Means (`kmeans`)
- **Conceitos-chave:** agrupamento sem rótulos, centroides, distância euclidiana, inércia (soma dos quadrados das distâncias), convergência passo a passo.
- **Aplicação prática:** segmentação de consumidores de eletricidade por perfil de consumo; compressão de cores e análise de imagens de satélite.
- **O que os alunos observam:** com o botão "Próximo passo" acompanham dois passos alternados:
  1. *Atribuição:* cada ponto toma a cor do centroide mais próximo.
  2. *Atualização:* cada centroide desloca-se para o centro dos seus pontos.
- **Efeito de K:** K = 2 em dados com três grupos mostra um centroide a "cobrir" dois grupos; a inércia deixa de descer muito a partir do número real de grupos.

---

### 2.6. Módulo 6: Perceptrão e rede neuronal (`neural_network`)
- **Conceitos-chave:** perceptrão, separabilidade linear, camada oculta, funções de ativação (sigmoide, tanh, ReLU), épocas, função de perda (log-loss), taxa de aprendizagem, pesos iniciais.
- **Metáfora pedagógica:** Um neurónio é uma régua: só traça uma reta e diz "deste lado vermelho, daquele azul". Uma camada oculta é uma equipa de réguas cujas retas o neurónio de saída combina numa fronteira curva – como recortar um círculo de papel com vários cortes retos. Mais neurónios, cortes mais finos.
- **Aplicação prática:** distinguir peças boas e defeituosas a partir de várias medições de sensores. As redes neuronais estão na base do reconhecimento de fala e de imagem e dos grandes modelos de linguagem – aqui a mesma ideia aparece na sua forma mais pequena.
- **O que os alunos observam:**
  1. *Dados linearmente separáveis, 0 camadas ocultas:* o perceptrão encontra depressa uma reta e chega a quase 100%.
  2. *XOR (quadrantes cruzados), 0 camadas ocultas:* a exatidão fica nos 50%, por mais tempo que treine – nenhuma reta separa as classes. A simulação di-lo no conselho.
  3. *XOR com uma camada oculta de 2–4 neurónios:* a fronteira curva-se e a exatidão passa os 98%. A espessura das linhas no desenho da rede mostra que ligações ficaram fortes.
  4. *Círculo dentro de círculo:* com 2 neurónios a rede fica muitas vezes presa; com 3–4 neurónios e o botão "Novos pesos" encontra uma fronteira fechada à volta da classe interior.
  5. *Taxa de aprendizagem demasiado alta (p. ex. 3):* a perda "explode" – a mesma lição da descida do gradiente, porque treinar uma rede *é* descida do gradiente.
- **Tarefa típica de teste:**
  > *"Nos dados XOR o perceptrão fica nos 50%. Acrescente uma camada oculta e atinja exatidão ≥ 98% com no máximo 4 neurónios ocultos."*

---

### 2.7. Módulo 7: Convolução e filtros de imagem (`convolution`)
- **Conceitos-chave:** imagem como tabela de números, núcleo 3×3, divisor, desfocagem, nitidez, deteção de contornos (Sobel, Laplaciano), valor absoluto, redes neuronais convolucionais (CNN).
- **Metáfora pedagógica:** O núcleo é uma pequena "lupa com uma regra" que desliza sobre a imagem. Em cada posição olha para 9 píxeis, multiplica-os pelos seus pesos e escreve um número. Se todos os pesos forem iguais, a lupa faz a média (desfoca). Se subtrair um lado ao outro, só "vê" onde o brilho muda bruscamente – os contornos.
- **Aplicação prática:** inspeção ótica da qualidade (riscos, fissuras, contornos das peças na linha); remoção de ruído em imagens de câmaras e imagens médicas. As primeiras camadas de qualquer rede de reconhecimento de imagem (CNN) aprendem precisamente núcleos destes.
- **O que os alunos observam:**
  1. *Tocar num píxel* mostra o cálculo completo: 9 valores, 9 pesos, a soma e a divisão. É o momento-chave da aula – a convolução deixa de ser "magia".
  2. *Desfocagem (média) e desfocagem gaussiana:* o ruído desaparece, mas os contornos também ficam mais suaves.
  3. *Soma dos pesos / divisor ≠ 1:* a imagem inteira fica mais clara ou mais escura – a simulação avisa.
  4. *Sobel sem valor absoluto:* os contornos do claro para o escuro dão valores negativos e aparecem a preto. Com |x| veem-se os dois tipos de contorno.
  5. *Imagem-alvo e correspondência (R²):* os alunos adivinham ou constroem o filtro que produz o resultado pedido.
- **Tarefa típica de teste:**
  > *"A imagem está cheia de ruído e os filtros prontos estão ocultos. Escreva os pesos do núcleo e o divisor para que o resultado fique como o alvo, com correspondência ≥ 95%."* (Uma solução: pesos 1 2 1 / 2 4 2 / 1 2 1 e divisor 16.)

---

### 2.8. Módulo 8: Aprendizagem por reforço – Q-learning (`q_learning`)
- **Conceitos-chave:** agente, estado, ação, recompensa, tabela Q, episódio, taxa de aprendizagem α, fator de desconto γ, exploração ε (exploração vs. aproveitamento), política.
- **Metáfora pedagógica:** Treinar um cão: ninguém lhe explica as regras, recebe apenas um biscoito quando faz a coisa certa. Depois de muitas tentativas lembra-se de que movimento compensa em cada situação. O agente na simulação faz o mesmo: −0,1 por cada passo, +10 no objetivo, −10 se cair da falésia.
- **Aplicação prática:** controlo de robôs e veículos autónomos; gestão de energia (carregamento de baterias, aquecimento) e semáforos; IA de jogos, do xadrez e Go aos videojogos.
- **O que os alunos observam:**
  1. *Labirinto:* no início o agente vagueia ao acaso. Após algumas dezenas de episódios as cores das casas e as setas formam uma "placa indicadora" até ao objetivo e o caminho aprendido encurta para 10 passos.
  2. *Falésia com ε grande (p. ex. 0,4):* o agente aprende o caminho mais curto rente à borda, mas durante o treino cai frequentemente ao acaso – a recompensa média é baixa. Com ε = 0,05 cai muito menos.
  3. *Perto ou longe com γ = 0,5:* o agente fica com o pequeno tesouro T (+3) porque o futuro vale pouco para ele. Aumentar γ (p. ex. 0,9) torna o grande objetivo G (+10) mais valioso – mas só se o agente o tiver descoberto, o que exige ε suficiente.
  4. *Tocar numa casa* mostra os valores Q dos quatro movimentos – os alunos veem os números por trás da seta.
- **Tarefa típica de teste:**
  > *"O agente escolhe o pequeno tesouro T (+3) ali perto. Altere o fator de desconto γ e a exploração ε para que aprenda a ir até ao grande objetivo G (+10) em 4 passos, no máximo em 1000 episódios. Explique o papel de cada parâmetro."*

---

### 2.9. Módulo 9: Divisão de dados e enviesamento (`data_bias`)
- **Conceitos-chave:** conjuntos de treino e de teste, generalização, memorização (k = 1), amostra representativa, enviesamento dos dados (*data bias*), exatidão por grupo, equidade do modelo.
- **Metáfora pedagógica:** Um teste feito só com exercícios que o aluno já viu não diz nada. E se os exercícios e o teste vierem de um manual sem um único problema de geometria, o aluno tem nota máxima – e falha no primeiro triângulo que vê. O conjunto de teste só revela os erros que a sua amostra contém.
- **Aplicação prática:** um modelo de controlo de qualidade treinado com peças de uma fábrica e usado noutra (outra máquina, outro material). Sistemas de crédito, recrutamento ou diagnóstico médico que funcionam pior para grupos raros nos dados. O Regulamento Europeu da IA (AI Act) exige precisamente dados de qualidade e representativos e verificação para diferentes grupos.
- **O que os alunos observam:**
  1. *Quota da fábrica B de 5%:* o teste indica cerca de 95%, mas a exatidão real para a fábrica B é muito mais baixa. A média esconde o grupo raro.
  2. *Aumentar a quota de B (p. ex. para 30–50%):* a exatidão para B sobe muito, embora o tamanho total da amostra seja o mesmo – *quais* dados importa mais do que *quantos* dados.
  3. *"B comporta-se de forma semelhante":* a sub-representação não prejudica quando os grupos não diferem – o problema surge quando diferem e não estão representados.
  4. *k = 1:* a exatidão no treino é 100% (o modelo memorizou cada ponto), o que nada diz sobre a sua qualidade.
  5. *Conjunto de teste pequeno:* o resultado do teste salta com cada nova amostra (🎲) – um teste pequeno é uma medida pouco fiável.
- **Tarefa típica de teste:**
  > *"Recolher dados é caro. Com no máximo 120 amostras, leve a exatidão real das duas fábricas a ≥ 88%. O que importa mais: quantos dados ou quais dados?"*

---

## 3. Instruções técnicas para professores

### 3.1. Inserir simulações com o editor visual TinyMCE (`tiny_aisim`)
1. Na barra de ferramentas do TinyMCE, clique no **ícone de simulação de IA** (ou escolha *Inserir > AI Simulation...*).
2. Configure os parâmetros:
   - Escolha uma das 9 simulações.
   - Ajuste o cenário, limiar, taxa de aprendizagem, grau, número de grupos, arquitetura da rede, imagem e filtro, mapa do agente ou quota do grupo nos dados.
   - Marque **Apenas leitura** se quiser que os alunos só observem a simulação.
   - Na parte inferior da caixa de diálogo vê uma pré-visualização ao vivo.
3. Clique em **Inserir simulação**.
4. **Editar uma simulação existente:** basta fazer **duplo clique** no cartão da simulação no editor – a caixa de diálogo abre com as suas definições.

---

### 3.2. Criar tarefas com avaliação automática (`qtype_aisim`)
1. No teste do Moodle, escolha **Perguntas > Adicionar nova pergunta**.
2. Selecione **AI Simulation (`aisim`)**.
3. Indique o nome da pergunta, o enunciado, a pontuação e a penalização.
4. Na secção **Definições da simulação de IA**:
   - Escolha o tipo de simulação e o cenário.
   - Defina a métrica-alvo 1 e 2 (p. ex. sensibilidade `0.80` e precisão `0.70` na matriz de confusão). O significado depende da simulação – veja a ajuda (?) junto aos campos e a tabela no anexo.
   - A tolerância é usada na descida do gradiente (metade da pontuação até alvo + tolerância) e na convolução (metade da pontuação até alvo − tolerância).
   - Escreva o feedback para resposta correta, parcialmente correta e incorreta.
5. Guarde a pergunta. Durante o teste o Moodle mostra a simulação interativa e, após a submissão, calcula a pontuação automaticamente.

---

### 3.3. Importar o banco de perguntas pronto (Moodle XML)
O repositório `moodle-qtype_aisim` contém 27 tarefas prontas – três para cada uma das 9 simulações (nível básico, intermédio e avançado):
1. Descarregue [`questions_aiforvet.xml`](https://github.com/blagojevicboban/moodle-qtype_aisim/raw/master/demo/questions_aiforvet.xml).
2. No seu curso, abra **Banco de perguntas > Importar**.
3. Escolha o **formato Moodle XML**.
4. Carregue `questions_aiforvet.xml` e clique em **Importar**.
5. É criada a categoria *"AI za VET - Baza simulacionih zadataka (27 pitanja)"* com todas as perguntas prontas para um teste. (Os enunciados estão em sérvio; adapte-os conforme necessário.)

---

### 3.4. AISim Lab: as simulações sem Moodle
Em [sim.aiforvet.eu](https://sim.aiforvet.eu/) as 9 simulações funcionam de forma independente – sem conta e sem recolha de dados – em português, inglês, sérvio e eslovaco. Cada simulação tem uma breve explicação e três desafios.
- **Trabalho de casa e telemóveis:** os alunos podem praticar fora do Moodle; o progresso nos desafios fica guardado só no próprio navegador.
- **"Copiar ligação para esta configuração":** cria uma ligação para exatamente este estado da simulação – útil para o projetor ou para enviar aos alunos.
- **"Incorporar no Moodle":** dá o código HTML pronto desta configuração para colar em qualquer texto do Moodle (requer `filter_aisim`).

---

## 4. Recomendações metodológicas e organização da aula

### 4.1. Estrutura de uma aula prática de 90 minutos
```
[00-15 min] Problema real e contexto industrial
     │
     ▼
[15-35 min] Exploração orientada da simulação na lição
     │
     ▼
[35-50 min] Discussão em pares, hipóteses e conceitos
     │
     ▼
[50-75 min] Resolução individual no teste com avaliação automática
     │
     ▼
[75-90 min] Reflexão e ligação à área profissional
```

### 4.2. Avaliação formativa e aprendizagem com os erros
- No teste, configure o comportamento das perguntas como **Feedback imediato** ou **Interativo com várias tentativas**.
- O aluno não deve ser penalizado por testar valores-limite, mas valorizado por perceber porque é que o algoritmo diverge ou classifica mal.
- As tarefas avançadas do banco (p. ex. Q-learning e enviesamento dos dados) terminam com uma pergunta "Explique…" – use-a para uma breve resposta oral ou escrita depois do teste.

### 4.3. Ensino diferenciado
- **Nível básico:** cursores e verificar se o indicador do objetivo fica verde. Na convolução os alunos escolhem filtros prontos; no Q-learning observam o agente a aprender o labirinto.
- **Nível avançado:** compromisso precisão–sensibilidade, comparação de modelos regularizados, construção manual de núcleos, procura da rede mais pequena que resolve a tarefa e explicação do efeito de γ e ε no comportamento do agente.

### 4.4. Ordem sugerida das simulações
| Aula / tema | Simulações | Ideia-chave |
| :--- | :--- | :--- |
| 1. Como aprende um modelo | Descida do gradiente | Aprender = reduzir o erro em pequenos passos |
| 2. Como medimos um modelo | Matriz de confusão, Divisão de dados e enviesamento | A exatidão não chega; o teste tem de ser justo e representativo |
| 3. Complexidade do modelo | Regressão polinomial, k-NN e árvore de decisão | Demasiado simples vs. sobreajustado |
| 4. Aprender sem respostas | K-Means | Agrupar sem rótulos |
| 5. Redes neuronais e imagem | Perceptrão e rede neuronal, Convolução | De um neurónio ao reconhecimento de imagem |
| 6. Aprender com recompensas | Q-learning | Exploração vs. aproveitamento, o valor do futuro |

---

## 5. Anexo e resumo das fórmulas

| Simulação | Parâmetros-chave | Modelo / fórmula | Critério de sucesso típico |
| :--- | :--- | :--- | :--- |
| **Descida do gradiente** | α (passo), w₀ (início) | w ← w − α · J′(w) | J(w) ≤ 1,01 (o mínimo é 1,00) |
| **Matriz de confusão** | θ (limiar) | F1 = 2 · P · R / (P + R) | Sensibilidade ≥ 0,80 e precisão ≥ 0,70 |
| **Regressão polinomial** | d (grau), λ (regularização) | J(w) = (1/2m) · Σ(h(x) − y)² + (λ/2) · Σw² | MSE de validação ≤ 0,19 com d ≤ 7 |
| **k-NN** | k (vizinhos) | ŷ = classe mais frequente entre os k vizinhos mais próximos | Exatidão ≥ 98% |
| **K-Means** | K (grupos) | min Σ ‖x − μ‖² (soma dos quadrados das distâncias aos centroides) | Convergência com K = 3 e WCSS ≤ 10 |
| **Rede neuronal** | camadas, neurónios, ativação, taxa de aprendizagem | a = f(w · x + b); w ← w − α · ∂L/∂w | Exatidão ≥ 98% com no máximo 4 neurónios ocultos |
| **Convolução** | núcleo 3×3, divisor, \|x\| | y = Σ (peso · píxel) / divisor | Correspondência com o alvo (R²) ≥ 95% |
| **Q-learning** | α, γ, ε, episódios | Q(s,a) ← Q(s,a) + α · (r + γ · max Q(s′,·) − Q(s,a)) | Caminho mais curto (p. ex. 10 passos) em ≤ 200 episódios |
| **Divisão de dados e enviesamento** | tamanho da amostra, quota do grupo B, teste %, k | exatidão do pior grupo = min(exatidão A, exatidão B) | Ambas as fábricas ≥ 88% com ≤ 120 amostras |

#### Significado dos campos "Métrica-alvo" nas perguntas (`qtype_aisim`)
| Simulação | Métrica-alvo 1 | Métrica-alvo 2 | Metade da pontuação |
| :--- | :--- | :--- | :--- |
| Descida do gradiente | custo máximo permitido J(w) | – | custo até alvo + tolerância |
| Matriz de confusão | sensibilidade mínima (Recall) | precisão mínima (Precision) | uma condição cumprida |
| Regressão polinomial | MSE de validação máximo | grau máximo d | uma condição cumprida |
| k-NN e árvore de decisão | exatidão mínima no teste | – | exatidão até 5% abaixo do alvo |
| K-Means | inércia máxima (WCSS) | número exato de grupos K (0 = qualquer) | uma condição cumprida |
| Rede neuronal | exatidão mínima | máximo de neurónios ocultos (0 = sem limite) | uma condição cumprida |
| Convolução | correspondência mínima (R²) | – | correspondência até alvo − tolerância |
| Q-learning | máximo de passos até ao objetivo | máximo de episódios de treino (0 = sem limite) | o agente chega ao objetivo, mas por um caminho mais longo ou após mais episódios |
| Divisão de dados e enviesamento | exatidão mínima da fábrica mais fraca | tamanho máximo da amostra (0 = sem limite) | exatidão cumprida, mas com demasiadas amostras |

---

*Este guia foi desenvolvido no âmbito do projeto Erasmus+ AI for VET e é disponibilizado como recurso educativo aberto (REA) para todos os professores e instituições de ensino.*
