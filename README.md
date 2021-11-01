# ClassificationProject


1) Instruções de Instalação:

  - Selecionar uma pasta local onde o repositório será clonado.
  - Abrir o Git Bash na pasta selecionada.
  - Clonar o repositório com o comando 'git clone https://github.com/EddSB/ClassificationProject.git' na pasta.
  - Abrir e executar o script 'main.py'.


1) Introdução

  O objetivo deste projeto é a classificação de imagens através da utilização de redes neurais. O Dataset utilizado é o CIFAR-100, disponível em 'https://www.cs.toronto.edu/~kriz/cifar.html', restringido ao uso da superclasse 'people'.

1) Arquitetura do Projeto:

  Para melhor organização, o projeto foi dividido nas seguintes pastas:
	- configs: Contem as variáveis e constantes globais.
	- data: Guarda uma versão local dos dados
	- data_handler: Package responsável pela importação e preprocessamento dos dados
	- models: Contem uma classe abstrata de modelo de rede neural, e suas implementações 
		  as quais correspondem aos diferentes modelos aplicáveis.
	- utils: Utilidades que não caem em outras categorias, como visualização de imagens.

1) Notas de Desenvolvimento:

  Primeiramente foi estabelecida uma arquitetura básica utilizando o modelo Keras Sequencial. Esta rede neural simples analiza padrões presentes na imagem inteira, mas não possui uma boa adaptação em casos em que os padrões sofrem mudanças de posição, orientação ou tamanho. Esta rede apresentou uma precisão entre 28 e 25%. Considerando que haviam 5 classes nas quais as imagens deveriam ser classificadas, este resultado mostra que o desempenho da rede é pouco melhor do que um chute aleatório.
  Considerando que o objetivo é uma classificação de imagens, a bibliografia indica que as Redes Neurais Convolucionais se adequam à esta tarefa. Estas possuem camadas de convolução, as quais percorrem filtros sobre a imagem, identificando padrões. Adicionalmente, o valor dos pixels foi normalizado para que ficassem entre 0 e 1, ao invés de 0 e 255. Esta mudança torna os valores menores, melhorando a velocidade de processamento, bem como a facilidade de treinamento da rede neural, considerando que a ativação utilizada foi a 'relu'. Esta rede apresentou uma precisão inicial de 38%, um avanço com relação a rede anterior, mas um resultado ainda insatisfatório.
  Para verificar o impacto da quantidade de dados no resultado, o dataset CIFAT-100 foi utilizado como comparativo por possuir 50000 imagens de treinamento. Com estes dados, a rede neural alcançou uma precisão de XX %, indicando que uma quantidade maior de dados leva a um resultado melhor.
  Dessa forma, para aumentar a quantidade de dados relativo ao Dataset CIFAR-10, superclasse 'people' (2500 imagens de treinamento), foi realizado um Aumento dos Dados (Data Augmenting).
Cada uma das imagens do conjunto recebeu uma transformada aleatória, e a imagem resultante foi concatenada ao conjunto original. Esta prática evita que a rede neural sofra um sobreajuste (overfitting) muito rapidamente, pois distribui as características da imagem para diferentes pontos. Com esta técnica, foi possível alcansar uma precisão de 44% para a CNN. 









