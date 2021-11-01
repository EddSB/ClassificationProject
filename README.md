# ClassificationProject


1) Instruções de Instalação:

  - No Anaconda Comsol, executar "pip install opencv-python"

  - Selecionar uma pasta local onde o repositório será clonado.
  - Abrir o Git Bash na pasta selecionada.
  - Clonar o repositório com o comando 'git clone https://github.com/EddSB/ClassificationProject.git' na pasta.
  - Abrir e executar o script 'main.py'.


1) Introdução

  O objetivo deste projeto é a classificação de imagens através da utilização de redes neurais. O Dataset utilizado é o CIFAR-100, disponível em 'https://www.cs.toronto.edu/~kriz/cifar.html', restringido ao uso da superclasse 'people'.

1) Arquitetura do Projeto:


1) Notas de Desenvolvimento:

  Primeiramente foi estabelecida uma arquitetura básica utilizando o modelo Keras Sequencial. Esta rede neural simples analiza padrões presentes na imagem inteira, mas não possui uma boa adaptação em casos em que os padrões sofrem mudanças de posição, orientação ou tamanho. Esta rede apresentou uma precisão entre 28 e 25%. Considerando que haviam 5 classes nas quais as imagens deveriam ser classificadas, este resultado mostra que o desempenho da rede é pouco melhor do que um chute aleatório.
  Considerando que o objetivo é uma classificação de imagens, a bibliografia indica que as Redes Neurais Convolucionais se adequam à esta tarefa. Estas possuem camadas de convolução, as quais percorrem filtros sobre a imagem, identificando padrões. Adicionalmente, o valor dos pixels foi normalizado para que ficassem entre 0 e 1, ao invés de 0 e 255. Esta mudança torna os valores menores, melhorando a velocidade de processamento, bem como a facilidade de treinamento da rede neural, considerando que a ativação utilizada foi a 'relu'. Esta rede apresentou uma precisão inicial de 38%, um avanço com relação a rede anterior, mas um resultado ainda insatisfatório.
  










 - normalizar os pixels entre 0 e 1 melhorou MUITO o treinamento