# TrabalhoFinalGPU
Simulação de Embates em GPU

Uma  característica importante ao se desenvolver um jogo é o fator de "replay", i.e., a capacidade de um jogo ser jogado novamente ao ser tUma característica importante em um jogo é o balanceamento, isso ocorre quando os modos ou estilos são equivalentes, por exemplo, em um jogo de luta o personagem mais lento em atacar ainda terá chance de ganhar de um mais rápido, caso ele seja mais forte, já que, ele precisa acertar menos vezes para causar o mesmo dano, ou em um jogo de corrida onde o carro mais veloz tem mais dificuldade em curvas.
Entretanto encontrar um equilíbrio entre atributos não é simples devido  a complexidade das combinações, i.e., um jogo onde os NPCs(non-player character) possuem força, velocidade, vida e armadura como atributos, possibilita diversas combinações de distribuições de valores que irão está balanceados entre si.
Esse trabalho propõe que ao usar CUDA, que é uma “linguagem de programação” para GPU NVIDIA, é possível simular as possíveis combinações de atributos e assim chegar a padrões balanceados 
As simulações foram inspiradas em jogos onde o jogador não controla diretamente seus personagens, o jogador pode escolher o posicionamento ou os atributos porém eles lutarão sozinhos. Exemplos desse estilo de jogo é o Teamfight Tacticst da empresa Riot Games e Underlords da empresa Valve.
