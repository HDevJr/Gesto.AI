# Escopo do MVP

## Objetivo do MVP
O MVP (Produto Mínimo Viável) tem como objetivo validar tecnicamente a proposta do projeto por meio de um sistema funcional capaz de reconhecer um conjunto reduzido de sinais em Libras utilizando visão computacional, convertendo esses sinais em texto e áudio.

## Definição do MVP
Nesta primeira versão, o sistema deverá:
- utilizar webcam como entrada;
- capturar gestos realizados em frente à câmera;
- extrair landmarks das mãos e, se necessário, pose corporal;
- classificar sinais isolados;
- exibir o texto correspondente ao sinal reconhecido;
- reproduzir áudio correspondente ao texto identificado.

## O que entra no MVP
- reconhecimento de sinais isolados;
- vocabulário reduzido;
- uso de visão computacional;
- pipeline de landmarks;
- modelo de classificação temporal;
- saída textual;
- síntese de voz;
- ambiente controlado de teste.

## O que não entra no MVP
- tradução completa de frases contínuas;
- interpretação semântica avançada;
- dispositivo vestível;
- fusão multimodal;
- uso offline embarcado;
- suporte a vocabulário extenso;
- robustez total para qualquer ambiente real;
- interface final de produto.

## Hipótese Principal
É possível reconhecer com desempenho satisfatório um conjunto inicial de sinais em Libras a partir de landmarks extraídos por visão computacional, utilizando modelos de aprendizado de máquina ou Deep Learning.

## Critérios de Sucesso do MVP
O MVP será considerado bem-sucedido se alcançar os seguintes resultados:

### Critérios técnicos
- pipeline completo funcionando de ponta a ponta;
- landmarks extraídos com consistência;
- dataset inicial organizado e rotulado;
- modelo capaz de aprender os sinais definidos;
- inferência funcional em demonstração controlada.

### Critérios de desempenho
- reconhecimento consistente do conjunto inicial de sinais;
- latência aceitável para demonstração;
- taxa de acerto suficiente para validar continuidade do projeto.

## Cenário de Uso do MVP
O usuário realiza um gesto correspondente a um dos sinais previstos no vocabulário inicial, diante de uma webcam. O sistema processa a entrada, identifica o sinal e retorna o texto e o áudio correspondentes.

## Tipo de Problema Neste MVP
O problema nesta fase será tratado como:
- classificação de sinais isolados;
- entrada temporal baseada em landmarks;
- saída de classe mapeada para texto e áudio.

## Restrições do MVP
- dataset inicialmente pequeno;
- número limitado de participantes;
- ambiente de gravação controlado;
- foco em sinais isolados;
- dependência de câmera e iluminação adequadas.

## Riscos do MVP
- sinais muito parecidos entre si;
- pouca generalização com poucos usuários;
- ruído visual;
- dificuldade na segmentação temporal;
- desempenho insuficiente para tempo real.

## Estratégia de Mitigação
- começar com poucos sinais;
- priorizar sinais mais claros visualmente;
- padronizar gravação e coleta;
- avaliar erros por classe;
- ajustar gradualmente o vocabulário e o pipeline.

## Resultado Esperado
Ao final do MVP, o projeto deverá ter:
- uma base arquitetural organizada;
- um dataset inicial utilizável;
- um modelo treinado para sinais isolados;
- uma demonstração funcional com texto e voz;
- insumos técnicos para a evolução do sistema.