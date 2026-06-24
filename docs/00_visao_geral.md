# Visão Geral do Projeto

## Nome do Projeto
LibrasSense

## Descrição
O LibrasSense é um projeto de sistema inteligente voltado ao reconhecimento de gestos em Libras, com o objetivo de converter sinais gestuais em texto e áudio, promovendo acessibilidade comunicacional e inclusão social.

A proposta inicial do projeto é utilizar visão computacional como base de aprendizado, validação e construção do pipeline de inteligência artificial. Em fases futuras, o sistema poderá evoluir para incorporar sensores vestíveis, reduzindo dependência de câmeras e ampliando possibilidades de uso em contextos reais.

## Problema
Pessoas que utilizam Libras ou apresentam dificuldades na fala podem enfrentar barreiras de comunicação em ambientes sociais onde a maioria das pessoas não compreende a linguagem gestual.

Essas barreiras afetam:
- interação cotidiana;
- solicitação de ajuda;
- expressão de necessidades básicas;
- inclusão em contextos sociais, acadêmicos e profissionais.

## Proposta de Solução
Desenvolver um sistema capaz de:
1. capturar gestos por meio de webcam;
2. extrair informações relevantes do movimento das mãos e do corpo;
3. reconhecer sinais em Libras com apoio de modelos de Deep Learning;
4. converter os sinais reconhecidos em texto;
5. gerar saída em áudio por síntese de voz.

## Estratégia de Desenvolvimento
O projeto será desenvolvido de forma incremental, em etapas de complexidade crescente:

1. reconhecimento de sinais simples e isolados;
2. ampliação do vocabulário;
3. reconhecimento de expressões curtas;
4. reconhecimento de sequências gestuais;
5. integração com sensores vestíveis;
6. construção de sistema multimodal.

## Objetivo Geral
Construir uma base técnica, arquitetural e experimental sólida para um sistema de reconhecimento gestual em Libras com saída textual e sonora.

## Objetivos Específicos
- estruturar um pipeline de visão computacional para captura e processamento de gestos;
- criar um dataset inicial para treinamento do MVP;
- treinar modelos para reconhecimento de sinais isolados;
- validar o uso de landmarks como representação dos gestos;
- integrar saída em texto e áudio;
- preparar a arquitetura para futura expansão com sensores.

## Público-Alvo Inicial
Nesta fase inicial, o foco do projeto será a validação técnica do sistema. O usuário-alvo de referência é:
- pessoa que utiliza sinais para comunicação;
- contexto de interação social simples;
- necessidade de expressar mensagens curtas e objetivas.

## Proposta de Valor
O projeto busca criar uma ponte entre comunicação gestual e linguagem oral/escrita, utilizando inteligência artificial para ampliar acessibilidade e autonomia comunicacional.

## Diferencial da Abordagem
O diferencial do projeto está em:
- começar pela visão computacional como ambiente de aprendizado;
- evoluir de forma gradual, do simples ao complexo;
- preparar a arquitetura para sensores vestíveis no futuro;
- estruturar o sistema desde o início com visão de escalabilidade.

## Visão de Longo Prazo
No longo prazo, o projeto poderá evoluir para:
- reconhecimento de frases e expressões mais complexas;
- uso em tempo real com maior robustez;
- integração com sensores vestíveis;
- fusão multimodal entre visão e sensores;
- aplicação prática em ambientes sociais reais.