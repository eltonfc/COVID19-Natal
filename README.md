# COVID19-Natal
Montecarlo simulations to estimate the impact of COVID19 in healthcare system in Natal for different scenarios.

## Visão Geral

Este programa tem por objetivo simular a evolução de doenças infecciosas como a
COVID19 em função de parâmetros como taxa de infecção, mortalidade, taxa de
hospitalização e taxa de cuidados intensivos, dependência de faixa etária,
número de leitos disponíveis e a implantação de medidas de supressão e
mitigação.

O código retorna o número de novos casos, internações, internações em UTI,
mortes e recuperações em função do tempo e, opcionalmente, faixa etária.

Este código pode ser classificado como um modelo SEIR, que considera a
população como suscetível, exposta, infecciosa e recuperada/removida, mas conta
explicitamente os óbitos.

Esta é uma extensão do algoritmo desenvolvido por C. Pellicer da Escola de
Ciências e Tecnologia da Universidade Federal do Rio Grande do norte
(ECT-UFRN). Implementação de grafos de contato baseado no
[`seirsplus`](https://github.com/ryansmcgee/seirsplus).

## Autores

- Elton José Figueiredo de Carvalho - ECT-UFRN
- Carlos Eduardo Pellicer de Oliveira - ECT-UFRN
- Efrain Pantaleón Matamoros - ECT-UFRN

## [CRediT](https://onlinelibrary.wiley.com/doi/full/10.1002/leap.1210)

- **Elton Carvalho** Methodology, Software, Validation, Investigation
- **Carlos Pellicer** Methodology, Software, Validation, Investigation
- **Efrain Matamoros** Conceptualization, Supervision

## References

- [SEIRS+](https://github.com/ryansmcgee/seirsplus)
- [Gillepsie algorithm with networks](https://andrewmellor.co.uk/blog/articles/2014/12/19/gillespie-epidemics/)
