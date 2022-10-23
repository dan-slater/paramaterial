# Notes for hyde study

Hyde used the following equations to fit a constitutive model.

First calculate the Zener-Holloman parameter, using the mean temperature between 0.8 and 0.9 strain. The universal gas constant, R, is 8.3145 J/mol.K , and the activation energy , Q, is 155 JK/mol (from hyde ref 43). Strain-rate values also calculated.
$$
Z = \dot{\varepsilon} \exp \left( \frac{Q}{RT} \right)
$$
Can then plot ln(Z) against flow stress ![lnz vs flow stress](C:\Users\DS\paramaterial\examples\hyde study\report\lnz vs stress.PNG)

Can then use
$$
Z = A \exp \left( \beta \sigma \right)
$$
and (1) to fit beta and A.
$$
\sigma = \frac{1}{\beta} \ln Z - \frac{1}{\beta} \ln A
$$
