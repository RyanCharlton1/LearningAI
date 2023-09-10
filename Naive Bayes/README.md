# Naive Bayesian Method
A method to classify new data entries probabilistically based off a set of complete data based of Bayes theorem.

### Bayes theorem
$P(B|A) = \frac{P(B|A) * P(A)}{P(B)}$ \
Can be derived trivially from $P(A|B) = \frac{P(A\cap B)}{P(B)}$.

For this method A will be a set of features, for $A=\set{a_0,a_1...a_n}$ 

$P(B|\set{a_0,a_1...a_n}) = \frac{P(a_0|B)P(a_1|B)...P(a_n|B)P(B)}{P(a_0)P(a_1)P(a_n)}$

### Method
The method makes the assumption that each feature of a entry has a equal and independant effect on the outcome. Strictly this is never true but in practice the relationships between features are so complex that it works, this assumption is why it's naive.

The method calculates the chances for each of the feature's possibilities given the features you have and chooses the most likely. Because the features given are the same for comparisons for a entry we can ignore the denominator of the Bayes theorem as it's going to be the same.

$P(B|\set{a_0,a_1...a_n})\propto P(B)\Pi^n_{i=0}P(a_i|B)$

So for $B = \set{b_0, b_1...b_n}$ compute each $P(b|A)$ and the pick the feature that's most likely.