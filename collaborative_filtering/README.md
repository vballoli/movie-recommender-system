# Collaborative Filtering (CF)
The implemented algorithm is an approach to recommendation systems where we estimate the rating a user might give to an item based on the ratings the user has given to other items. This is done by computing the similarity between two items and then using the score as weights to the ratings already present.

### Pearson Correlation Coefficient
`method similarities.pearson_sim`

The similarity metric being used here is the Pearson Correlation Coefficient, given by the Cosine similarity between two standardized vectors (i.e., rows correspoinding to items).

sim(x, y) =                                       
 Σ(r<sub>xs</sub>-µ<sub>x</sub>)                   (r<sub>ys</sub>-µ<sub>y</sub>) /                 √[Σ(r<sub>xs</sub>-µ<sub>x</sub>)<sup>2</sup> x    Σ(r<sub>ys</sub>-µ<sub>y</sub>)<sup>2</sup>]      
### Item-Item CF
`class collaborate.Collaborate`

The actual rating is estimated here and the input matrix is filled.

To estimate rating for a given (user, item) pair:

r<sub>xi</sub> = Σsim(i,j).r<sub>xj</sub> / Σsim(i,j)

A common practice to get a better estimate is to account for baseline offset.

r<sub>xi</sub> = b + Σsim(i,j).(r<sub>xj</sub> - b<sub>x</sub>) / Σsim(i,j)

where b is given by,  
b = µ + b<sub>x</sub> + b<sub>i</sub>

**µ**: Overall Mean  
**b<sub>x</sub>**: Deviation for user *x*  
**b<sub>i</sub>**: Deviation for item *i*
