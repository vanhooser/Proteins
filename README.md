# Proteins
_"But what protein structure **defines** a mouse?"_

![](https://upload.wikimedia.org/wikipedia/commons/6/60/Myoglobin.png)

Find the dataset used [here](http://archive.ics.uci.edu/ml/datasets/Mice+Protein+Expression) and the starting code for SKLearn [here](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)


_Requirements_ 

* Python 2.7.x
* [SKLearn](http://scikit-learn.org/stable/install.html) SVC
* [Numpy](http://www.scipy.org/Installing_SciPy) _"Shouldn't you already have that installed?"_
* You're running OS X, right?

**Special thanks to Higuera C, Gardiner KJ and Cios KJ for their data collection and work as well as the UCI Machine Learning Repository**

##Quickstart

**~/ usr$ python proteins.py**
>----------Kernel :  linear ----------
>
>
>Genotype Analysis : 
>
> Correct : 154.00 (95.06 %)  
> Incorrect : 8.00 (4.94 %)  
> False Positive : 3.00 (1.85 %)  
> False Negative : 5.00 (3.09 %)  
>
>
> Treatment Analysis : 
>
> Correct : 121.00 (74.69 %)  
> Incorrect : 41.00 (25.31 %)  
> False Positive : 20.00 (12.35 %)  
> False Negative : 21.00 (12.96 %)  
>
>
> Behavior Analysis : 
>
> Correct : 162.00 (100.00 %)  
> Incorrect : 0.00 (0.00 %)  
> False Positive : 0.00 (0.00 %)  
> False Negative : 0.00 (0.00 %)  
>
> ----------Kernel :  sigmoid ----------
>
>
> Genotype Analysis : 
>
> Correct : 88.00 (54.32 %)  
> Incorrect : 74.00 (45.68 %)  
> False Positive : 0.00 (0.00 %)  
> False Negative : 74.00 (45.68 %)  
>
>
> Treatment Analysis : 
>
> Correct : 83.00 (51.23 %)  
> Incorrect : 79.00 (48.77 %)  
> False Positive : 0.00 (0.00 %)  
> False Negative : 79.00 (48.77 %)  
>
>
> Behavior Analysis : 
>
> Correct : 82.00 (50.62 %)  
> Incorrect : 80.00 (49.38 %)  
> False Positive : 80.00 (49.38 %)  
> False Negative : 0.00 (0.00 %)  
>
> ----------Kernel :  rbf ----------
>
>
> Genotype Analysis : 
>
> Correct : 123.00 (75.93 %)  
> Incorrect : 39.00 (24.07 %)  
> False Positive : 10.00 (6.17 %)  
> False Negative : 29.00 (17.90 %)  
>
>
> Treatment Analysis : 
>
> Correct : 98.00 (60.49 %)  
> Incorrect : 64.00 (39.51 %)  
> False Positive : 18.00 (11.11 %)  
> False Negative : 46.00 (28.40 %)  
>
>
> Behavior Analysis : 
>
> Correct : 162.00 (100.00 %)  
> Incorrect : 0.00 (0.00 %)  
> False Positive : 0.00 (0.00 %)  
> False Negative : 0.00 (0.00 %)  

_**Sweet mother of Turing, does that say 100%?**_


##Overview
This dataset is a very interesting one as it has direct and predictive possibilities across a number of organisms, not just mice with the particular genotypes/behaviors/treatments sought by Higuera, Gardiner, and Cios.  The simple floating-point measurement of the concentrations of the proteins of interest make for easy insertion into a training/validation set, and the binary nature of the three output class types make it simple to implement SVC machines here with whatever kernel our hearts so desire.

I've adapted this implementation slightly from my previous example, [Fire](https://github.com/thepropterhoc/Fire), to include a more agnostic view of the kernel functions used by the SVM, allowing us to dig a little deeper into which classifies the three groups best. 

##Differences
I've interpreted a sample from a mouse missing a particular protein measurement to have a value of _0.0_ for that protein, but it may be safer to remove these samples from the set instead.  This could be looked into next, but I've kept them here with this _0.0_ for the sake of keeping the training/validation sizes of the sets reasonable. 

##Results
Wow.  Yes in fact, a linear kernel has near-perfect prediction abilities for each of the three groups of classes sought (Genotype and Behavior).  