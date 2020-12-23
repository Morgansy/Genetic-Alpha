# Genetic-Alpha
A genetic programming algorithm used for generating alpha factors in the multi-factor investment strategy 


# Introduction 

Want to find investment factors that are significantly correlated to the returns of securities? Use Genetic-alpha! Recently, more and more investors are using the factor investment strategy, where the most important task is to find alpha your own factors. Most of tranditional alpha factors are known by many investors, then gradually becomes invalid. Genetic-Alpha is build based on the Genetic programming algorithm, which is a symbolic regression technique. It begins by building a population of naive random formulas to represent a relationship between known independent variables and their dependent variable targets in order to predict new data. In our content, the dependent variable is the security returns, while the independent variables can be any variables that you think are related to the security returns. Each successive generation of programs is then evolved from the one that came before it by selecting the fittest individuals from the population to undergo genetic operations.  

# functionality

The **fitness.py** documents provides API where you could design your own measures to test the performance of the generated alpha factors, several commonly used measures includes *IR*, *ICIR*, etc. I have already buile these measures in this package.

The **function.py** documents can produce functions (or operators) objects that will be used in the formulas of the alpha facors. Still, you can also customize your own fucntions to design alpha factors that are in your interest. 

Parallel computting is supported by **utils.py**, details can be checked.

The main machanism of genetic programming algorithm is defined and implemented in **program.py** and  **genetic.py**

# data

This package supports time series data, which is a big difference to the original genetic programming algorithm. For example, the data of the vairable *adjusted price* is should be a two dimensional dataframe, the first dimension represents tickers, while the second dimensioan is time. 

# test run

To run a test program for this package, please see **demo.py**

Finally, have fun with alpha investing!
