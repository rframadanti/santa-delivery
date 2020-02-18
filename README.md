# santa-delivery
Inspired by Kaggle's Santa's Stolen Sleigh 

Applied stochastic local search method (Random Search and Simulated Annealing) to approximate optimisation of delivery problem.

Description:
Another year has passed, 2019 brings many well-behaved boys and girls who deserve to get their Christmas presents on the 25th December. After a yearlong wait, Santa is back with his 8 reindeer to ensure children receive their gifts on time for Christmas. 
However, this comes with a number of implications, which includes a magic-less sleigh due to the magic one being stolen. This is why as a team we have ensured the most efficient solution to allow Santa to have an easy, stress free Christmas.

We were given the dataset gifts.csv which we ran in python. This excel file consists of a total 100,000 gifts and 100,000 destinations worldwide, with the longitude and latitude of the various destinations. 
Santa has to return to the North Pole after every trip. The overall goal is to minimize total weighted reindeer weariness (weighted distance).

The steps: 
•	Began by understanding the problem, which is an optimisation solution. 
•	Creating and utilising the objective function
•	We ran two different approaches, random search, and simulated annealing
•	After running both of these approaches we will see the results. 
•	From the results, we can then implement 2 conditions which consist of neighbourhood move 3 and neighbourhood move 6 with simulated annealing. 
•	Then adding the temperature decay, which is changing the alpha to {0.98, 0.95, 0.90, 0.8,0.5}
•	Times and seeing the response.
•	We did this by using the universe data set which we were given, with the variables, longitude, latitude and the WRW
•	We ran this with 3 different sample sizes 10,100,1000.
•	From this data we created a random sample by setting a random sample seed, because we distributed the data between 
•	Observed the data behaviour for each algorithm and analysed the results and findings through graphs, such as scatter plots and box plots. 
•	We will see how the results work and see how each step creates a different solution. 
