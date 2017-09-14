from scipy.stats import binom, beta
import numpy as np
import matplotlib.pyplot as plt

def betaPrior(x, a, b):
	return(beta.pdf(x, a, b))

def posterior(theta, successes, a, b, trials):
	post = beta.pdf(theta,successes+a, trials-successes+b)
	return(post)



if __name__ == "__main__":

	# n = int(input("Enter number of trials: "))
	# success = int(input("Enter number of successes: "))
	n = 16
	success = 13

	fig, ax = plt.subplots(2, 1, sharex = True)

	# Note: likelihood is binomial distribution w/ n = n
	# Note: prior is beta distribution
	thetas = linspace(0.001, 0.999, n)
	b55 = betaPrior(thetas, 0.5, 0.5)
	b11 = betaPrior(thetas, 1, 1, n)
	b22 = betaPrior(thetas, 2, 2, n)

