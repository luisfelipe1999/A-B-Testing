#!/usr/bin/env python
# coding: utf-8

# # Probability 2: Loaded dice 
# 
# In this assignment you will be reinforcening your intuition about the concepts covered in the lectures by taking the example with the dice to the next level. 
# 
# This assignment will not evaluate your coding skills but rather your intuition and analytical skills. You can answer any of the exercise questions by any means necessary, you can take the analytical route and compute the exact values or you can alternatively create some code that simulates the situations at hand and provide approximate values (grading will have some tolerance to allow approximate solutions). It is up to you which route you want to take! 
# 
# This graded notebook is different from what you might seen in other assignments of this specialization since only your answers are graded and not the code you used to get that answer. For every exercise there is a blank cell that you can use to make your calculations, this cell has just been placed there for you convenience but **will not be graded** so you can leave empty if you want to. 
# 
# However **you need to submit the answer for that exercise by running the cell that contains the `utils.exercise_x()` function**. By running this cell a widget will appear in which you can place your answers. Don't forget to click the `Save your answer!` button.

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import utils


# ## Some concept clarifications 🎲🎲🎲
# 
# During this assignment you will be presented with various scenarios that involve dice. Usually dice can have different numbers of sides and can be either fair or loaded.
# 
# - A fair dice has equal probability of landing on every side.
# - A loaded dice does not have equal probability of landing on every side. Usually one (or more) sides have a greater probability of showing up than the rest.
# 
# Let's get started!

# ## Exercise 1:
# 
# 

# Given a 6-sided fair dice (all of the sides have equal probability of showing up), compute the mean and variance for the probability distribution that models said dice. The next figure shows you a visual represenatation of said distribution:
# 
# <img src="./images/fair_dice.png" style="height: 300px;"/>
# 
# **Submission considerations:**
# - Submit your answers as floating point numbers with three digits after the decimal point
# - Example: To submit the value of 1/4 enter 0.250

# Hints: 
# - You can use [np.random.choice](https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html) to simulate a fair dice.
# - You can use [np.mean](https://numpy.org/doc/stable/reference/generated/numpy.mean.html) and [np.var](https://numpy.org/doc/stable/reference/generated/numpy.var.html) to compute the mean and variance of a numpy array.

# In[5]:


# You can use this cell for your calculations (not graded)
np.random.seed(0)
dice_rolls = np.random.choice([1, 2, 3, 4, 5, 6], size=1000000)
mean = np.mean(dice_rolls)
variance = np.var(dice_rolls)
print(f"Mean: {mean:.3f}, Variance: {variance:.3f}")


# In[6]:


# Run this cell to submit your answer
utils.exercise_1()


# ## Exercise 2:
# 
# Now suppose you are throwing the dice (same dice as in the previous exercise) two times and recording the sum of each throw. Which of the following `probability mass functions` will be the one you should get?
# 
# <table><tr>
# <td> <img src="./images/hist_sum_6_side.png" style="height: 300px;"/> </td>
# <td> <img src="./images/hist_sum_5_side.png" style="height: 300px;"/> </td>
# <td> <img src="./images/hist_sum_6_uf.png" style="height: 300px;"/> </td>
# </tr></table>
# 

# Hints: 
# - You can use numpy arrays to hold the results of many throws.
# - You can sum to numpy arrays by using the `+` operator like this: `sum = first_throw + second_throw`
# - To simulate multiple throws of a dice you can use list comprehension or a for loop

# In[7]:


# You can use this cell for your calculations (not graded)
np.random.seed(0)
num_trials = 1000000
first_throw = np.random.choice([1, 2, 3, 4, 5, 6], size=num_trials)
second_throw = np.random.choice([1, 2, 3, 4, 5, 6], size=num_trials)
sums = first_throw + second_throw

plt.hist(sums, bins=range(2, 14), align='left', rwidth=0.8, density=True)
plt.xticks(range(2, 13, 2))
plt.xlabel('Sum of dice')
plt.ylabel('Probability')
plt.title('Histogram of sum')
plt.show()


# In[8]:


# Run this cell to submit your answer
utils.exercise_2()


# ## Exercise 3:
# 
# Given a fair 4-sided dice, you throw it two times and record the sum. The figure on the left shows the probabilities of the dice landing on each side and the right figure the histogram of the sum. Fill out the probabilities of each sum (notice that the distribution of the sum is symmetrical so you only need to input 4 values in total):
# 
# <img src="./images/4_side_hists.png" style="height: 300px;"/>
# 
# **Submission considerations:**
# - Submit your answers as floating point numbers with three digits after the decimal point
# - Example: To submit the value of 1/4 enter 0.250

# In[9]:


# You can use this cell for your calculations (not graded)
np.random.seed(0)
num_trials = 1000000
first_throw = np.random.choice([1, 2, 3, 4], size=num_trials)
second_throw = np.random.choice([1, 2, 3, 4], size=num_trials)
sums = first_throw + second_throw

plt.hist(sums, bins=range(2, 10), align='left', rwidth=0.8, density=True)
plt.xticks(range(2, 9))
plt.xlabel('Sum of dice')
plt.ylabel('Probability')
plt.title('Histogram of sum')
plt.show()


# In[10]:


# Run this cell to submit your answer
utils.exercise_3()


# ## Exercise 4:
# 
# Using the same scenario as in the previous exercise. Compute the mean and variance of the sum of the two throws  and the covariance between the first and the second throw:
# 
# <img src="./images/4_sided_hist_no_prob.png" style="height: 300px;"/>
# 
# 
# Hints:
# - You can use [np.cov](https://numpy.org/doc/stable/reference/generated/numpy.cov.html) to compute the covariance of two numpy arrays (this may not be needed for this particular exercise).

# In[11]:


# You can use this cell for your calculations (not graded)
np.random.seed(0)
num_trials = 1000000
first_throw = np.random.choice([1, 2, 3, 4], size=num_trials)
second_throw = np.random.choice([1, 2, 3, 4], size=num_trials)
sums = first_throw + second_throw

print(f"Mean of sum: {np.mean(sums):.3f}")
print(f"Variance of sum: {np.var(sums):.3f}")
print(f"Covariance between first and second throw: {np.cov(first_throw, second_throw)[0, 1]:.3f}")


# In[12]:


# Run this cell to submit your answer
utils.exercise_4()


# ## Exercise 5:
# 
# 
# Now suppose you are have a loaded 4-sided dice (it is loaded so that it lands twice as often on side 2 compared to the other sides): 
# 
# 
# <img src="./images/4_side_uf.png" style="height: 300px;"/>
# 
# You are throwing it two times and recording the sum of each throw. Which of the following `probability mass functions` will be the one you should get?
# 
# <table><tr>
# <td> <img src="./images/hist_sum_4_4l.png" style="height: 300px;"/> </td>
# <td> <img src="./images/hist_sum_4_3l.png" style="height: 300px;"/> </td>
# <td> <img src="./images/hist_sum_4_uf.png" style="height: 300px;"/> </td>
# </tr></table>

# Hints: 
# - You can use the `p` parameter of [np.random.choice](https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html) to simulate a loaded dice.

# In[13]:


# You can use this cell for your calculations (not graded)
np.random.seed(0)
num_trials = 1000000
first_throw = np.random.choice([1, 2, 3, 4], p=[1/5, 2/5, 1/5, 1/5], size=num_trials)
second_throw = np.random.choice([1, 2, 3, 4], p=[1/5, 2/5, 1/5, 1/5], size=num_trials)
sums = first_throw + second_throw

plt.hist(sums, bins=range(2, 10), align='left', rwidth=0.8, density=True)
plt.xticks(range(2, 9))
plt.xlabel('Sum of dice')
plt.ylabel('Probability')
plt.title('Histogram of sum')
plt.show()


# In[14]:


# Run this cell to submit your answer
utils.exercise_5()


# ## Exercise 6:
# 
# You have a 6-sided dice that is loaded so that it lands twice as often on side 3 compared to the other sides:
# 
# <img src="./images/loaded_6_side.png" style="height: 300px;"/>
# 
# You record the sum of throwing it twice. What is the highest value (of the sum) that will yield a cumulative probability lower or equal to 0.5?
# 
# <img src="./images/loaded_6_cdf.png" style="height: 300px;"/>
# 
# Hints:
# - The probability of side 3 is equal to $\frac{2}{7}$

# In[15]:


# You can use this cell for your calculations (not graded)
np.random.seed(0)
num_trials = 1000000
dice = [1,2,3,4,5,6]
probs = [1/7, 1/7, 2/7, 1/7, 1/7, 1/7]

first_throw = np.random.choice(dice, p=probs, size=num_trials)
second_throw = np.random.choice(dice, p=probs, size=num_trials)

sums = first_throw + second_throw

for i in range(2, 13):
    cdf = (sums <= i).mean()
    if cdf > 0.5:
        break

print("The highest value (of the sum) that will yield a cumulative probability lower or equal to 0.5 is", i-1)


# In[16]:


# Run this cell to submit your answer
utils.exercise_6()


# ## Exercise 7:
# 
# Given a 6-sided fair dice you try a new game. You only throw the dice a second time if the result of the first throw is **lower** or equal to 3. Which of the following `probability mass functions` will be the one you should get given this new constraint?
# 
# <table><tr>
# <td> <img src="./images/6_sided_cond_green.png" style="height: 250px;"/> </td>
# <td> <img src="./images/6_sided_cond_blue.png" style="height: 250px;"/> </td>
# <td> <img src="./images/6_sided_cond_red.png" style="height: 250px;"/> </td>
# <td> <img src="./images/6_sided_cond_brown.png" style="height: 250px;"/> </td>
# 
# </tr></table>
# 
# Hints:
# - You can simulate the second throws as a numpy array and then make the values that met a certain criteria equal to 0 by using [np.where](https://numpy.org/doc/stable/reference/generated/numpy.where.html)

# In[17]:


# You can use this cell for your calculations (not graded)
np.random.seed(0)
num_trials = 1000000
first_throw = np.random.choice([1, 2, 3, 4, 5, 6], size=num_trials)
second_throw = np.random.choice([1, 2, 3, 4, 5, 6], size=num_trials)

second_throw = np.where(first_throw <= 3, second_throw, 0)
final_outcomes = first_throw + second_throw

plt.hist(final_outcomes, bins=range(2, 11), align='left', rwidth=0.8, density=True)
plt.xticks(range(2, 10))
plt.xlabel('Sum of dice')
plt.ylabel('Probability')
plt.title('Histogram of sum')
plt.show()


# In[18]:


# Run this cell to submit your answer
utils.exercise_7()


# ## Exercise 8:
# 
# Given the same scenario as in the previous exercise but with the twist that you only throw the dice a second time if the result of the first throw is **greater** or equal to 3. Which of the following `probability mass functions` will be the one you should get given this new constraint?
# 
# <table><tr>
# <td> <img src="./images/6_sided_cond_green2.png" style="height: 250px;"/> </td>
# <td> <img src="./images/6_sided_cond_blue2.png" style="height: 250px;"/> </td>
# <td> <img src="./images/6_sided_cond_red2.png" style="height: 250px;"/> </td>
# <td> <img src="./images/6_sided_cond_brown2.png" style="height: 250px;"/> </td>
# 
# </tr></table>
# 

# In[19]:


# You can use this cell for your calculations (not graded)
np.random.seed(0)
num_trials = 1000000
first_throw = np.random.choice([1, 2, 3, 4, 5, 6], size=num_trials)
second_throw = np.random.choice([1, 2, 3, 4, 5, 6], size=num_trials)

second_throw = np.where(first_throw >= 3, second_throw, 0)
final_outcomes = first_throw + second_throw

plt.hist(final_outcomes, bins=range(1, 14), align='left', rwidth=0.8, density=True)
plt.xticks(range(2, 13))
plt.xlabel('Sum of dice')
plt.ylabel('Probability')
plt.title('Histogram of sum')
plt.show()


# In[20]:


# Run this cell to submit your answer
utils.exercise_8()


# ## Exercise 9:
# 
# Given a n-sided fair dice. You throw it twice and record the sum. How does increasing the number of sides `n` of the dice impact the mean and variance of the sum and the covariance of the joint distribution?

# In[21]:


# You can use this cell for your calculations (not graded)
def simulate_dice_throws(n, num_trials=100000):
    np.random.seed(0)
    first_throw = np.random.choice(np.arange(1, n+1), size=num_trials)
    second_throw = np.random.choice(np.arange(1, n+1), size=num_trials)
    sum_throws = first_throw + second_throw

    mean = np.mean(sum_throws)
    variance = np.var(sum_throws)
    covariance = np.cov(first_throw, second_throw)[0][1]

    return mean, variance, covariance

for n in range(2, 11):
    mean, variance, covariance = simulate_dice_throws(n)
    print(f"For a {n}-sided dice:")
    print(f"Mean of the sum: {mean:.3f}")
    print(f"Variance of the sum: {variance:.3f}")
    print(f"Covariance of the joint distribution: {covariance:.3f}\n")


# In[22]:


# Run this cell to submit your answer
utils.exercise_9()


# ## Exercise 10:
# 
# Given a 6-sided loaded dice. You throw it twice and record the sum. Which of the following statements is true?

# In[23]:


# You can use this cell for your calculations (not graded)
def simulate_loaded_dice(n, loaded_side, num_trials=100000):
    np.random.seed(0)
    p = [2/7 if i == loaded_side else 1/7 for i in range(1, n+1)]
    first_throw = np.random.choice(np.arange(1, n+1), p=p, size=num_trials)
    second_throw = np.random.choice(np.arange(1, n+1), p=p, size=num_trials)
    sum_throws = first_throw + second_throw

    mean = np.mean(sum_throws)
    variance = np.var(sum_throws)

    return mean, variance

for loaded_side in range(1, 7):
    mean, variance = simulate_loaded_dice(6, loaded_side)
    print(f"For a 6-sided dice with side {loaded_side} loaded:")
    print(f"Mean of the sum: {mean:.3f}")
    print(f"Variance of the sum: {variance:.3f}\n")


# In[24]:


# Run this cell to submit your answer
utils.exercise_10()


# ## Exercise 11:
# 
# Given a fair n-sided dice. You throw it twice and record the sum but the second throw depends on the result of the first one such as in exercises 7 and 8. Which of the following statements is true?

# In[25]:


# You can use this cell for your calculations (not graded)
np.random.seed(0)
num_trials = 1000000

first_throw = np.random.choice(range(1, 7), size=num_trials)

second_throw1 = np.where(first_throw <= 3, np.random.choice(range(1, 7), size=num_trials), 0)
cov1 = np.cov(first_throw, second_throw1)[0, 1]

second_throw2 = np.where(first_throw >= 3, np.random.choice(range(1, 7), size=num_trials), 0)
cov2 = np.cov(first_throw, second_throw2)[0, 1]

print(f"Covariance when second throw is made if first throw is <= 3: {cov1:.3f}")
print(f"Covariance when second throw is made if first throw is >= 3: {cov2:.3f}")


# In[26]:


# Run this cell to submit your answer
utils.exercise_11()


# ## Exercise 12:
# 
# Given a n-sided dice (could be fair or not). You throw it twice and record the sum (there is no dependance between the throws). If you are only given the histogram of the sums can you use it to know which are the probabilities of the dice landing on each side?
# 
# In other words, if you are provided with only the histogram of the sums like this one:
# <td> <img src="./images/hist_sum_6_side.png" style="height: 300px;"/> </td>
# 
# Could you use it to know the probabilities of the dice landing on each side? Which will be equivalent to finding this histogram:
# <img src="./images/fair_dice.png" style="height: 300px;"/>
# 

# In[27]:


# You can use this cell for your calculations (not graded)
import numpy as np
import matplotlib.pyplot as plt

def simulate_dice_throws(n, num_trials=100000):
    np.random.seed(0)
    first_throw = np.random.choice(np.arange(1, n+1), size=num_trials)
    second_throw = np.random.choice(np.arange(1, n+1), size=num_trials)
    sum_throws = first_throw + second_throw

    plt.hist(sum_throws, bins=np.arange(1.5, 2*n+1.5), density=True, alpha=0.7)
    plt.title("Histogram of Sum")
    plt.xlabel("Sum of dice")
    plt.ylabel("Probability")
    plt.show()

n = 6
simulate_dice_throws(n)


# In[30]:


# Run this cell to submit your answer
utils.exercise_12()


# ## Before Submitting Your Assignment
# 
# Run the next cell to check that you have answered all of the exercises

# In[31]:


utils.check_submissions()


# **Congratulations on finishing this assignment!**
# 
# During this assignment you tested your knowledge on probability distributions, descriptive statistics and visual interpretation of these concepts. You had the choice to compute everything analytically or create simulations to assist you get the right answer. You probably also realized that some exercises could be answered without any computations just by looking at certain hidden queues that the visualizations revealed.
# 
# **Keep up the good work!**
# 
