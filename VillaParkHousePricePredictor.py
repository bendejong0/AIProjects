import numpy as np

#our cost function
def cost(guess, answer):
    return 0.5 * np.square(answer - guess)

#the derivative of our cost function
def derivative_cost(guess, answer):
    return guess - answer

# houses = [squarefoot]
# answer = [price]
houses = np.array([1135, 1280, 1323, 1200, 1808, 1522, 2500, 1743, 900]) # square footage of houses I found in Villa Park
prices = np.array([245000, 319700, 270000, 299900, 369000, 329000, 449900, 350000, 199900]) # the prices of the said houses

average = 0 #This is so that I can see the average house pricing.
for i in range(len(prices)):
	average+=prices[i]
average=average/9 #nine houses in our array

weight = np.random.randint(0, 100) #create a random weight
firstweight = weight #This is so that I can see how far the weight has moved

epochs = 1000 #make it iterate through all the data 1000 times
learning_rate = 0.001 #we want a very small learning rate because it's easy to overshoot with such large numbers

for epoch in range(epochs):
    for i in range(len(houses)):
        guess = weight * houses[i] #create a guess by multiplying a weight by a house's square footage
        gradient = derivative_cost(guess, prices[i]) # find the derivative of the cost function so that we know how to adjust our weights
        weight -= learning_rate * gradient  # use subtraction for gradient descent
	    				    # if gradient is negative, then we increase the weight to compensate.
    print(f"Epoch: {epoch + 1} out of {epochs}")


print("Current Weight: ", weight, " Initial Weight: ", firstweight)
print(f"Average house price: {average}")

#allows the user to give inputs to the algorithm 
while True:
    print("Try it!")
    user = int(input())
    prediction = weight * user
    print("House's price: $", prediction)
