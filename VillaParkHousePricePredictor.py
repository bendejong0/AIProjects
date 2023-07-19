import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def cost(guess, answer):
    error.append(.5*np.square(answer-guess))
    return 0.5 * np.square(answer - guess)

def derivative_cost(guess, answer):
    return guess - answer

# houses = [squarefoot]
# answer = [price]
houses = np.array([1135, 1280, 1323, 1200, 1808, 1522, 2500, 1743, 900])
prices = np.array([245000, 319700, 270000, 299900, 369000, 329000, 449900, 350000, 199900])

average = 0
for i in range(len(prices)):
	average+=prices[i]
average=average/9

weight = np.random.randint(0, 100)
firstweight = weight

epochs = 1000
learning_rate = 0.001

for epoch in range(epochs):
    for i in range(len(houses)):
        guess = weight * houses[i]
        gradient = derivative_cost(guess, prices[i])
        weight -= learning_rate * gradient  # Use subtraction for gradient descent
    print(f"Epoch: {epoch + 1} out of {epochs}")


print("Current Weight: ", weight, " Initial Weight: ", firstweight)
print(f"Average house price: {average}")

while True:
    print("Try it!")
    user = int(input())
    prediction = weight * user
    print("House's price: $", prediction)
