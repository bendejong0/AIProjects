import numpy as np
import threading
import time

class MyThread(threading.Thread):
    def __init__(self, func):
        threading.Thread.__init__(self)
        self.func = func

    def run(self):
        print("Training... \n")
        self.func.train_new()

class costs():
    # our cost function
    def cost(self, guess, answer):
        return 0.5 * np.square(answer - guess)

    # the derivative of our cost function
    def derivative_cost(self, guess, answer):
        return guess - answer

# houses = [squarefoot]
# answer = [price]
houses = np.array([1135, 1280, 1323, 1200, 1808, 1522, 2500, 1743, 900]) # square footage of houses I found in Villa Park
prices = np.array([245000, 319700, 270000, 299900, 369000, 329000, 449900, 350000, 199900]) # the prices of the said houses

epochs = 1000
learning_rate = 0.001

class training():
    def __init__(self):
        np.random.seed(1)
        self.weight = np.random.randint(0, 100) * 2 - 1  # Random weight between 0 and 100
        self.firstweight = self.weight

    def train(self):
        for epoch in range(epochs):
            for i in range(len(houses)):
                guess = self.weight * houses[i]
                gradient = costs().derivative_cost(guess, prices[i])
                self.weight -= learning_rate * gradient

            print(f"Epoch: {epoch + 1} out of {epochs}")
        print("Current Weight:", self.weight, " Initial Weight:", self.firstweight)

    def train_new(self):
        # using this function, we can train over the new data
        for i in range(1):
            for i in range(len(houses)):
                guess = self.weight * houses[i]
                gradient = costs().derivative_cost(guess, prices[i])
                self.weight -= learning_rate * gradient

    def get_weight(self):
        return self.weight

# Create the training object and thread once
my_training = training()
my_thread = MyThread(my_training)

# Train the model using the initial data
my_thread.start()
my_thread.join()  # Wait for the thread to finish before starting a new one

# Allow the user to provide additional inputs
while True:
    print("\n---Model is ready!---\n")

    user = int(input("Insert the square footage of a house in Villa Park! \n"))
    if user == -1:
        print("Quitting...")
        break
    if user < 0:
        print(f"{user} is not an option. Please provide a valid input. \n")
        user = int(input("Insert the square footage of a house in Villa Park! \n"))

    houses = np.append(houses, user)
    prediction = my_training.weight * user

    print("House's price: $", prediction)
    real_price = int(input("What was the actual price? \n"))
    if real_price < 0:
        print(f"{real_price} is not an option. Please provide a valid input. \n")
        real_price = int(input("What was the actual price? \n"))
    prices = np.append(prices, real_price)

    # Train the model with the new data
    my_thread = MyThread(my_training)
    my_thread.start()
    my_thread.join()  # Wait for the thread to finish before starting a new one
    time.sleep(2)
