in a first approach to the use of reinforcement learning for driving our vehicle, I started developing a Q-learning solution. This algorithm provides a novel approach to controlling the vehicle based on the output from our established lane detection system.

Q-learning is a value-based reinforcement learning algorithm used widely in machine learning. This algorithm enables an agent to learn an optimal policy, a mapping of observed environment states to actions, in controlled Markovian domains.

Our lane detection system, using the robust Convolutional Neural Networks (CNN), gives us a detected lane from which we aim to identify the lane's center. We then declare distinct states in our Q-learning model based on where the lane's center could be found in the camera's image.

A critical aspect of the Q-learning algorithm is its rewards and penalties mechanism. Our system rewards the model when the center of the image and the lane align as closely as possible. Conversely, we penalize the model when the centers diverge, incentivizing the vehicle to maintain its position within the lane, thus promoting safer and more efficient autonomous driving.

## Qlearning specifications
The vehicle has five possible actions it can take: advancing forward and four distinct turning actions, two for each side with varying intensities. This configuration balances decision simplicity and the necessary control level to navigate the driving scenarios effectively.

To better tune our Q-learning model, we use the following configuration parameters:

    Learning Rate (alpha): 0.5
    Discount Factor (gamma): 0.95
    Exploration Factor (epsilon): 0.95

The learning rate determines to what extent the newly acquired information will override the old information. A factor of 0 will make the agent not learn anything, while a factor of 1 would make the agent consider only the most recent information. We've set our learning rate at 0.5, allowing for a balanced mix of old and new information.

The discount factor influences the weight of future rewards in the total reward calculation. A factor of 0 will make the agent "short-sighted" by only considering current rewards, while a factor approaching 1 will make it strive for a long-term high reward. A value of 0.95 was chosen for our discount factor, which encourages long-term reward maximization.

Finally, the exploration factor determines the likelihood of the agent taking a random action. A high value encourages more exploration, thereby allowing the model to learn more about its environment, while a low value promotes exploiting learned behaviors. Our value of 0.95 indicates a strong focus on exploration, enabling our model to learn more effectively.

By using Q-learning, our vehicle learns to maximize its total reward, in this case, keeping the car centered within the lane. It demonstrates how the vehicle can adapt and make optimal decisions based on past experiences, showcasing the power of reinforcement learning in achieving truly autonomous driving.
## Problems and solutions
