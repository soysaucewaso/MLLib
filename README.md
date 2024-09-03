A neural network framework used to easily make large or small efficient networks.

Implements a model as a linked list of layers
Recursively forward and backward propagates through layers.

main.layer is an abstract class allowing for runtime polymorphism with different layer implementations.
So far I've only implemented dense and convolutional layers.

Input must be a Batch Size X Features numpy matrix for a dense layer
Must be a Batch Size X FeatureD1 X ... X FeatureDN X Channels
Example 2 layer model:
```
# most of these params have smart defaults and are explicity assigned to demonstrate functionality

# amtNeurons is output dimension
l1 = layers.dense(amtNeurons: 10, activation=main.layer.RELU)

# kernel is the dimensionality of the convolutions window
# normal is an optional batch normalization function
# weightinit is a weight initialization function, None chooses a default based on the activation function
# padb determines whether to pad the convolution so output dim is the same as the input (assuming stride is 1)
l2 = l1.append(layers.conv(inputD=(3), kernel = (2), stride = (1), normal = main.layer.ZMNORMAL, weightinit=main.layer.HENORM, padb=True))

# MSR is mean square error
model = main.model(rootLayer: l1, cost=main.layer.MSR,learningRate=.1,optimizer=ADAM)

# return array of cost at each iterations
costs = model.train(inputData,outputData,numIterations=100)

predicted = model.predict(testInput)
```

Implementation Notes:
Implements various functions as tuples of functions.
For instance: for every activation function a, a[0] is the activation function, while a[1] is it's derivative.
Polymorphism could've also been used but would've added unneccessary complexity
