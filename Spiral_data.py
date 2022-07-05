import numpy as np
import matplotlib.pyplot as plt
import neuralnetwork as nnet

np.random.seed(0)


def spiral_data(points, classes):
    X = np.zeros((points * classes, 2))
    y = np.zeros(points * classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points * class_number, points * (class_number + 1))
        r = np.linspace(0.0, 1, points)  # radius
        t = np.linspace(class_number * 4, (class_number + 1) * 4, points) + np.random.randn(points) * 0.2
        X[ix] = np.c_[r * np.sin(t * 2.5), r * np.cos(t * 2.5)]
        y[ix] = class_number
    return X, y


n_classes = 3
lmbda = 0
X_train, y_train = spiral_data(200, n_classes)

nn = nnet.NeuralNetwork(2, [20, 20], n_classes, loss_object=nnet.Loss_CrossEntropy,
                        activation_object=nnet.Activation_Sigmoid, activation_output_layer=nnet.Activation_Sigmoid)

nn.train_SGD(X_train, y_train, 1, 3000, lmbda=lmbda, mini_batch_size=50, print_data=True)

outputs_train = nn.forward_propagate(X_train)
predictions_train = nn.predictions(outputs_train)
accuracy_train = nn.calculate_accuracy(predictions_train, y_train)
loss_train = nn.calculate_loss(outputs_train, y_train, lmbda=lmbda)

print(f'Train accuracy {(accuracy_train * 100):.2f}%')
print(f'Train loss {loss_train}')

np.random.seed(78)
X_test, y_test = spiral_data(100, n_classes)

outputs_test = nn.forward_propagate(X_test)
predictions_test = nn.predictions(outputs_test)
accuracy_test = nn.calculate_accuracy(predictions_test, y_test)
loss_test = nn.calculate_loss(outputs_test, y_test, lmbda=lmbda)

print(f'Test accuracy {(accuracy_test * 100):.2f}%')
print(f'Test loss {loss_test}')

fig, axes = plt.subplots(2, 2, constrained_layout=True)

axes[0, 0].scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='brg', s=6)
axes[0, 0].axis('equal')
axes[0, 0].set(title='Train data')

axes[1, 0].scatter(X_train[:, 0], X_train[:, 1], c=predictions_train, cmap='brg', s=6)
axes[1, 0].axis('equal')
axes[1, 0].set(title='Train result')

axes[0, 1].scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='brg', s=6)
axes[0, 1].axis('equal')
axes[0, 1].set(title='Test data')

axes[1, 1].scatter(X_test[:, 0], X_test[:, 1], c=predictions_test, cmap='brg', s=6)
axes[1, 1].axis('equal')
axes[1, 1].set(title='Test result')
plt.show()
