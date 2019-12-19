import numpy as np
import pickle
import matplotlib.pyplot as plt


def loadfile():
    train_data = []
    train_label = []
    val_data = []
    val_label = []
    test_data = []
    test_label = []
    for i in range(1, 6):
        train_file = open(
            '/Users/limingxia/desktop/study/AI/cifar10/cifar-10-batches-py/data_batch_'+str(i), 'rb')
        train_file_object = pickle.load(train_file, encoding='bytes')
        for line in train_file_object[b'data']:
            train_data.append(line)
        for line in train_file_object[b'labels']:
            train_label.append(line)
    train_data = np.array(train_data)
    train_label = np.array(train_label)
    subtraintoval = np.random.choice(
        train_data.shape[0], int(0.1*train_data.shape[0]))
    print(len(subtraintoval))
    val_data = train_data[subtraintoval].astype("float")
    val_label = train_label[subtraintoval]
    train_data = np.delete(train_data, subtraintoval, 0)
    train_label = np.delete(train_label, subtraintoval, 0)
    train_data = train_data.astype("float")
    test_file = open(
        '/Users/limingxia/desktop/study/AI/cifar10/cifar-10-batches-py/test_batch', 'rb')
    test_file_object = pickle.load(test_file, encoding='bytes')
    # print(test_file_object)
    for line in test_file_object[b'data']:
        test_data.append(line)
    for line in test_file_object[b'labels']:
        test_label.append(line)
    test_data = np.array(test_data).astype("float")
    test_label = np.array(test_label)
    print("train_data  shape:" + str(train_data.shape))
    print("train_label shape:" + str(train_label.shape))
    print("val_data    shape:" + str(val_data.shape))
    print("val_label   shape:" + str(val_label.shape))
    print("test_data   shape:" + str(test_data.shape))
    print("test_label  shape:" + str(test_label.shape))
    return train_data, train_label, val_data, val_label, test_data, test_label


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3 * 32 * 32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - dropout: Scalar between 0 and 1 giving dropout strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian with standard deviation equal to   #
        # weight_scale, and biases should be initialized to zero. All weights and  #
        # biases should be stored in the dictionary self.params, with first layer  #
        # weights and biases using the keys 'W1' and 'b1' and second layer weights #
        # and biases using the keys 'W2' and 'b2'.                                 #
        ############################################################################
        self.params['W1'] = weight_scale * \
            np.random.randn(input_dim, hidden_dim)
        # print(W1.shape)
        self.params['b1'] = np.zeros(hidden_dim)
        # print(b1.shape)
        self.params['W2'] = weight_scale * \
            np.random.randn(hidden_dim, num_classes)
        self.params['b2'] = np.zeros(num_classes)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def loss(self, X, y=None, reg=0.0):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape

        # Compute the forward pass
        scores = None
        h1 = np.maximum(0, np.dot(X, W1) + b1)  # 隐藏层末端有 relu函数
        scores = np.dot(h1, W2) + b2

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        # computer the loss and gradient
        loss = None
        scores = scores - np.reshape(np.max(scores, axis=1), (N, -1))
        p = np.exp(scores) / \
            np.reshape(np.sum(np.exp(scores), axis=1), (N, -1))
        loss = -np.sum(np.log(p[range(N), list(y)])) / N
        loss += 0.5 * reg * np.sum(W1 * W1) + 0.5 * reg * np.sum(W2 * W2)

        # compute grads   这里的求导过程和之前的softmax是相似的，正确分类括号内减1 之后再3
        grads = {}
        dscores = p
        dscores[range(N), list(y)] -= 1.0
        dscores /= N
        dw2 = np.dot(h1.T, dscores)
        dh2 = np.sum(dscores, axis=0, keepdims=False)
        da2 = np.dot(dscores, W2.T)
        da2[h1 <= 0] = 0
        dw1 = np.dot(X.T, da2)
        dh1 = np.sum(da2, axis=0, keepdims=False)
        dw2 += reg * W2
        dw1 += reg * W1
        grads['W1'] = dw1
        grads['b1'] = dh1
        grads['W2'] = dw2
        grads['b2'] = dh2
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=1e-5, num_iters=100,
              batch_size=200, verbose=True):
        """
        train this neural network using SGD
        inputs:
        :param X: a numpy array of shape (N,D),giving training data
        :param y: a numpy array of shape (N,),giving training labels;y[i] =c means that x[i] has label.where 0<=c< C
        :param X_val: validation data,a numpy array of shape(N_val,D),
        :param y_val: validation label,a numpy array of shape(N_val,)
        :param learning_rate: 学习率
        :param learning_rate_decay:Scalar giving factor used to decay the learning rate after each epoch.
        :param reg:正则化的系数
        :param num_iters: 迭代次数
        :param batch_size: 每次执行的数据数量
        :param verbose:boolean if true print progress during optimizing
        :return:a dictionary{
            'loss history': loss_history,
            'train_acc_history':train_acc_history,
            'val_acc_history':val_acc_history,
            }
        """
        num_train = X.shape[0]
        iterations_per_epoch = max(
            num_train / batch_size, 1)  # 总的训练数据 除以 batch_size
        # 这里要理清楚这里面的关系，每次迭代中，所有的训练数据都要进行运算一遍的，那么每次迭代中，一个batch_size是200，所以大迭代中还要进行多次运算
        # 错错错，每次迭代，只用了batch_size大小的数据量来进行训练，这就是随机梯度法，通过一个较大的迭代次数，效果也行，而且计算代价降低了很多
        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for i in range(num_iters):   # 这是最外层的大的迭代次数
            X_batch = None
            y_batch = None
            indices = np.random.choice(num_train, batch_size)
            X_batch = X[indices]
            y_batch = y[indices]
            #########################################################################
            # TODO: Create a random minibatch of training data and labels, storing  #
            # them in X_batch and y_batch respectively.                             #
            #########################################################################

            #########################################################################
            #                             END OF YOUR CODE                          #
            #########################################################################

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y_batch, reg=reg)

            loss_history.append(loss)

            W1 = grads['W1']
            b1 = grads['b1']
            W2 = grads['W2']
            b2 = grads['b2']

            self.params['W1'] -= learning_rate * W1
            self.params['b1'] -= learning_rate * b1
            self.params['W2'] -= learning_rate * W2
            self.params['b2'] -= learning_rate * b2
            #########################################################################
            # TODO: Use the gradients in the grads dictionary to update the         #
            # parameters of the network (stored in the dictionary self.params)      #
            # using stochastic gradient descent. You'll need to use the gradients   #
            # stored in the grads dictionary defined above.                         #
            #########################################################################

            #########################################################################
            #                             END OF YOUR CODE                          #
            #########################################################################

            if verbose and i % 100 == 0:
                print('iteration %d / %d:loss %f' % (i, num_iters, loss))

            # every epoch, check train and val accuracy and decay learning rate
            if i % iterations_per_epoch == 0:  # 相当于minibatch经过多次迭代，数据量刚好达到训练数据量
                train_acc = np.mean(self.predict(X_batch) == y_batch)
                val_acc = np.mean(self.predict(X_val) == y_val)
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # decay learning rate
                learning_rate *= learning_rate_decay

        return {
            'loss_history': train_acc_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        """
        using trained weights of this two-layer network to predict labels for data points
        For each data point we predict scores for each of the C classes, and assign each data point to the class
        with the highest score.
        :param X: a numpy array of shape(N,D) giving N D-dimensional data points to classify
        :return:    y_pred: A numpy array of shape (N,) giving predicted labels for each of
                    the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
                    to have class c, where 0 <= c < C.
        """
        # 用上一步训练出来的权重矩阵来计算分数并进行分类
        y_pred = None
        h1 = np.maximum(0, np.dot(X, self.params['W1'])+self.params['b1'])
        scores = np.dot(h1, self.params['W2'])+self.params['b2']
        y_pred = np.argmax(scores, axis=1)
        ###########################################################################
        # TODO: Implement this function; it should be VERY simple!                #
        ###########################################################################

        ###########################################################################
        #                              END OF YOUR CODE                           #
        ###########################################################################
        return y_pred

# input_dim=3 * 32 * 32, hidden_dim=100, num_classes=10,
  #               weight_scale=1e-3, reg=0.0


input_dim = 32*32*3
hidden_dim = 100
num_classes = 10
neural_network = TwoLayerNet(input_dim, hidden_dim, num_classes)

X_train, y_train, X_val, y_val, X_test, y_test = loadfile()
# train the network
stats = neural_network.train(X_train, y_train, X_val, y_val, num_iters=30000,
                             batch_size=200, learning_rate=1e-4, learning_rate_decay=0.95,
                             reg=0.5, verbose=True)

# predict on the validation set
# plt.plot(stats['loss_history'])
#plt.title("loss history")
# plt.ylabel('Loss')
# plt.show()

val_acc = np.mean(neural_network.predict(X_val) == y_val)
print("val accuracy:", val_acc)

test_acc = np.mean(neural_network.predict(X_test) == y_test)
print("test accuracy:", test_acc)
