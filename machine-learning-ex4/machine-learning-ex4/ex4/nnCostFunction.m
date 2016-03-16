function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

X = [ones(m, 1) X];
y_recoded = zeros(size(y, 1), num_labels);
for a = 1:m
    y_recoded(a, y(a)) = 1;
end

a2 = sigmoid(X * Theta1');
a2 = [ones(m, 1) a2];
a3 = sigmoid(a2 * Theta2');
sum_J = y_recoded .* log(a3) + (1 - y_recoded) .* log(1 - a3);
J_unreg = sum(sum(-(sum_J / m)));

sum_reg = sum(sum(Theta1(:, 2:end) .* Theta1(:, 2:end))) + sum(sum(Theta2(:, 2:end) .* Theta2(:, 2:end)));
reg = lambda * sum_reg / (2 * m);
J = J_unreg + reg;

Delta1 = zeros(size(Theta1));
Delta2 = zeros(size(Theta2));
for t = 1:m
% ----Feedforward
    a_1 = [X(t, :)];
%     size(a_1)
%     size(Theta1)
    z_2 = a_1 * Theta1';
    a_2 = sigmoid(z_2);
    a_2 = [1 a_2];
    z_3 = a_2 * Theta2';
    a_3 = sigmoid(z_3);
    
% ----Backpropagation
    d_3 = a_3 - y_recoded(t, :);
    d_2 = Theta2(:, 2:end)' * d_3' .* sigmoidGradient(z_2');
    Delta1 = Delta1 + d_2 * a_1;
    Delta2 = Delta2 + d_3' * a_2;
end
Theta1_grad = Delta1 / m;
Theta2_grad = Delta2 / m;

% ----Regularization
Theta1_reg = Theta1;
Theta2_reg = Theta2;
Theta1_reg(:, 1) = 0;
Theta2_reg(:, 1) = 0;
Theta1_grad = Theta1_grad + lambda * Theta1_reg / m;
Theta2_grad = Theta2_grad + lambda * Theta2_reg / m;
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
