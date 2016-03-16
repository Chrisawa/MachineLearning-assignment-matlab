function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

add = 0;
reg = 0;
for i = 1:m
    add = add + y(i, 1) * log(sigmoid(theta' * X(i, :)')) + (1 - y(i, 1)) * log(1 - sigmoid(theta' * X(i, :)'));
end

for l = 1:size(X, 2)
    if l == 1
        reg = reg + 0;
    else
        reg = reg + theta(l) * theta(l);
    end
end

J = -(add / m) + ((lambda * reg) / (2 * m));

for j = 1:size(X, 2)
    summ = 0;
    for k = 1:m
        summ = summ + (sigmoid(theta' * X(k, :)') - y(k, 1)) * X(k, j);
    end
    if j == 1
        grad(j, 1) = summ / m;
    else
        grad(j, 1) = summ / m + lambda * theta(j) / m;
    end
end

% =============================================================

end
