function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    temp = zeros(size(X, 2), 1);
    for i = 1:(size(X, 2))
        summ = zeros(size(X, 2), 1);
        for j = 1:m    
	        summ(i, 1) = summ(i, 1) + (theta' * X(j, :)' - y(j, 1)) * X(j, i);
        end
            temp(i, 1) = theta(i, 1) - alpha * summ(i, 1) / m;
        
    end

    for k = 1:(size(X, 2))        
        theta(k, 1) = temp(k, 1);
    end



    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
