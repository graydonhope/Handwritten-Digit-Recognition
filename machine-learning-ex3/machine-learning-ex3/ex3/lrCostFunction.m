function [J, grad] = lrCostFunction(theta, X, y, lambda)
%   LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%   regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

m = length(y); % number of training examples

grad = zeros(size(theta));


h_theta = sigmoid(X * theta);
logged_h_theta = log(h_theta);

first_half = (-1*y) .* logged_h_theta;
second_half = (1 - y) .* log(1 - h_theta);

sum_values = first_half .- second_half ;
sum_elements = sum(sum_values);

vectorized_cost = (1 / m) * sum_elements;

theta_squared = theta(2:end) .^ 2;
sum_of_theta_squared = sum(theta_squared);
regularize = (lambda / (2 * m)) * sum_of_theta_squared;

regularized_cost = vectorized_cost + regularize;
J = regularized_cost;


j_0_gradient = (1 / m) * X' * (h_theta .- y);
grad(1) = j_0_gradient(1);

lambda_m = (lambda / m) * theta;


grad(2:end) = j_0_gradient(2:end) .+ lambda_m(2:end);

end
