function p = predict(Theta1, Theta2, X)
%   PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

m = size(X, 1);
num_labels = size(Theta2, 1);

p = zeros(size(X, 1), 1);
X = [ones(m, 1) X];

for i = 1:m
  a_1 = X(i, :);
  z_2 = a_1 * Theta1';  
  
  a_2 = sigmoid(z_2');
  % Add a_0 bias value.
  a_2 = [ones(1,columns(a_2)) ; a_2]
  
  % Theta2 is 10 x 26 
  % a_2 is 26 x 1
  
  z_3 = Theta2 * a_2;   % 10 x 1
  a_3 = sigmoid(z_3);
  
  [value, index] = max(a_3)
  
  p(i) = index;
  
endfor
  

end
