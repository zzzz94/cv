function [f,g] = softmax_regression_vec(theta, X, y)
  %
  % Arguments:
  %   theta - A vector containing the parameter values to optimize.
  %       In minFunc, theta is reshaped to a long vector.  So we need to
  %       resize it to an n-by-(num_classes-1) matrix.
  %       Recall that we assume theta(:,num_classes) = 0.
  %
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %
  m=size(X,2);% number of examples
  n=size(X,1);
  

  % theta is a vector;  need to reshape to n x num_classes.
  theta=reshape(theta, n, []);% n * num_classes-1
  num_classes=size(theta,2)+1;
  % initialize objective value and gradient.
  f = 0;
  g = zeros(size(theta));
  theta = [theta zeros(size(theta, 1), 1)];% add a column representing kth class, n * num_classes
  %
  % TODO:  Compute the softmax objective function and gradient using vectorized code.
  %        Store the objective function value in 'f', and the gradient in 'g'.
  %        Before returning g, make sure you form it back into a vector with g=g(:);
  %
%%% YOUR CODE HERE %%%
  A = exp(theta' * X);
  J = 0;
  for i = 1:m
      J = J + log(A(y(i), i)/sum(A(:, i)));
  end
  f = -J;
  
  for j = 1 : num_classes-1
      for i = 1:m
          if y(i) == j
              g(:, j) = g(:, j) + X(:, i) * (1 - A(j, i)/sum(A(:, i)));
              
          else
              g(:, j) = g(:, j) + X(:, i) * (0 - A(j, i)/sum(A(:, i)));
          end
      end
  end
  g = -g(:); % make gradient a vector for minFunc

