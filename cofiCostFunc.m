function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
Collaborative filtering cost function


% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));



J= (1/2)*sum(sum(((X*Theta'.*R) - Y).^2));
X_grad= (X*Theta'.*R - Y)*Theta;
Theta_grad= (X*Theta'.*R - Y)'*X;

J= (1/2)*sum(sum(((X*Theta'.*R) - Y).^2)) + (lambda/2)*sum(sum(Theta.^2)) + (lambda/2)*sum(sum(X.^2));


X_grad= (X*Theta'.*R - Y)*Theta + lambda*X;
Theta_grad= (X*Theta'.*R - Y)'*X + lambda*Theta;


grad = [X_grad(:); Theta_grad(:)];

end
