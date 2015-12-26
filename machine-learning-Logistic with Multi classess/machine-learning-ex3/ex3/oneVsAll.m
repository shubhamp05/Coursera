function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%corresponds to the classifier for label i
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logisitc regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds 
%   to the classifier for label i

% Some useful variables
m = size(X, 1);
n = size(X, 2);

% You need to return the following variables correctly 
all_theta = zeros(num_labels, n + 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the following code to train num_labels
%               logistic regression classifiers with regularization
%               parameter lambda. 
%
% Hint: theta(:) will return a column vector.
%
% Hint: You can use y == c to obtain a vector of 1's and 0's that tell use 
%       whether the ground truth is true/false for this class.
%
% Note: For this assignment, we recommend using fmincg to optimize the cost
%       function. It is okay to use a for-loop (for c = 1:num_labels) to
%       loop over the different classes.
%
%       fmincg works similarly to fminunc, but is more efficient when we
%       are dealing with large number of parameters.
%
% Example Code for fmincg:
%
%     % Set Initial theta
%     initial_theta = zeros(n + 1, 1);
%     
%     % Set options for fminunc
%     options = optimset('GradObj', 'on', 'MaxIter', 50);
% 
%     % Run fmincg to obtain the optimal theta
%     % This function will return theta and the cost 
%     [theta] = ...
%         fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
%                 initial_theta, options);
%
% fprintf('%%%experiment begins%%%%');
% theta = zeros(size(X, 2), 1);
% grad = zeros(size(theta));
%disp(size(theta));
%for k=1:num_labels
 %   all_theta=sum((-(y==k)'.*(log(sigmoid(all_theta'*X'))))-(((1-(y==k)').*(log(1-sigmoid(all_theta'*X')))))/m;
   
%ans=(-(y==k)'.*(log(sigmoid(theta'*X'))))-((((1-(y==k))')).*(log(1-sigmoid(theta'*X'))));
%disp(ans)
%ans=log(1-sigmoid(theta'*X'));
%ans=(-(y==k)'.*(log(sigmoid(theta'*X'))));
%disp(size(ans));
% for k=1:num_labels
%     grad0=(((((sigmoid(X*theta))-(y==k))')*X(:,1))')/m;
%     regtheta=lambda/m*theta(2:end);
%     gradrest=(((((sigmoid(X*theta))-(y==k))')*X(:,2:end))')/m+regtheta;
%     grad(1)=grad0;
%     grad(2:end)=gradrest;
%     all_theta(k,1:end)=grad';
%     k=k+1;
% end

% Set Initial theta
    initial_theta = zeros(n + 1, 1);
    
    % Set options for fminunc
    options = optimset('GradObj', 'on', 'MaxIter', 50);
for c=1:num_labels
    % Run fmincg to obtain the optimal theta
    % This function will return theta and the cost 
    [theta] = ...
        fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
                initial_theta, options);
    all_theta(c,:)=(theta');
    c=c+1;
end    








% =========================================================================


end
