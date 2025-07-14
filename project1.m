% Project: Applied Macroeconometrics
% Author: Pohsun Lo
% Country: Austria
%%
clear;
%% set up mydata
dt_IPindex = readtable("industrial_production_index.csv")
dt_unemploy = readtable("unemployment.csv","Delimiter",",")
dt_unemploy = dt_unemploy(strcmp(dt_unemploy.LOCATION, 'AUT'), :)

mydata = innerjoin(dt_IPindex, dt_unemploy,"Keys","TIME")
%%
% insert the MP shock
OIS_6M = readtable("MP_shock.csv")
mydata1 = innerjoin(mydata,OIS_6M,"Keys","TIME")

%%
% filter the data from 1999-2021
dates = split(mydata1.TIME, '-');
years = str2double(dates(:, 1));
months = str2double(dates(:, 2));
dates = table(years, months);
mydata2 = [mydata1,dates]

% clean other columns
keepColumns = {'TIME','Value_dt_IPindex', 'Value_dt_unemploy','OIS_6M','years','months'};
mydata3 = mydata2(:,keepColumns)
mydata3.log_dt_IPindex = log(mydata3.Value_dt_IPindex)

% 创建逻辑索引以筛选出年份在1999至2021之间的行
%idx = (years > 1999 & years < 2021) | (years == 1999 & months >= 1) | (years == 2021 & months <= 12);

% 使用逻辑索引筛选数据表的行
%mydata = mydata(idx, :);
%% add dummies to control time trend (if needed)
n = size(mydata3, 1)
date_dummies = zeros(n, 12); % should be (n,11): **revise later

% extract the 'year' and 'month'
years = table2array(mydata3(:, 5));
months = table2array(mydata3(:, 6));

for i = 1:n
    month = months(i);  % current month

    % compute index for the correspond month 
    index = (month - 1)

    % 将对应月份的虚拟变量设为 1
    date_dummies(i, index+1) = 1;
end

% Convert date_dummies matrix into a table
date_dummies_table = array2table(date_dummies, 'VariableNames', {'dummy_jan', 'dummy_feb', 'dummy_mar', 'dummy_apr', 'dummy_may', 'dummy_jun', 'dummy_jul', 'dummy_aug', 'dummy_sep', 'dummy_oct', 'dummy_nov', 'dummy_dec'});

mydata3_with_dummy = [mydata3,date_dummies_table]
%% plot
mydata3.TIME = datetime(mydata3.TIME, 'InputFormat', 'yyyy-MM')
figure
tiledlayout(4,1)
nexttile
plot(mydata3.TIME, mydata3.log_dt_IPindex,'r')
title('log industrial production index')
grid on
nexttile
plot(mydata3.TIME, mydata3.Value_dt_IPindex,'g')
title('industrial production index')
grid on
nexttile
plot(mydata3.TIME, mydata3.Value_dt_unemploy,'b')
title('unemplyment rate')
grid on
nexttile
plot(mydata3.TIME, mydata3.OIS_6M,'k')
title('OIS_6M')
grid on
%%
% first difference
% non-stationary to stationary
OIS = mydata3.OIS_6M(2:end)
logIPI= diff(mydata3.log_dt_IPindex)
diff_unemploy = diff(mydata3.Value_dt_unemploy)
% graph
data = array2timetable([OIS logIPI  diff_unemploy], 'RowTimes', mydata3.TIME(2:end), 'variableNames', {'OIS','logIPI','unemploy_rate'})
figure
plot(data.Time,OIS, 'r')
hold on
plot(data.Time,data.logIPI, 'b')
hold on
plot(data.Time,data.unemploy_rate, 'k')
grid on
legend('OLS','logIPI','unemploy_rate')
hold off
%% AIC % with ordering
orderingdata = [data.OIS, data.logIPI, data.unemploy_rate];

csvwrite('ordered.csv', orderingdata);


maxLag = 24;
minLag = 1;
numLags = maxLag - minLag + 1;
AIC_values = zeros(numLags, 1);

for lag = minLag:maxLag
    X = orderingdata(lag+1:end, :);
    Y = zeros(size(X, 1), size(X, 2) * lag);
    for i = 1:lag
        Y(:, (i-1)*size(X, 2)+1 : i*size(X, 2)) = orderingdata(lag-i+1:end-i, :);
    end
    coefficients = (Y' * Y) \ (Y' * X);
    residuals = X - Y * coefficients;
    sigma = cov(residuals) + eye(size(residuals, 2)) * 1e-6;
    try
        L = chol(sigma, 'lower'); % Cholesky 分解将实对称正定矩阵分解为下三角矩阵
    catch
        error("Cholesky decomposition failed. The covariance matrix may not be positive definite.");
    end
    transformed_residuals = residuals / L';
    sigma_transformed = cov(transformed_residuals);
    logL = -0.5 * (size(residuals, 1) * size(residuals, 2) * log(2*pi) + size(residuals, 1) * log(det(sigma_transformed)) + sum(sum(transformed_residuals / sigma_transformed * transformed_residuals')));
    numParams = numel(coefficients);
    AIC_values(lag) = 2 * numParams - 2 * logL;
end

% optimal p = 22
[~, minAIC_index] = min(AIC_values);
optimalLag = minAIC_index + minLag - 1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% VAR estimate (reduce form)
orderingdata = [data.OIS, data.logIPI, data.unemploy_rate];
trainData = orderingdata(1:end-22, :); % data before T-p 
testData = orderingdata(end-21:end, :); % data after 22 lags p:end

p = optimalLag; % lag 22
T = size(trainData, 1); % no. of data point
K = size(trainData, 2); % no. of input parameters

% structure martices
X = zeros(T-p, K*p+1);
Y = zeros(T-p, K);
for t = p+1:T
    for lag = 1:p
        X(t-p, (lag-1)*K+1 : lag*K) = trainData(t-lag, :);
    end
    X(t-p, K*p+1) = 1; % add intercept
    Y(t-p, :) = trainData(t, :);
end

% estimate the VAR parameters by OLS
df = size(X, 1) - size(X,2);

coefficients = (X' * X) \ (X' * Y);

% residuals represent the orthogonalized shocks
residuals = Y - X * coefficients;
%% Cholesky decomposition (triangular VAR model)
% variance and covariance of reduced form matrix
sigma = (residuals' * residuals) / (size(residuals, 1) - df);
num_variables = size(sigma, 1);

% transfer sigma to lower triangle matrix for identification
% inverse_B
inverse_B = chol(sigma, 'lower'); % setting lower to get lower triangle

% RF residuals variance and covariance matrix
transformed_residuals = residuals * inverse_B';

% Estimate the variance-covariance matrix of the ε_t
sigma_transformed = cov(transformed_residuals);

%% Calculate impulse responses (using SF, thus transform A and et fist)
% transfom RF to SF
% stucture_coefficient = coefficients / B
% stucture_residual = residuals / B;
% then we need to transform the VAR model to the Moving average model (present by the MA(infinity))
%% alternative (Lutz Kilian) impulse responses function
% restructure the impulse response matrix from RF to 'companion form'
% companion form: system of multivariate linear difference equations

A = coefficients(1:66, :)'; % RF without intercept to transform into the companion form

% size of companion form matrix
n = size(A, 1)  % no. of input parameters
p = optimalLag  % number of lags

% Create the companion matrix
companion_matrix = zeros(n*p);

% Fill the companion matrix with the elements
for i = 1:p
    companion_matrix((i - 1) * n + 1 : i * n, 1 : n) = A(:, (i - 1) * n + 1 : i * n);
end

% Set the diagonal factors as an identity matrix
for i = 1 : n
    companion_matrix(i : n : end - (n - i), end - (n - i) : n : end) = 1;
end

% OIS & IPI
% OIS & unemployrate
num_periods = 30;  % Specify the number of periods for which you want to compute the IRF
impulse_response = irfvar(companion_matrix,inverse_B,22,num_periods);
%% polt the impulse responses matrix

% 3 variables in this equation
num_variables = 3;

% plot
figure;
y_labels = {'MP_OIS', 'logIPI', 'Unemployment Rate'};

for i = 1:num_variables
    subplot(num_variables, 1, i);
    plot(1:size(impulse_response, 2), impulse_response(i, :), 'LineWidth', 2);
    xlabel('Periods');
    ylabel(y_labels{i});
end
% interpretation
% when a MPshock (OIS) changes, the IPI react one periods late

%% Bootstrapping method for standard error band

% resample times
num_bootstraps = 1000;

% obtain the info of dimension of impulse response function
num_variables = size(impulse_response, 1); % 3
num_periods = size(impulse_response, 2); % 31

% create a emppty table for saving data from each loop
bootstrap_responses = zeros(num_variables * num_periods, num_bootstraps);

% resampling
for b = 1:num_bootstraps
    % same size（replacement）
    sample_indices = randi(num_periods, num_periods, 1);
    bootstrap_sample = reshape(impulse_response(:, sample_indices), [], 1);
    
    % saving
    bootstrap_responses(:, b) = bootstrap_sample;
end

% covariance_matrix of impulse response of companion form on each time period
covariance_matrix_companion = cov(bootstrap_responses');

% calculate 'irf_errors' or SE
irf_errors = sqrt(diag(covariance_matrix_companion));
%%
num_variables = 3;

% y labels
y_labels = {'MP_OIS', 'logIPI', 'Unemployment Rate'};

% plot separtely
for i = 1:num_variables
    figure;
    plot_impulse_response(impulse_response, irf_errors, i, y_labels{i});
end

function plot_impulse_response(impulse_response, irf_errors, i, y_label)
    % plot the impulse response curve
    plot(1:size(impulse_response, 2), impulse_response(i, :), 'LineWidth', 2);
    hold on;
    
    % 95% confidence interval of SE
    upper_band = impulse_response(i, :) + 1.96 * irf_errors(i);
    lower_band = impulse_response(i, :) - 1.96 * irf_errors(i);
    
    % plot the CI
    fill([1:size(impulse_response, 2), size(impulse_response, 2):-1:1], [upper_band, fliplr(lower_band)], 'b', 'FaceAlpha', 0.2);
    
    % label
    xlabel('Periods');
    ylabel(y_label);
    
    % title and legend
    title(['Impulse Response of ' y_label]);
    legend('Impulse Response', '95% Confidence Interval');
end
