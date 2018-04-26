function main(responseIndex, numStartStations, stationInitScheme, R, lambda, eta, beta)

% responseIndex:
% 1 : tmax; 2: tmin; 3: tmean; 4: precip
% stationInitScheme: 1 - random initilize index, 2 - use cluster centroids
% after do a clustering method. This index is returned from a function
% called getStationInitIndex

targetNames = {'tmax', 'tmin', 'tmean', 'prcp'};
X1=[]; X2=[]; Y=[];stationLat=[];stationLon=[];
load(['../../data/', targetNames{responseIndex}, '_Deseason_std.mat']);

[S, T, d1] = size(X1);
[S, T, d2] = size(X2);


TrainingSize = 120;
ValidationSize = 120;
TestingSize = T - TrainingSize - ValidationSize;

select_rate = double(T)/(T+S);
% run learning method
% rng(0);

% randomly choose whether to incremental over time or space
[InitialStations, addStations] = getStationInitIndex(stationLat, stationLon, numStartStations, stationInitScheme, randSeed);
pAll = ones(S+T - numStartStations - 1, 1);
pAll(randperm(length(pAll), T-1)) = 0;
pAll = [pAll;0]; % manually set the last update is time

X1_random = X1([InitialStations; addStations], :, :);
X2_random = X2([InitialStations; addStations], :, :);
Y_random = squeeze(Y([InitialStations; addStations], :));
Y_hat = NaN(S, T);
spaceIndex = (1 : length(InitialStations))';

MAE_valid = NaN(1, ValidationSize);
MAE_test = NaN(1, TestingSize);
MAE_test_station = NaN(S,TestingSize);


t = 0;
s = length(spaceIndex);
% randomly initialize the models
A = rand(s, R); B = rand(t, R); C1 = rand(d1, R); C2 = rand(d2, R);
W1 = rand(R,d1); V1 = rand(R,d1);
W2 = rand(R,d2); V2 = rand(R,d2);
pIndex = 1;
while s <S || t < T
    p = pAll(pIndex);
    pIndex = pIndex + 1;
    tos = ' ';
    if p <= select_rate
        tos = 'time';
        
    else
        tos = 'space';
    end
    
    if s >= S || t == 0
        tos = 'time';
    end
    if t >= T
        tos = 'space';
    end
    if strcmp(tos, 'time')
        % fprintf('t');
        % incremental over time
        %             fprintf('Incremental over time\n');
        % increase t
        t = t + 1;
        % prepare data
        X1_T = squeeze(X1_random(spaceIndex, t, :));
        X2_T = squeeze(X2_random(spaceIndex, t, :));
        Y_T = squeeze(Y_random(spaceIndex, t));
        % update models
        %             fprintf(['update models for t + 1 = ' num2str(t) ' ']);
        % call update model method for incremental over time
        % preUpdate
        BT = muscat_incremental_sparsa_time_preUpdate(X1_T, X2_T, A, C1, C2, R, lambda, beta);
        % do prediction on X1_T
        Y_hat(spaceIndex,t) = (sum(X1_T .* bsxfun(@plus, A * W1, BT'*V1),2)) + (sum(X2_T .* bsxfun(@plus, A * W2, BT'*V2),2));
        MAE_local = mean(abs(Y_hat(spaceIndex,t) - Y_T));
        % record the loss
        % if t is in validation period
        if t > TrainingSize && t <= TrainingSize + ValidationSize
            MAE_valid(t-TrainingSize) = MAE_local;
        end
        % if t is in testing period
        if t > TrainingSize + ValidationSize
            MAE_test(t-TrainingSize - ValidationSize) = MAE_local;
            MAE_test_station(spaceIndex, t - TrainingSize - ValidationSize) = abs(Y_hat(spaceIndex,t) - Y_T);
        end
        
        %             fprintf(['MAE = ' num2str(MAE_local) ', lambda = ' num2str(lambda_local)...
        %                 ', eta = ' num2str(eta_local) ', beta = ' num2str(beta_local) '\n']);
        % postUpdate
        [W1, W2, V1, V2, A, BT, C1, C2] = muscat_incremental_sparsa_time_postUpdate...
            (X1_T, X2_T, Y_T, W1, W2, V1, V2, A, C1, C2, lambda, eta, beta, R);
        B = [B; BT'];
    elseif strcmp(tos, 'space')
        % fprintf('s');
        % incremental over space
        %             fprintf('Incremental over space\n');
        % incease s
        s = s + 1;
        % prepare data
        newStationIndex = s;
        spaceIndex = [spaceIndex; newStationIndex];
        X1_S =reshape( X1_random(newStationIndex, 1:t, :), [], d1);
        X2_S =reshape( X2_random(newStationIndex, 1:t, :), [], d2);
        Y_S = reshape(Y_random(newStationIndex, 1:t), [], 1);
        % update models
        %             fprintf(['update models for s + 1 = ' num2str(s) '\n']);
        % call update modele method for incremental over space
        [W1, W2, V1, V2, AS, B, C1, C2] = muscat_incremental_sparsa_space...
            (X1_S, X2_S, Y_S, W1, W2, V1, V2, B, C1, C2, lambda, eta, beta, R);
        A = [A; AS'];
    end
    
end
% record the models
models.A = A; models.B = B; models.C = C; models.W = W; models.V = V;


save(['MUSCAT-' num2str(responseIndex) '-' num2str(numStartStations) '.mat'], ...
    'Y', 'MAE_valid', 'MAE_test', 'MAE_test_station', ...
    'lambda', 'eta', 'beta', 'R');