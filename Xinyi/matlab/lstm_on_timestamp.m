function [net] = lstm_on_boolean()
%     opts = detectImportOptions("timestamp.csv");
%     opts.SelectedVariableNames = [2:11822];
%     T = readmatrix('timestamp.csv', opts);
    T = readmatrix('timestamp.csv');
    X = num2cell(T(:,2:end),2);
%     size(cell2mat(X(1,1)))
%     T = num2cell(T);
%     X = (T(:,1:3080));
%     opts.SelectedVariableNames = [1];
%     Y = readmatrix('timestamp.csv', opts);
    Y = T(:,1);
    l = size(Y, 1);
    XTrain = [];
    YTrain = [];
    for i = 1:l
        if Y(i) <= 1
            XTrain = [XTrain; num2cell(rmmissing(cell2mat(X(i))),2)];
            YTrain = [YTrain; Y(i)];
        end
    end
    l = size(YTrain, 1);
    idx = randperm(size(XTrain, 1));
    XTrain = XTrain(idx,:);
    YTrain = YTrain(idx);
    YTrain = categorical(YTrain);
%     XTrain = reshape(XTrain, [size(XTrain,2),size(XTrain,1)]);
    
    numTrain = floor(0.6*l);
    numVal = floor(0.7*l);
    XTest = XTrain(numVal+1:end);
    XVal = XTrain(numTrain+1:numVal+1);
    XTrain = XTrain(1:numTrain+1);
    YTest = YTrain(numVal+1:end);
    YVal = YTrain(numTrain+1:numVal+1)
    YTrain = YTrain(1:numTrain+1);
    inputSize = 1;
    numHiddenUnits = 20;
    numClasses = 2;
    
    layers = [ ...
        sequenceInputLayer(inputSize)
        bilstmLayer(numHiddenUnits,'OutputMode','sequence')
        dropoutLayer(0.4)
        reluLayer()
        bilstmLayer(numHiddenUnits,'OutputMode','sequence')
        dropoutLayer(0.4)
        reluLayer()
        convolution1dLayer(1,1)
        dropoutLayer(0.4)
        reluLayer()
        bilstmLayer(numHiddenUnits,'OutputMode','last')
        dropoutLayer(0.4)
        fullyConnectedLayer(numClasses)
        softmaxLayer
        classificationLayer]
    maxEpochs = 150;
    miniBatchSize = 27;
    
    options = trainingOptions('adam', ...%         'InitialLearnRate', 0.01, ...
        'ValidationData',{XVal,YVal}, ...
        'ValidationFrequency',20, ...
        'ExecutionEnvironment','cpu', ...%         'GradientThreshold',1, ...
        'MaxEpochs',maxEpochs, ...
        'MiniBatchSize',miniBatchSize, ...
        'SequenceLength','longest', ...
        'Shuffle','every-epoch', ...
        'Verbose',0, ...
        'Plots','training-progress');
    net = trainNetwork(XTrain,YTrain,layers,options);

    miniBatchSize = 27;
    YPred = classify(net,XTest, ...
        'MiniBatchSize',miniBatchSize, ...
        'SequenceLength','longest');
    
    acc = sum(YPred == YTest)./numel(YTest)
end