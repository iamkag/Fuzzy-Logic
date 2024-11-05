function [equalSepB] = splitFreq(trnData,validationData,tstData, listOfClasses)

    tol = 0.05;
    trnCounter = zeros(1, length(listOfClasses));
    valCounter = zeros(1, length(listOfClasses));
    tstCounter = zeros(1, length(listOfClasses));
    
    trnFreq = zeros(1, length(listOfClasses));
    valFreq = zeros(1, length(listOfClasses));
    tstFreq = zeros(1, length(listOfClasses));
    
    equalSep = zeros(1, length(listOfClasses));
    
    for i=1:length(listOfClasses)
        trnCounter(1, i ) = sum(trnData(:,end)==listOfClasses(i));
        valCounter(1, i ) = sum(validationData(:,end)==listOfClasses(i));
        tstCounter(1, i ) = sum(tstData(:,end)==listOfClasses(i));
    end
    
    for i=1:length(listOfClasses)
        trnFreq(1, i) = trnCounter(1, i) / sum(trnCounter);
        valFreq(1, i) = valCounter(1, i) / sum(valCounter);
        tstFreq(1, i) = tstCounter(1, i) / sum(tstCounter);
        
        if (abs(trnFreq(1, i) - valFreq(1, i)) < tol) && (abs(trnFreq(1, i) - tstFreq(1, i)) < tol) && (abs(valFreq(1, i) - tstFreq(1, i)) < tol)
            equalSep(1, i) = 1;
        end
    end
    
    if sum(equalSep) == length(listOfClasses)
        equalSepB = true;
    else
        equalSepB = false;
    end

end