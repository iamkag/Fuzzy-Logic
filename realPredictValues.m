function realPredictValues(yreal,ypredict)

trueList=zeros(5,1);
falseList = zeros(5,1);

for i=1:length(yreal)
    value = yreal(i);
    if yreal(i) == ypredict(i)
        trueList(value) = trueList(value) + 1;
        
    else
        falseList(value) = falseList(value) + 1;
      
    end
end

x = [1 2 3 4 5]
vals = [trueList(1) falseList(1);
        trueList(2) falseList(2);
        trueList(3) falseList(3);
        trueList(4) falseList(4);
        trueList(5) falseList(5)];
b = bar(x, vals);
set(b, {'DisplayName'}, {'True','False'}');
legend();
end
