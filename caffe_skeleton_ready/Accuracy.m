List=strcat('test.csv');
M=importdata(List);
GT2=M.data(:,1);
Prob2 = M.data(:,2);
[X,Y,~,auc] = perfcurve(GT2,Prob2,1);
plot(X,Y);
xlabel('False positive rate')
ylabel('True positive rate')
title('ROC for Classification')
set(gca,'XLim',[0 1]);
set(gca,'YLim',[0 1]);
saveas(gcf,'AUC.png');
auc