link_train =zeros(noOfUser,noOfUser); 
for i = 1:size(train_pair,1)
    link_train(find(userList == train_pair(i,1)),find(userList == train_pair(i,2))) = 1; 
    link_train(find(userList == train_pair(i,2)),find(userList == train_pair(i,1))) = 1; 
end

link_test =zeros(noOfUser,noOfUser); 
for i = 1:size(test_pair,1)
    link_test(find(userList == test_pair(i,1)),find(userList == test_pair(i,2))) = 1; 
    link_test(find(userList == test_pair(i,2)),find(userList == test_pair(i,1))) = 1; 
end

%compute auc value based on D
scores_googlenet2 = [];
y_googlenet2 = [];
for i=1:noOfUser
    for j=i+1:noOfUser
        if link_train(i,j)==0
            scores_googlenet2 = [scores_googlenet2;D(i,j)];
            y_googlenet2 = [y_googlenet2;link_test(i,j)];
        end
    end
end
[fpr,tpr,thresh,AUC] = perfcurve(y_googlenet2,scores_googlenet2,1)
[RECALL, PRECISION, INFO] = vl_pr(y_googlenet2-0.5,scores_googlenet2);
INFO.auc