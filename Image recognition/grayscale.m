function gray= grayscale(X_train)
    [r,c]=size(X_train);
    c2=c/3;
    gray=zeros(r,c2);

    for i=1:c2
        gray(:,i)= 0.33*X_train(:,i) + 0.33*X_train(:,i+1024) + 0.33*X_train(:,i+2048);
    end
end