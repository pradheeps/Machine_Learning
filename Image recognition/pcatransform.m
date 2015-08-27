function x= pcatransform(X_train,y)
    co=cov(X_train);
    [u,s,v]=svd(co);
    x=u*s';

end