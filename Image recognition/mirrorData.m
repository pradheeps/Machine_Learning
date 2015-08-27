load ./data/cross.mat
ref=size(img);
for i=1:3700
    temp=img(1,:,:,:);
    size(temp)
    [m,n]=size(temp)
    temp=reflection(img(i,:,:,:),32);
    
end
