function  [center, label] = k_means ( data, k, center0)
    [n, p ] = size(data); 
    
    max_iter = 1000; 
    disMat   = zeros(  n, k);
    label    = zeros( n,1) ; 
    center   = center0;
    epsilon  = .001;
    
    for t = 1 : max_iter
        % update label
        for i = 1 : k 
            c = center(i,:);
            disMat(:,i) =  sqrt(sum(( data - repmat(c, n,1)).^2,2));

        end 
        
        [val, label] =  min (disMat, [],2);
        
        % update center
        center_pre = center;
        for i = 1 : k 
            idx = (label == i); 
            center (i,:) =  sum ( data.* repmat(idx, 1,p))/ sum(idx); 
        end 
        
        if ( norm( center- center_pre, 'fro')/sqrt(k) < epsilon) 
            fprintf('total iteractions: %i\n',t);
            break;
        end 
    end

end