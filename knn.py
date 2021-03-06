def distance(v1,v2):
    return np.sqrt(((v1-v2)**2).sum())


def knn(train,test,k=5):
    dist=[]
    
    for i in range(train.shape[0]):
        # get the vector and label
        ix=train[i,:-1]
        iy=train[i,-1]
        # compute the distance from the point
        d=distance(test,ix)
        dist.append([d,iy])
        #sort based on distance and get top k
    
    dk=sorted(dist,key=lambda x:x[0])[:k]
    
    #retrieve only the labels
    
    labels=np.array(dk)[:,-1]
    
    #gett freq of each labels
    
    output=np.unique(labels,return_counts=True)
    
    #find max frq and corresponding label
    index=np.argmax(output[1])
    return output[0][index]
