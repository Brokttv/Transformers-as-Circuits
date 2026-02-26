

def data(seq_len:int, num_samples:int,Majority:bool):
  
  if Majority:
    X = np.random.randint(0,2,(num_samples,seq_len))
    y = (X.sum(axis=1) > seq_len //2).astype(int)
  else:
    X = np.zeros((num_samples,seq_len))
    X[0::2,:] = np.random.randint(0,2,(num_samples//2, seq_len))
    y = (X.sum(axis=1)>0).astype(int)
  
  return X,y

def get_datalaoders(batch_size, train_num_samples,train_seq_len,test_num_samples,test_seq_len,Majority:bool):
  
  #train data
  x_train,y_train = data(train_seq_len,train_num_samples, Majority = Majority)
  x_train = torch.from_numpy(x_train).long()
  y_train = torch.from_numpy(y_train).float()
  
  #test data
  x_test,y_test = data(test_seq_len,test_num_samples, Majority=Majority)
  x_test = torch.from_numpy(x_test).long()
  y_test= torch.from_numpy(y_test).float()
 
  #Loading data
  train_data = TensorDataset(x_train,y_train)
  test_data = TensorDataset(x_test,y_test)
  
  train_dataloader = DataLoader(train_data, batch_size = batch_size, drop_last = True shuffle=True)
  test_dataloader = DataLoader(test_data,batch_size = batch_size,drop_last=True, shuffle= False)
  
  return train_datalaoder, test_datalaoder


    
