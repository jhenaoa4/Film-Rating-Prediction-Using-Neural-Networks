def shuffled_sampling(data, k, v):
  '''
    Este método hace un sampleo de los datos, separando en proporción k
    para entrenamiento y 1-k-v para test  y 1-k-v para validación.
    '''
  n = data.shape[0]
  m = int(k*n)
  l = int(v*n)
  data = data.sample(frac=1).reset_index(drop=True)
  data = data.sample(frac=1).reset_index(drop=True)
  train_data = data.iloc[:m,:]
  test_data = data.iloc[m:m+l,:]
  val_data = data.iloc[m+l:,:]

  return train_data, test_data, val_data