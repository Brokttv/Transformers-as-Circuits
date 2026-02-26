def train(model,dataloader,loss_fn,optimizer,device):
  model.train()
  total_loss = 0
  for x,y in dataloader:
    x,y = x.to(device), y.to(device)
    optimizer.zero_grad()
    output,att_weight= model(x)
    loss = loss_fn(output,y.unsqueeze(1))
    loss.backward()
    optimizer.step()
    total_loss +=loss.item()
  att_weight = att_weight
  train_loss = total_loss / len(dataloader)
  return train_loss,att_weight



def test(model,dataloader,device):
  with torch.inference_mode():
    model.eval()
    correct = 0
    for x,y in dataloader:
      x,y = x.to(device),y.to(device)
      pred,_ = model(x)
      output = (torch.sigmoid(pred)>0.5).float()

      correct += (output==y.unsqueeze(1)).sum().item()

    acc  = (correct / len(dataloader.dataset)) *100
    return acc
