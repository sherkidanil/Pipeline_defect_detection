import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset

from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score

def train(model, opt, criterion, train_loader, val_loader, scheduler=None, 
             path ='/home/', name='name.pth',
             num_epochs=100, device = torch.device("cpu"), early_stop = 20,
             transformer: bool = False):
    logging = []
    best_val_loss = np.inf
    val_loss = 0
    f = 0
    pbar = tqdm(range(num_epochs))
    for epoch in pbar:
        model.train()
        
        train_batch_loss = []
        for step, batch in enumerate(train_loader):
            sample, target = batch
            sample, target = sample.to(device, torch.float), target.to(device, torch.float) 
            if transformer:
                pred = model(sample, target)
            else:
                pred = model(sample)
            loss = criterion(pred, target)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_batch_loss.append(loss.item())
        train_loss = np.mean(train_batch_loss)
    
        model.eval()
        
        val_batch_loss = []
        for batch in val_loader:
            sample, target = batch
            sample, target = sample.to(device, torch.float), target.to(device, torch.float)
            
            if transformer:
                pred = model(sample, target)
            else:
                pred = model(sample)
            loss = criterion(pred, target)
            val_batch_loss.append(loss.item())
            
        val_loss = np.mean(val_batch_loss)
          
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            folder = path
            model_path = folder + name
            torch.save(model, model_path)
        
        if scheduler is not None:
            scheduler.step()
            
        logging.append(np.array([epoch, train_loss, val_loss]))
        
        if f >= early_stop:
            print("Early stopping...")
            break

        pbar.set_description(f'epoch: {epoch}, train_loss: {train_loss}, val_loss {val_loss}')
    
    logging = np.array(logging)
    np.save('../logs/logging.npy',logging)
    
    model = torch.load(model_path)
    
    return model, logging


def test(model, test_loader, device):
        f1s = []
        model.eval()
        for batch in test_loader:
            sample, target = batch
            sample, target = sample.to(device, torch.float), target.to(device, torch.float)
            pred = model(sample)
            pred = torch.argmax(pred,1)
            f1 = f1_score(target.detach().cpu().numpy(), pred.detach().cpu().numpy())
            print(target.detach().cpu().numpy(), '\n', pred.detach().cpu().numpy())
            f1s.append(f1)
        return np.mean(f1s)