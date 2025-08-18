import argparse
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from model import HC_model
import h5py
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def mape_loss(y_true, y_pred):
    # 计算绝对误差百分比，并避免除零
    epsilon = 1e-8  # 防止除以零的情况
    percentage_errors = torch.abs((y_true - y_pred) / (y_true + epsilon))

    # 计算平均值并乘以100（MAPE通常以百分比形式表达）
    return torch.mean(percentage_errors)
def min_max_normalize(data):
    min_val = data.min()
    max_val = data.max()
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data,min_val,max_val
def min_max_renormalize(data,min_val,max_val):
    renormalized_data = (data ) * (max_val - min_val)+ min_val
    return renormalized_data
def create_dataset(data, time_steps):
    X, y = [], []
    for i in range(len(data) - 2*time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps:i + 2*time_steps])
    return np.array(X), np.array(y)
def load_navier_stokes_data(path, sub=1, T_in=10, T_out=10, batch_size=5, reshape=None):
    ntrain = 4000
    neval = 400
    ntest = 400
    total = ntrain + neval + ntest
    with h5py.File(path, 'r') as f:
        data = f['data'][..., 0:total]#7220*2*32*32
    data = torch.tensor(data, dtype=torch.float32)
    normalized_data,min_val,max_val = min_max_normalize(data)
    time_steps = 12  # 用过去12个时间步作为输入
    X, y = create_dataset(normalized_data, time_steps) #normalized_data:7220   X: 7220*12*2*32*32  y:7220*12*2*32*32
    X=torch.tensor(X, dtype=torch.float32)
    y=torch.tensor(y, dtype=torch.float32)

    train_a, eval_a,test_a = X[:ntrain, :, :, :,:],X[ntrain:ntrain + neval, :, :, :,:], X[ntrain + neval:ntrain + neval+ntest, :, :, :,:]
    train_u,eval_u, test_u = y[:ntrain, :, :, :,:],y[ntrain:ntrain + neval, :, :, :,:], y[ntrain + neval:ntrain + neval+ntest, :, :, :,:]

    # eval_a = data[ntrain:ntrain + neval, :, :, :,:]
    # eval_u = data[ntrain:ntrain + neval, ::sub, ::sub, T_in:T_out+T_in]
    # eval_a = eval_a.unsqueeze(-1).permute(0, 3, 1, 2, 4).permute(0, 1, 4, 2, 3)
    # eval_u = eval_u.unsqueeze(-1).permute(0, 3, 1, 2, 4).permute(0, 1, 4, 2, 3)
    #
    # test_a = data[ntrain + neval:ntrain + neval + ntest, ::sub, ::sub, :T_in]
    # test_u = data[ntrain + neval:ntrain + neval + ntest, ::sub, ::sub, T_in:T_out+T_in]
    # test_a = test_a.unsqueeze(-1).permute(0, 3, 1, 2, 4).permute(0, 1, 4, 2, 3)
    # test_u = test_u.unsqueeze(-1).permute(0, 3, 1, 2, 4).permute(0, 1, 4, 2, 3)
    #
    # if reshape:
    #     train_a = train_a.permute(reshape)
    #     train_u = train_u.permute(reshape)
    #     eval_a = eval_a.permute(reshape)
    #     eval_u = eval_u.permute(reshape)
    #     test_a = test_a.permute(reshape)
    #     test_u = test_u.permute(reshape)

    train_loader = DataLoader(TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(TensorDataset(eval_a, eval_u), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False)

    return train_loader, eval_loader, test_loader,min_val,max_val

def evaluate_model(model, eval_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_loss2=0.0
    input_list = []
    output_list = []
    targets_list = []
    total_samples = 0
    with torch.no_grad():
        for inputs, targets in eval_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            input_list.append(inputs)
            targets_list.append(targets)
            outputs = model(inputs)
            output_list.append(outputs)
            loss = criterion(outputs, targets)
            loss2=mape_loss(outputs, targets)
            #outputs=min_max_renormalize(outputs)
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
            total_loss2 += loss2.item() * inputs.size(0)
            input_list_tensor = torch.cat(input_list)
            output_list_tensor = torch.cat(output_list)
            targets_list_tensor = torch.cat(targets_list)
    torch.save({'input': input_list_tensor, 'output': output_list_tensor, 'target': targets_list_tensor},
                       'tensors_taxi.pth')

    return total_loss / total_samples,total_loss2 / total_samples

def test_model(model, test_loader, criterion, device,min_val,max_val):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    inputs_bk=[]
    targets_bk=[]
    output_bk=[]
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs_bk.append(min_max_renormalize(inputs,min_val,max_val))
            targets_bk.append(min_max_renormalize(targets,min_val,max_val))
            outputs = model(inputs)
            output_bk.append(min_max_renormalize(outputs,min_val,max_val))
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

    return total_loss / total_samples

def train_model(model, train_loader, eval_loader, criterion, optimizer, device, num_epochs=150,min_val=0,max_val=0):
    best_loss = float('inf')
    best_loss,best_loss2=evaluate_model(model, eval_loader, criterion, device)
    print("first")
    print(best_loss)
    print(best_loss2)

    for epoch in range(50):
        model.train()
        total_loss = 0.0
        total_samples = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False)
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            preds = model(inputs)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
            progress_bar.set_postfix(loss=loss.item())
        average_loss = total_loss / total_samples
        print(f'Epoch {epoch + 1}, Train Loss: {average_loss:.7f}')
        eval_loss,_ = evaluate_model(model, eval_loader, criterion, device)
        print(f'Epoch {epoch + 1}, Eval Loss: {eval_loss:.7f}')
        if eval_loss < best_loss:
            best_loss = eval_loss
            print(f'New best model found at epoch {epoch + 1} with loss {best_loss:.7f}. Saving model...')
            torch.save(model.state_dict(), 'best_model_weights_taxi.pth')
    print("Training complete.")
def main(args):
    train_loader, eval_loader, test_loader, min_val,max_val= load_navier_stokes_data(
        path=args.data_path,
        sub=args.sub,
        T_in=args.T_in,
        T_out=args.T_out,
        batch_size=args.batch_size,
        reshape=args.reshape
    )

    model = HC_model(shape_in=(12, 2, 32, 32))  #shape_in=(shape_in=(10, 1, 64, 64))T C H W
    #criterion = nn.MSELoss()
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
   # model.load_state_dict(torch.load('best_model_weights_taxi.pth'))
    train_model(model, train_loader, eval_loader, criterion, optimizer, device, num_epochs=args.num_epochs,min_val=min_val,max_val=max_val)
    # Load the best model weights for testing
    model.load_state_dict(torch.load('best_model_weights_taxi.pth'))
    test_loss = test_model(model, test_loader, criterion, device,min_val,max_val)
    print(f'Test Loss: {test_loss:.7f}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a DGODE model on Navier-Stokes data.")
    parser.add_argument('--data_path', type=str, required=True, help='Path to the Navier-Stokes dataset.')
    parser.add_argument('--sub', type=int, default=1, help='Subsampling factor.')
    parser.add_argument('--T_in', type=int, default=10, help='Number of input time steps.')
    parser.add_argument('--T_out', type=int, default=10, help='Number of output time steps.')
    parser.add_argument('--batch_size', type=int, default=5, help='Batch size for training.')
    parser.add_argument('--reshape', type=int, nargs='+', help='Optional reshape permutation.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs.')

    args = parser.parse_args()
    main(args)
