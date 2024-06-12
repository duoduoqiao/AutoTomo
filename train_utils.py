import os
import torch
import numpy as np

from torch import nn
from pathlib import Path
from tqdm.auto import tqdm
from torch.optim import Adam


def cycle(dl):
    while True:
        for data in dl:
            yield data


class Trainer_autotomo(object):
    def __init__(self, model, rm, data_loader, results_folder='./check_points', train_lr=1e-4, known_id=[],
                 train_epoch=1000, batch_size=64, patience=100, loss_type='mse', weight=1.):
        super().__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.rm, self.known_id, self.weight = rm.to(self.device), known_id, weight

        self.dl = cycle(data_loader)
        self.num_samples = len(data_loader.dataset)
        self.milestone_cycle = int(self.num_samples // batch_size)
        self.train_num_steps = train_epoch * self.milestone_cycle

        self.opt = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=train_lr)
        if loss_type == "l1norm":
            self.criteon = nn.L1Loss().to(self.device)
        elif loss_type == "mse":
            self.criteon = nn.MSELoss().to(self.device)

        self.results_folder = Path(results_folder)
        os.makedirs(self.results_folder, exist_ok=True)

        self.counter = 50
        self.early_stopping = EarlyStopping(patience=int(patience), path=os.path.join(results_folder, 'AutoTomo_best.pt'),verbose=False)
        self.step = 0

    def loss_func(self, output, target, input):
        all_flows = np.arange(input.shape[-1])
        unknown_id = np.setdiff1d(all_flows, self.known_id)

        output_update = torch.empty_like(output, device=self.device)
        output_update[:, unknown_id] = output[:, unknown_id]
        output_update[:, self.known_id] = target[:, self.known_id]

        loss1 = self.criteon(output[:, self.known_id], target[:, self.known_id])
        loss2 = self.criteon(output_update @ self.rm, input)

        return loss1 + self.weight * loss2
    
    def load(self, mode='best'):
        assert mode in ['best', 'last'], 'Sorry, there is only the best checkpoint and last checkpoint saved during training!'
        if mode == 'best':
            self.model.load_state_dict(torch.load(os.path.join(self.results_folder, 'AutoTomo_best.pt')))
        else:
            self.model.load_state_dict(torch.load(os.path.join(self.results_folder, 'AutoTomo_last.pt')))

    def train(self):
        device = self.device
        total_loss = []
        counter_loss = 0

        with tqdm(initial=self.step, total=self.train_num_steps) as pbar:

            while self.step < self.train_num_steps:
                x, y = next(self.dl)
                x, y = x.to(device), y.to(device)

                h_hat, _ = self.model(x)
                loss = self.loss_func(h_hat, y, x)
                loss.backward()
                counter_loss += loss.item() / self.counter
                total_loss.append(loss.item())
                if self.step % self.counter == 0:
                    pbar.set_description(f'loss: {counter_loss:.6f}')
                    counter_loss = 0

                self.opt.step()
                self.opt.zero_grad()
                self.step += 1

                with torch.no_grad():
                    if self.step != 0 and self.step % self.milestone_cycle == 0:
                        self.early_stopping(np.average(total_loss), self.model)
                        if self.early_stopping.early_stop:
                            pbar.close()
                            print(f"Early Stopping at Epoch {int(self.step//self.milestone_cycle)}.")
                            break
                        else:
                            total_loss = []

                pbar.update(1)

        print('Training Complete')
        torch.save(self.model.state_dict(), os.path.join(self.results_folder, 'AutoTomo_last.pt'))
        self.load('best')

    @torch.no_grad()
    def estimate(self, data_loader, use_em=False):
        test_loss = []
        # counter_loss = 0
        estimations = np.empty([0, data_loader.dataset.feat_dim])

        # with tqdm(initial=0, total=len(data_loader.dataset)) as pbar:
        for idx, (x, y) in enumerate(data_loader):
            x, y = x.to(self.device), y.to(self.device)
            h_hat, _ = self.model(x)
            if use_em:
                h_hat = traffic_em(h_hat, y, self.rm, 5, self.device)
            estimations = np.row_stack([estimations, h_hat.cpu().numpy()])
            test_loss_y = self.criteon(h_hat, y)
            test_loss.append(test_loss_y.item())
                # counter_loss += test_loss_y.item() / self.counter
                # if self.step % self.counter == 0:
                #     pbar.set_description(f'Error: {test_loss_y.item():.6f}')
                # pbar.update(1)

        # pbar.close()
        test_loss = np.average(test_loss)
        print('Testing Mean Error:', test_loss.item())
        return estimations
        

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def em_step(x, y, rm, n, device):
    x = x.clamp_min_(1e-6).to(device)
    rm = rm.to(device)
    minloss = 100
    x_final = x
    for i in range(n):
        temp_y = x @ rm
        temp_y = temp_y.clamp_min_(1e-6)
        a = torch.div(x, rm.sum(dim=1))
        b = torch.div(rm, temp_y)
        c = y @ rm @ torch.transpose(b, 0, 1)
        x = torch.mul(a, c)
        temploss = torch.abs(x @ rm - y @ rm )
        temploss = temploss.sum(dim=0)

        # x_final = x
        if temploss < minloss:
            minloss = temploss
            x_final = x

        # return x_final
    return x_final

def replace_neg(x):
    x1 = x.cpu().numpy()
    x_0_idx = np.argwhere(x1 < 0)
    x1[x_0_idx] = 1
    xmin = np.min(x1)
    x[x_0_idx] = torch.tensor(xmin)
    return x

def traffic_em(flows_loads, links_loads, rm, n, device):
    b_size = flows_loads.shape[0]
    flows_loads_final = flows_loads.clone()

    for i in range(b_size):
        flows_loads_i = em_step(flows_loads[i], links_loads[i], rm, n, device)
        flows_loads_final[i] = flows_loads_i

    return flows_loads_final  