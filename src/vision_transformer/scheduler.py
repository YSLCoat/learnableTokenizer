import torch
from torch.optim.lr_scheduler import _LRScheduler, SequentialLR, CosineAnnealingLR

class LinearWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, lr_start, lr_stop, epochs, last_epoch=-1):
        assert 0 <= lr_start, "lr_start must be non-negative"
        assert 0 <= lr_stop, "lr_stop must be non-negative"
        self.lr_start = lr_start
        self.lr_stop = lr_stop
        self.epochs = epochs
        super(LinearWarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.epochs:
            alpha = self.last_epoch / self.epochs
            return [self.lr_start + alpha * (self.lr_stop - self.lr_start) for _ in self.base_lrs]
        else:
            return [self.lr_stop for _ in self.base_lrs]
        

def CosineAnnealingLR_LinearWarmup(optimizer, n_warmup_epochs, warmup_lr_start, warmup_lr_stop, T_max=20, eta_min=0.00001, last_epoch=-1):
    # Create the linear warmup scheduler
    linear_warmup_scheduler = LinearWarmupScheduler(optimizer, warmup_lr_start, warmup_lr_stop, n_warmup_epochs)
    
    # Create the cosine annealing scheduler
    cosine_annealing_scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

    # Combine the schedulers using SequentialLR
    scheduler = SequentialLR(optimizer, schedulers=[linear_warmup_scheduler, cosine_annealing_scheduler], milestones=[n_warmup_epochs])
    
    return scheduler

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from utils import calculate_warmup_epochs
    
    def plot_lr_scheduler(scheduler, num_epochs=30):
        lrs = []
        for epoch in range(num_epochs):
            scheduler.step()
            lr = scheduler.get_last_lr()[0]
            lrs.append(lr)

        plt.plot(range(num_epochs), lrs)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.show()
        
    # Example usage
    optimizer = torch.optim.SGD([torch.randn(3, 3)], lr=0.01)
    batch_size = 4096
    n_warmup_epochs = calculate_warmup_epochs(1200000, batch_size, 10000)
    print(n_warmup_epochs)
    scheduler = CosineAnnealingLR_LinearWarmup(optimizer, n_warmup_epochs, 0.000001, 0.1)

    # Assuming you have a DataLoader named train_loader
    # (You should replace this with your actual DataLoader)
    #train_loader = torch.utils.data.DataLoader(torch.randn(1200000), batch_size=batch_size)

    plot_lr_scheduler(scheduler, 300)
