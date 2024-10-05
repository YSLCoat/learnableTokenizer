from timm.data import create_transform, Mixup

# Define mixup utility
def mixup_augmentation(n_classes):
    mixup_fn = Mixup(
        mixup_alpha=0.8,  # mixup alpha parameter
        cutmix_alpha=1.0, # optionally also apply CutMix
        prob=1.0,         # probability of applying mixup or cutmix
        switch_prob=0.5,  # probability of switching between mixup and cutmix
        mode='batch',     # mixup mode (batch or pairwise)
        label_smoothing=0.1,
        num_classes=n_classes
    )
    
    return mixup_fn