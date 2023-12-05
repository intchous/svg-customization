from torch.nn import functional as F
import kornia

dsample = kornia.geometry.transform.PyrDown()


def gaussian_pyramid_loss(recons, input):
    recon_loss = F.mse_loss(recons, input, reduction='none').mean(
        dim=[1, 2, 3])  # + self.lpips(recons, input)*0.1
    for j in range(2, 5):
        recons = dsample(recons)
        input = dsample(input)
        recon_loss = recon_loss + F.mse_loss(
            recons, input, reduction='none').mean(dim=[1, 2, 3]) / j
    return recon_loss
