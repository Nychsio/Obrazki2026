import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleStructureTensor(nn.Module):
    def __init__(self, device=None):
        super(MultiScaleStructureTensor, self).__init__()
        
        # Filtry Sobela do liczenia gradientów (Ix, Iy)
        sobel_x = torch.tensor([[-1., 0., 1.], 
                                [-2., 0., 2.], 
                                [-1., 0., 1.]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1., -2., -1.], 
                                [0., 0., 0.], 
                                [1., 2., 1.]], dtype=torch.float32).view(1, 1, 3, 3)
        
        if device:
            sobel_x, sobel_y = sobel_x.to(device), sobel_y.to(device)
            
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        
        # Pooling dla artefaktów VAE (Stable Diffusion używa kompresji 8x8)
        self.pool_8 = nn.AvgPool2d(kernel_size=8, stride=8)
        self.pool_16 = nn.AvgPool2d(kernel_size=16, stride=16)

    def rgb_to_ycbcr(self, image):
        """Konwersja RGB do przestrzeni YCbCr (kluczowe dla wyłapywania błędów barw GAN/Diffusion)"""
        r, g, b = image[:, 0:1, :, :], image[:, 1:2, :, :], image[:, 2:3, :, :]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = -0.1687 * r - 0.3313 * g + 0.5 * b + 0.5
        cr = 0.5 * r - 0.4187 * g - 0.0813 * b + 0.5
        return y, cb, cr

    def compute_structure_tensor(self, channel, pool_layer):
        """Liczy macierz kowariancji gradientów i wyciąga cechy (Eigenvalues, Anisotropy)"""
        gx = F.conv2d(channel, self.sobel_x, padding=1)
        gy = F.conv2d(channel, self.sobel_y, padding=1)
        
        Ixx = gx * gx
        Iyy = gy * gy
        Ixy = gx * gy
        
        # Lokalne uśrednianie w oknach (Tensor Struktury)
        Vxx = pool_layer(Ixx)
        Vyy = pool_layer(Iyy)
        Vxy = pool_layer(Ixy)
        
        # Wartości własne (Eigenvalues lambda_1, lambda_2)
        trace = Vxx + Vyy
        det = (Vxx * Vyy) - (Vxy * Vxy)
        
        discriminant = torch.clamp(trace**2 - 4 * det, min=1e-6)
        sqrt_disc = torch.sqrt(discriminant)
        
        lambda_1 = (trace + sqrt_disc) / 2.0
        lambda_2 = (trace - sqrt_disc) / 2.0
        
        # Energia gradientu i Anizotropia (według paperów naukowych)
        energy = lambda_1 + lambda_2
        anisotropy = (lambda_1 - lambda_2) / (energy + 1e-5)
        
        return torch.cat([lambda_1, lambda_2, energy, anisotropy], dim=1)

    def forward(self, x):
        # 1. Rozbicie na YCbCr
        y, cb, cr = self.rgb_to_ycbcr(x)
        
        # 2. Cechy Tensora Struktury dla Luminancji (Y) w skalach 8x8 i 16x16
        features_y_8 = self.compute_structure_tensor(y, self.pool_8)
        features_y_16 = self.compute_structure_tensor(y, self.pool_16)
        
        # Żeby połączyć skale, "rozciągamy" mapę 16x16 do rozmiaru 8x8
        features_y_16_up = F.interpolate(features_y_16, size=features_y_8.shape[2:], mode='nearest')
        
        # 3. Zwykłe uśrednione gradienty dla Chrominancji (szukamy szumu barwnego w CbCr)
        grad_cb = self.pool_8(torch.abs(F.conv2d(cb, self.sobel_x, padding=1)) + torch.abs(F.conv2d(cb, self.sobel_y, padding=1)))
        grad_cr = self.pool_8(torch.abs(F.conv2d(cr, self.sobel_x, padding=1)) + torch.abs(F.conv2d(cr, self.sobel_y, padding=1)))
        
        # Złączenie w potężny 10-kanałowy tensor: [Lambda1_8, Lambda2_8, Energy_8, Aniso_8, Lambda1_16, Lambda2_16, Energy_16, Aniso_16, Grad_Cb, Grad_Cr]
        return torch.cat([features_y_8, features_y_16_up, grad_cb, grad_cr], dim=1)