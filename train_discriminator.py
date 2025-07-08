import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from flame_scene.discriminator import Discriminator_512

model = Discriminator_512(3, 8).cuda()
D_optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
D_criterion = torch.nn.BCELoss()
train_iter = 500


D_optimizer.zero_grad()
d_true = model(gt_image.permute(0, 3, 1, 2))
d_false = model(rendering.permute(0, 3, 1, 2).detach())
true_label = torch.ones_like(d_true, device=d_true.device, dtype=d_true.dtype)
false_label = torch.zeros_like(d_false, device=d_false.device, dtype=d_false.dtype)
d_loss = D_criterion(d_true, true_label) + D_criterion(d_false, false_label)
d_loss.backward()
D_optimizer.step()

