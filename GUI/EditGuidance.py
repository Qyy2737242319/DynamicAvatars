import cv2
import torch
from numba.cuda.tests.nocuda.test_nvvm import original
import numpy as np
from threestudio.utils.misc import get_device, step_check, dilate_mask, erode_mask, fill_closed_areas
from threestudio.utils.perceptual import PerceptualLoss
import ui_utils
from threestudio.models.prompt_processors.stable_diffusion_prompt_processor import StableDiffusionPromptProcessor
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# Diffusion model (cached) + prompts + edited_frames + training config

def calculate_psnr_and_ssim(original, generated):
    with torch.no_grad():
        frame1 = torch.clamp(255 * original.detach(),min=0,max=255).cpu().numpy().astype(np.uint8).squeeze()
        frame2 = torch.clamp(255 * generated.detach(),min=0,max=255).cpu().numpy().astype(np.uint8).squeeze()
        psnr_value = psnr(frame1, frame2)
        ssim_value = ssim(frame1, frame2, channel_axis=2)
    return psnr_value, ssim_value


class EditGuidance:
    def __init__(self, guidance, gaussian, origin_frames, text_prompt, per_editing_step, edit_begin_step,
                 edit_until_step, lambda_l1, lambda_p, lambda_anchor_color, lambda_anchor_geo, lambda_anchor_scale,
                 lambda_anchor_opacity, train_frames, train_frustums, cams, server
                 ):
        self.guidance = guidance
        self.gaussian = gaussian
        self.per_editing_step = per_editing_step
        self.edit_begin_step = edit_begin_step
        self.edit_until_step = edit_until_step
        self.lambda_l1 = lambda_l1
        self.lambda_p = lambda_p
        self.lambda_anchor_color = lambda_anchor_color
        self.lambda_anchor_geo = lambda_anchor_geo
        self.lambda_anchor_scale = lambda_anchor_scale
        self.lambda_anchor_opacity = lambda_anchor_opacity
        self.origin_frames = origin_frames
        self.cams = cams
        self.server = server
        self.train_frames = train_frames
        self.train_frustums = train_frustums
        self.edit_frames = {}
        self.d_frames={}
        self.visible = True
        self.prompt_utils = StableDiffusionPromptProcessor(
            {
                "pretrained_model_name_or_path": "runwayml/stable-diffusion-v1-5",
                "prompt": text_prompt,
            }
        )()
        # self.standard_prompt = StableDiffusionPromptProcessor(
        #     {
        #         "pretrained_model_name_or_path": "runwayml/stable-diffusion-v1-5",
        #         "prompt": "turn the man ",
        #     }
        # )()
        self.perceptual_loss = PerceptualLoss().eval().to(get_device())


    def __call__(self, rendering, view_index, step):
        self.gaussian.update_learning_rate(step)

        # nerf2nerf loss
        if view_index not in self.edit_frames or (
                self.per_editing_step > 0
                and self.edit_begin_step
                < step
                < self.edit_until_step
                and step % self.per_editing_step == 0
        ):
            with torch.no_grad():
                result_diff = self.guidance(
                    rendering,
                    self.origin_frames[view_index],
                    self.prompt_utils,
                )

                self.d_frames[view_index] = result_diff["edit_images"].detach().clone()

            result = self.guidance(
                rendering,
                self.origin_frames[view_index],
                self.prompt_utils,
            )

            # cv2.imwrite(f"./perspect/{view_index}_result.jpg",cv2.cvtColor(255*result["edit_images"].detach().cpu().numpy().squeeze(), cv2.COLOR_BGR2RGB))
            # cv2.imwrite(f"./perspect/{view_index}_render.jpg",
            #             cv2.cvtColor(255 * rendering.detach().cpu().numpy().squeeze(), cv2.COLOR_BGR2RGB))
            # cv2.imwrite(f"./perspect/{view_index}_origin.jpg",
            #             cv2.cvtColor(255 * self.origin_frames[view_index].detach().cpu().numpy().squeeze(), cv2.COLOR_BGR2RGB))
            self.edit_frames[view_index] = result["edit_images"].detach().clone() # 1 H W C
            self.train_frustums[view_index].remove()
            self.train_frustums[view_index] = ui_utils.new_frustums(view_index, self.train_frames[view_index],
                                                                    self.cams[view_index], self.edit_frames[view_index], self.visible, self.server)
            # print("edited image index", cur_index)

        gt_image = self.edit_frames[view_index]
        d_image = self.d_frames[view_index]

        # loss = self.lambda_l1 * torch.nn.functional.l1_loss(rendering, gt_image) + \
        #        self.lambda_p * self.perceptual_loss(rendering.permute(0, 3, 1, 2).contiguous(),
        #                                             gt_image.permute(0, 3, 1, 2).contiguous(), ).sum()
        l1_loss = torch.nn.functional.l1_loss(rendering, gt_image)
        lpips = self.perceptual_loss(rendering.permute(0, 3, 1, 2).contiguous(),
                                     gt_image.permute(0, 3, 1, 2).contiguous(), ).sum()

        if step%100 == 0:
            psnr_value, ssim_value = calculate_psnr_and_ssim(gt_image, rendering)
            print(f"{step}_l1_loss:", l1_loss.item())
            print(f"{step}_lpips:", lpips.item())
            print(f"{step}_psnr:", psnr_value)
            print(f"{step}_ssim:", ssim_value)
            print("\n")
        loss = self.lambda_l1 * l1_loss + self.lambda_p * lpips

        # anchor loss
        if (
                self.lambda_anchor_color > 0
                or self.lambda_anchor_geo > 0
                or self.lambda_anchor_scale > 0
                or self.lambda_anchor_opacity > 0
        ):
            anchor_out = self.gaussian.anchor_loss()
            loss += self.lambda_anchor_color * anchor_out['loss_anchor_color'] + \
                    self.lambda_anchor_geo * anchor_out['loss_anchor_geo'] + \
                    self.lambda_anchor_opacity * anchor_out['loss_anchor_opacity'] + \
                    self.lambda_anchor_scale * anchor_out['loss_anchor_scale']

        return loss,rendering,gt_image,d_image

