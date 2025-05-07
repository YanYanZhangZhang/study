import torch
torch.backends.cudnn.enabled = False
from .base_model import BaseModel
from . import networks

import numpy
import cv2
from torch import nn
from torchvision import transforms
from skimage import morphology
import clip
import cn_clip.clip as cn_clip
from cn_clip.clip import load_from_name, available_models
import torch.nn.functional as F
# ziji-xunlian
from cn_clip.clip.model import convert_weights, CLIP
from cn_clip.training.main import convert_models_to_fp32
from cn_clip.eval.data import get_eval_img_dataset, get_eval_txt_dataset


def process(generated_images):
    gujia_img = torch.cat([generated_images]*3,dim=1)
    generated_images = (generated_images + 1.) / 2.  # 16.3.256.256
    """ resize to 224 """
    generated_images = F.interpolate(generated_images, size=224, mode='bicubic', align_corners=True)  # 16,3,224,224
    """ clip noralization """
    generated_images = clip_normalize(generated_images, device="cuda")
    return generated_images

def clip_normalize(image, device="cuda"):
    image = F.interpolate(image,size=224,mode='bicubic')
    mean=torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(device)
    std=torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(device)
    mean = mean.view(1,-1,1,1)
    std = std.view(1,-1,1,1)

    image = (image-mean)/std
    return image


class StsFontModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # changing the default values
        parser.set_defaults(norm='batch', netG='STS_MLAN', dataset_mode='font')
        
        if is_train:
            parser.set_defaults(batch_size=32, pool_size=0, gan_mode='hinge', netD='basic_64')
            # parser.add_argument('--lambda_L1', type=float, default=10.0, help='weight for L1 loss')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--lambda_style', type=float, default=1.0, help='weight for style loss')
            parser.add_argument('--lambda_content', type=float, default=1.0, help='weight for content loss')
            parser.add_argument('--dis_2', default=True, help='use two discriminators or not')
            parser.add_argument('--use_spectral_norm', default=True)

        return parser

    def __init__(self, opt):
        """Initialize the font_translator_gan class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.style_channel = opt.style_channel
        self.pre = True
        self.text_sim = opt.text_sim
            
        if self.isTrain:
            self.dis_2 = opt.dis_2
            self.visual_names = ['gt_images', 'generated_images']+['style_images_{}'.format(i) for i in range(self.style_channel)]
            self.base_prompt = '汉字的笔画是{}'
            # self.base_prompt = '汉字的笔画有{}'
            self.prompt = ["㇋", "㇆", "亅", "乛", "一", "𠃍", "㇃", "㇁", "𠄌", "ㄥ", "乚", "ㄣ", "ノ", "㇀",
                           "ㄋ", "㇅", "丶", "㇊", "㇏", "∟", "乙", "㇉", "㇜", "丨", "㇌", "㇎", "㇈", "ㄑ"]
            self.all_prompt = [self.base_prompt.format(i) for i in self.prompt]
            if self.dis_2:
                self.model_names = ['G', 'D_content', 'D_style']
                self.loss_names = ['G_GAN', 'G_L1', 'D_content', 'D_style', 'similarity' ]
            else:
                self.model_names = ['G', 'D']
                self.loss_names = ['G_GAN', 'G_L1', 'D']
        else:
            self.visual_names = ['gt_images', 'generated_images']
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.sanet, self.style_channel + 1, 1, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, )
        # self.clip_model, preprocess = clip.load('ViT-B/32', device=self.device)
        self.clip_model = CLIP(embed_dim=512,image_resolution=224,vision_layers=12,vision_width=768,vision_patch_size=16,vocab_size=21128,
                     text_attention_probs_dropout_prob=0.1,text_hidden_act='gelu',text_hidden_dropout_prob=0.1,text_hidden_size=768,
                     text_initializer_range=0.02,text_intermediate_size=3072,text_max_position_embeddings=512,text_num_attention_heads=12,
                     text_num_hidden_layers=12,text_type_vocab_size=2)
        convert_weights(self.clip_model)
        self.clip_model.to(self.device)
        checkpoint = torch.load('./pretrained_weights/checkpoints/epoch.pt', map_location='cpu')
        # start_epoch = checkpoint["epoch"]
        sd = checkpoint["state_dict"]
        if next(iter(sd.items()))[0].startswith('module'):
            sd = {k[len('module.'):]: v for k, v in sd.items() if "bert.pooler" not in k}
        self.clip_model.load_state_dict(sd)

        # self.cn_clip, preprocess = load_from_name('ViT-B-16', device=self.device)
        # self.clip_model = self.clip_model.to(self.device)
        """ freeze the network parameters """
        # for clip_param in self.clip_model.parameters():
        #     clip_param.requires_grad = False

        if self.isTrain:  # define discriminators; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            if self.dis_2:
                self.netD_content = networks.define_D(2, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids, use_spectral_norm=opt.use_spectral_norm)
                self.netD_style = networks.define_D(self.style_channel+1, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids, use_spectral_norm=opt.use_spectral_norm)
            else:
                self.netD = networks.define_D(self.style_channel+2, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids, use_spectral_norm=opt.use_spectral_norm)
            
        if self.isTrain:
            # define loss functions
            self.lambda_L1 = opt.lambda_L1
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.criterion = nn.CosineSimilarity(dim=1)
            if self.dis_2:
                self.lambda_style = opt.lambda_style
                self.lambda_content = opt.lambda_content
                self.optimizer_D_content = torch.optim.Adam(self.netD_content.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_D_style = torch.optim.Adam(self.netD_style.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_D_content)
                self.optimizers.append(self.optimizer_D_style)
            else:
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_D)

    def image_reverse(self, input_image):
        input_image_cp = numpy.copy(input_image)  # 输入图像的副本
        # pixels_value_max = numpy.max(input_image_cp)  # 输入图像像素的最大值
        output_imgae = 255 - input_image_cp  # 输出图像
        # retVal, output_imgae = cv2.threshold(output_imgae, 1 +0, 0, cv2.THRESH_BINARY)

        return output_imgae

    def gujia(self, a):
        temp = []
        for i in a:
            # tensor_i = i.permute(1, 2, 0)
            # img = tensor_i.cpu().numpy()*255
            # print(img)
            out1 = transforms.ToPILImage()(i)
            img = numpy.array(out1)


            # ret, binary = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
            #
            # # binary,img = numpy.expand_dims(binary,axis=2), numpy.expand_dims(img,axis=2)
            # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            # dict_i = cv2.dilate(binary, kernel, iterations=1)
            #
            # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            # dict_i = cv2.dilate(dict_i, kernel, iterations=1)
            #
            # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            # img = cv2.erode(dict_i, kernel, iterations=1)

            # print(img.size())
            # luokuo = cv2.Canny(img, 1, 255)
            cv2.bitwise_not(img, img)
            # cv2.bitwise_not(luokuo, luokuo)
            retVal, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
            img = numpy.expand_dims(img,axis=2)
            skeleton = morphology.skeletonize(img)
            # print(skeleton.shape)
            # cv2.imwrite('./gugjia.png', skeleton)
            skeleton = numpy.transpose(skeleton,(2,0,1))
            # skeleton = cv2.cvtColor(skeleton, cv2.COLOR_RGB2GRAY)
            # luokuo = numpy.expand_dims(luokuo, axis=0)
            # skeleton = numpy.expand_dims(skeleton, axis=0)  # 将（128，128）拓展成（128，128，1）

            skeleton = self.image_reverse(skeleton)
            # print(skeleton)
            temp.append(skeleton)
            # temp1.append(luokuo)

        out_bian = torch.Tensor(temp)
        # out_bian1 = torch.Tensor(temp1)
        # out_bian = out_bian.permute(0, 3, 1, 2)
        # for ii in out_bian1:
        #     out1 = transforms.ToPILImage()(ii)
        #     out1.show()
        return out_bian

    def set_input(self, data):
        self.gt_images = data['gt_images'].to(self.device)
        self.content_images = data['content_images'].to(self.device)
        self.style_images = data['style_images'].to(self.device)
        
        if not self.isTrain:
            # self.content_paths = data['content_paths']
            # self.gt_paths = data['gt_paths']
            if self.text_sim:
                self.sou_hde = data['sou_hde'].to(self.device)
                self.sty_hde = data['sty_hde'].to(self.device)
                self.image_paths = data['image_paths']
                self.style_char = data['style_char']
                self.style_source_image = data['style_source_image']

            else: self.image_paths = data['image_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if self.text_sim:
            self.generated_images = self.netG((self.content_images, self.style_images))
            self.style_images_tmp = self.style_source_image.view(-1,1,64,64)
            self.contest_sim_for =torch.zeros([1])
            for i in range(6):
                self.generated_images_style, self.cnt_fea_fake_style, self.content_feature_style = self.netG((self.style_images_tmp[i].view(1, 1, 64, 64), self.style_images))
                self.contest_sim_for += torch.cosine_similarity(self.content_feature[-1].view(-1), self.content_feature_style[-1].view(-1),dim=0)
            self.content_feature_mean= torch.div(self.contest_sim_for,6)
        else:
            if self.isTrain:
                self.clip_model.eval()
                # tokens = clip.tokenize(self.prompt).to(self.device)
                tokens = cn_clip.tokenize(self.all_prompt).to(self.device)
                # self.text_feature = self.clip_model.encode_text(tokens).detach()
                self.text_feature = self.clip_model(None, tokens)
                # text_features /= text_features.norm(dim=-1, keepdim=True)

            # self.gujia_img = self.gujia(self.content_images)
            # self.generated_images = self.netG((self.content_images, self.style_images, self.gujia_img))
            self.generated_images = self.netG((self.content_images, self.style_images))

            # self.text_feature = self.image_feature_r @ self.text_feature.t()
            # self.generated_images = self.netG((self.content_images, self.style_images))
        
    def compute_gan_loss_D(self, real_images, fake_images, netD):
        # Fake
        fake = torch.cat(fake_images, 1)
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real = torch.cat(real_images, 1)
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss
        loss_D = (loss_D_fake + loss_D_real) * 0.5
        return loss_D
    
    def compute_gan_loss_G(self, fake_images, netD):
        fake = torch.cat(fake_images, 1)
        pred_fake = netD(fake)
        loss_G_GAN = self.criterionGAN(pred_fake, True, True)
        return loss_G_GAN
    
    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        if self.dis_2:
            self.loss_D_content = self.compute_gan_loss_D([self.content_images, self.gt_images],  [self.content_images, self.generated_images], self.netD_content)
            self.loss_D_style = self.compute_gan_loss_D([self.style_images, self.gt_images], [self.style_images, self.generated_images], self.netD_style)

            self.clip_model.eval()
            self.generated_images111 = self.generated_images.detach()
            self.g_images, self.r_images = process(self.generated_images111), process(self.gt_images)
            self.image_feature_g, self.image_feature_r= self.clip_model(self.g_images,None), self.clip_model(self.r_images,None)
            # chuli image_features /= image_features.norm(dim=-1, keepdim=True)
            # self.image_feature_g, self.image_feature_r= self.cn_clip.encode_image(self.g_images), self.cn_clip.encode_image(self.r_images)
            similarity_g = F.cosine_similarity(self.image_feature_g.unsqueeze(1),self.text_feature.unsqueeze(0),dim=-1)
            similarity_r = F.cosine_similarity(self.image_feature_r.unsqueeze(1),self.text_feature.unsqueeze(0),dim=-1)
            self.loss_similarity = (torch.mean((1-0.5)*similarity_g.pow(2)+0.5*(torch.clamp(0.5-similarity_g,min=0).pow(2))) \
                                   + torch.mean((1-0.5)*similarity_r.pow(2)+0.5*(torch.clamp(0.5-similarity_r,min=0).pow(2)))) *0.5

            self.loss_D = self.lambda_content*self.loss_D_content + self.lambda_style*self.loss_D_style + self.loss_similarity
            # self.loss_D = self.lambda_content*self.loss_D_content + self.lambda_style*self.loss_D_style
        else:
            self.loss_D = self.compute_gan_loss_D([self.content_images, self.style_images, self.gt_images], [self.content_images, self.style_images, self.generated_images], self.netD)

        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        if self.dis_2:
            self.loss_G_content = self.compute_gan_loss_G([self.content_images, self.generated_images], self.netD_content)
            self.loss_G_style = self.compute_gan_loss_G([self.style_images, self.generated_images], self.netD_style)
            self.loss_G_GAN = self.lambda_content*self.loss_G_content + self.lambda_style*self.loss_G_style
            # self.loss_similarity = 1-self.criterion(self.image_feature_g,self.image_feature_r) + 1-self.criterion(self.image_feature_g,self.text_feature) # zijijia
            # similarity_g = F.cosine_similarity(self.image_feature_g.unsqueeze(1),self.text_feature.unsqueeze(0),dim=-1)
            # similarity_r = F.cosine_similarity(self.image_feature_r.unsqueeze(1),self.text_feature.unsqueeze(0),dim=-1)
            # self.loss_similarity = torch.mean((1-0.5)*similarity_g.pow(2)+0.5*(torch.clamp(0.5-similarity_g,min=0).pow(2))) \
            #                        + torch.mean((1-0.5)*similarity_r.pow(2)+0.5*(torch.clamp(0.5-similarity_r,min=0).pow(2))) # zijijia
        else:
            self.loss_G_GAN = self.compute_gan_loss_G([self.content_images, self.style_images, self.generated_images], self.netD)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.generated_images, self.gt_images) * self.opt.lambda_L1

        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        # self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_similarity
        self.loss_G.backward()
        
    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        if self.dis_2: #true
            self.set_requires_grad([self.netD_content, self.netD_style], True)
            self.optimizer_D_content.zero_grad()
            self.optimizer_D_style.zero_grad()
            self.backward_D()
            self.optimizer_D_content.step()
            self.optimizer_D_style.step()

        else:
            self.set_requires_grad(self.netD, True)      # enable backprop for D
            self.optimizer_D.zero_grad()             # set D's gradients to zero
            self.backward_D()                    # calculate gradients for D
            self.optimizer_D.step()                # update D's weights
        # update G
        if self.dis_2:
            self.set_requires_grad([self.netD_content, self.netD_style], False)
        else:
            self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()                  # set G's gradients to zero
        self.backward_G()                             # calculate graidents for G
        self.optimizer_G.step()                       # udpate G's weights

    def compute_visuals(self):
        if self.isTrain:
            self.netG.eval()
            with torch.no_grad():
                self.forward()
            for i in range(self.style_channel):
                setattr(self, 'style_images_{}'.format(i), torch.unsqueeze(self.style_images[:, i, :, :], 1))
            self.netG.train()
        else:
            pass