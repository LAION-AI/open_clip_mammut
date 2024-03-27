import torch.nn as nn
import torch
import torch.nn.functional as F


def build_image_tokenizer(name="image_patch", **kwargs):
    if name == "image_patch":
        return ImagePatchTokenizer(**kwargs)
    elif name == "taming":
        return TamingTokenizer(**kwargs)
    elif name == "icetk":
        return IceTkTokenizer(**kwargs)
    else:
        raise NotImplementedError()


class BaseImageTokenizer():

    def __init__(self):
        super().__init__()
    
    def tokenize(self, x):
        raise NotImplementedError()
    
    def reconstruct_from_tokens(self, ids):
        raise NotImplementedError()
    
    @property
    def num_image_tokens(self):
        raise NotImplementedError()

    @property
    def dim(self):
        pass

class TamingTokenizer(BaseImageTokenizer):

    def __init__(self, config_path, model_path):
        from vqgan_nodep import VQModel
        from omegaconf import OmegaConf
        config = OmegaConf.load(config_path)
        model = VQModel(**config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(model_path)
        self.config = config
        self.model = model
    
    def tokenize(self, x):
        ids =  self.model.tokenize(x)
        return ids.view(len(x), -1)
    
    def reconstruct_from_tokens(self, ids):
        ids = ids.view(-1, 14, 14)
        return self.model.reconstruct_from_tokens(ids)

    @property
    def num_image_tokens(self):
        return self.config.model.params.n_embed

    @property
    def discrete(self):
        return True

    @property
    def needs_0_1(self):
        return True

class IceTkTokenizer(BaseImageTokenizer):

    def __init__(self, compress_rate=16):
        super().__init__()
        from icetk import IceTokenizer
        self.tokenizer = IceTokenizer()
        self.compress_rate = compress_rate
    
    @property
    def num_image_tokens(self):
        return self.tokenizer.num_image_tokens

    def tokenize(self, x):
        return self.tokenizer.encode(image_torch=x, compress_rate=self.compress_rate)
    

    def loss(self, pred, target):
        return F.cross_entropy(pred, target)
    
    def reconstruct_from_tokens(self, ids):
        return self.tokenizer.decode(image_ids=ids, compress_rate=self.compress_rate)

    @property
    def discrete(self):
        return True

    @property
    def needs_0_1(self):
        return True

class ImagePatchTokenizer(BaseImageTokenizer):

    def __init__(self, patch_size, normalize=False):
        super().__init__()
        self.patch_size = (patch_size)
        self.normalize = normalize

    def tokenize(self, x):
        n, c, h, w = x.shape
        npatch_h = h // self.patch_size
        npatch_w = w // self.patch_size
        x = x.view(n, c, npatch_h, self.patch_size, npatch_w, self.patch_size)
        x = x.permute(0, 2, 4, 1, 3, 5)
        x = x.contiguous()
        x = x.view(n, npatch_h * npatch_w, c, self.patch_size, self.patch_size)
        x = x.view(n, npatch_h * npatch_w, -1)
        if self.normalize:
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True)
            x = (x - mean) / (var + 1.e-6)**.5
        return x

    def reconstruct_from_tokens(self, ids):
        raise NotImplementedError()

    def loss(self, pred, target):
        return F.cross_entropy(pred, target)

    def loss(self, pred, target):
        return F.mean_squared_error(pred, target)

    @property
    def dim(self):
        return self.patch_size * self.patch_size * 3

    @property
    def num_image_tokens(self):
        raise NotImplementedError()

    @property
    def discrete(self):
        return False

    @property
    def needs_0_1(self):
        return False


if __name__ == "__main__":
    import torchvision
    from PIL import Image
    # test
    #tokenizer = TamingTokenizer("../../pretrained/vqgan_imagenet_f16_16384.yaml", "../../pretrained/vqgan_imagenet_f16_16384.ckpt")
    tokenizer = ImagePatchTokenizer(patch_size=32, normalize=True)
    #tokenizer = IceTkTokenizer(compress_rate=16)
    img = Image.open("dog.jpg")
    img = img.resize((224, 224))
    x = torchvision.transforms.ToTensor()(img)
    x = x.unsqueeze(0)
    tokens = tokenizer.tokenize(x)
    xr = tokenizer.reconstruct_from_tokens(tokens)
    xr = xr.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    xr = (xr * 255).astype("uint8")
    xr = Image.fromarray(xr)
    xr.save("dog_recon.jpg")
