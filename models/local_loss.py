


class GramMatrix(nn.Module):
    def forward(self, input):
        a, b, c, d = input.size()
        # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)

        features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL
        G = torch.mm(features, features.t())  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(a * b * c * d)


# base style loss
class Base_StyleLoss(nn.Module):
    def __init__(self):
        super(Base_StyleLoss, self).__init__()
        self.gram = GramMatrix()
        self.criterion = nn.MSELoss()

    def __call__(self, input, target):
        input_gram = self.gram(input)
        target_gram = self.gram(target)
        loss = self.criterion(input_gram, target_gram)
        return loss


# define the style loss
class StyleLoss(nn.Module):
    def __init__(self, device, opt):
        super(StyleLoss, self).__init__()
        self.device = device
        self.vgg19 = VGG19().to(device)
        self.vgg19.load_model(opt.vgg_font)
        self.vgg19.eval()
        self.vgg_layers = ['conv2_2', 'conv3_2']
        self.criterion = Base_StyleLoss()

    def __call__(self, input, target):
        loss = 0.0
        for layer in self.vgg_layers:
        	inp_feat = self.vgg19(input)[self.vgg_layer]
        	tar_feat = self.vgg19(target)[self.vgg_layer]
        	loss += self.criterion(inp_feat, tar_feat)
        return loss