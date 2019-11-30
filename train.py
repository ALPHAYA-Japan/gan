'''
    ------------------------------------
    Author : SAHLI Mohammed
    Date   : 2019-11-13
    Company: Alphaya (www.alphaya.com)
    Email  : nihon.sahli@gmail.com
    ------------------------------------
'''

import sys
# Traditional GANs
from src.GAN        import GAN
from src.BGAN       import BGAN
from src.FGAN       import FGAN
from src.WGAN       import WGAN
from src.LSGAN      import LSGAN
from src.DCGAN      import DCGAN
from src.DRAGAN     import DRAGAN
from src.WGAN_GP    import WGAN_GP
from src.SoftmaxGAN import SoftmaxGAN

# --------------------------------------Main-----------------------------------------
if __name__ == "__main__":
    models = {# Standard GANs
              "GAN"       : GAN,
              "BGAN"      : BGAN,
              "FGAN"      : FGAN,
              "WGAN"      : WGAN,
              "LSGAN"     : LSGAN,
              "DCGAN"     : DCGAN,
              "DRAGAN"    : DRAGAN,
              "WGAN_GP"   : WGAN_GP,
              "SoftmaxGAN": SoftmaxGAN}

    # ...........................................
    if len(sys.argv) < 3:
        print("command 1: python train.py GAN_type train")
        print("command 2: python train.py GAN_type generate")
        print("GAN_type can be",[a for a in models.keys()])
        sys.exit()

    model = sys.argv[1] #.upper()
    mode  = sys.argv[2]

    if model not in models:
        print(model,"not in",[a for a in models.keys()])
        sys.exit()
    elif mode not in ["train", "generate"]:
        print(mode,"not in",["train","generate"])
        sys.exit()

    # ...........................................
    data_path   = 'data/MNIST/train_data/'          # training data location (see README)
    model_path  = 'models/'+model+'_MNIST/'         # specify where you wanna save your model

    # ...........................................
    image_size = 32
    if mode == "train":
        gan = models[model](data_path  = data_path,
                            model_path = model_path,
                            is_training= True,     # Must be True for training
                            batch_size = 32,
                            latent_dim = 100,
                            image_size = image_size,
                            hard_load  = True,     # if True, load all images at once
                            pretrained = False,    # if True, load a pretrained model
                            verbose    = True)
        gan.train(max_epoches = 25,                # Maximum number of epochs
                  show_images = True)              # if True, you can see some generated images
                                                   # during training
    elif len(sys.argv) == 3:
        gan = models[model](model_path = model_path,
                            batch_size = 32,
                            latent_dim = 100,
                            image_size = image_size)
        gan.generate(samples    = 30,
                     grid_width = 480,
                     grid_height= 240,
                     destination= 'images/'+model+'/grid.png')
    else:
        print("command 1: python train.py GAN_type train")
        print("command 2: python train.py GAN_type generate")
        print("GAN_type can be",[a for a in models.keys()])
        sys.exit()
