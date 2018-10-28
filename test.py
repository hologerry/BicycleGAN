import os

from data import CreateDataLoader
from models import create_model
from options.test_options import TestOptions
from util import html
from util.visualizer import save_images

# options
opt = TestOptions().parse()
opt.num_threads = 1   # test code only supports num_threads=1
opt.batch_size = 1   # test code only supports batch_size=1
opt.serial_batches = True  # no shuffle

# create dataset
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
model.setup(opt)
model.eval()
print('Loading model %s' % opt.model)

# create website
web_dir = os.path.join(opt.results_dir, opt.phase + '_sync' if opt.sync else opt.phase)
webpage = html.HTML(web_dir, 'Training = %s, Phase = %s, Class =%s' % (opt.name, opt.phase, opt.name))

# sample random z
if opt.sync:
    z_samples = model.get_z_random(opt.n_samples + 1, opt.nz)

# TODO: Update test

# test stage
for i, data in enumerate(dataset):
    model.set_input(data)
    print('process input image %3.3d/%3.3d' % (i, len(dataset)))
    if not opt.sync:
        z_samples = model.get_z_random(opt.n_samples + 1, opt.nz)
    for nn in range(opt.n_samples + 1):
        encode = nn == 0 and not opt.no_encode
        real_in, fake_out, real_out = model.test(z_samples[[nn]], encode=encode)
        if nn == 0:
            images = [real_in, real_out, fake_out]
            names = ['input', 'ground truth', 'encoded']
        else:
            images.append(fake_out)
            names.append('random_sample%2.2d' % nn)

    img_path = 'input_%3.3d' % i
    save_images(webpage, images, names, img_path, aspect_ratio=opt.aspect_ratio, width=opt.fineSize)

webpage.save()
