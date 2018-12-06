import os
from itertools import islice

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


# test stage
for i, data in enumerate(islice(dataset, opt.num_test)):
    model.set_input(data)
    print('process input image %3.3d/%3.3d' % (i, opt.num_test))
    real_in, fake_out, real_out = model.test()
    images = [real_in, real_out, fake_out]
    names = ['input', 'ground truth', 'encoded']

    img_path = 'input_%3.3d' % i
    save_images(webpage, images, names, img_path, aspect_ratio=opt.aspect_ratio, width=opt.fineSize)

webpage.save()
