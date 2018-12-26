import time

from data import CreateDataLoader
from models import create_model
from options.train_options import TrainOptions
from util.visualizer import Visualizer
from util.visualizer import save_images


if __name__ == '__main__':
    opt = TrainOptions().parse()
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)

    validate_opt = opt
    validate_opt.phase = 'val'
    validate_opt.num_threads = 1   # test code only supports num_threads=1
    validate_opt.batch_size = 1   # test code only supports batch_size=1
    validate_opt.serial_batches = True  # no shuffle
    val_data_loader = CreateDataLoader(opt)
    val_dataset = val_data_loader.load_data()
    val_dataset_size = len(val_data_loader)
    print('#validation images = %d' % val_dataset_size)

    model = create_model(opt)
    model.setup(opt)
    visualizer = Visualizer(opt)
    total_steps = 0

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        model.train()
        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_steps += opt.batch_size
            epoch_iter += opt.batch_size
            blk_epoch = False
            if opt.black_epoch_freq and epoch % opt.black_epoch_freq == 0:
                blk_epoch = True
            model.set_input(data, blk_epoch)
            if not model.is_train():
                continue

            model.optimize_parameters()

            if total_steps % opt.display_freq == 0:
                save_result = total_steps % opt.update_html_freq == 0
                visualizer.display_current_results(model.get_current_visuals(), epoch, total_steps, save_result)

            if total_steps % opt.print_freq == 0:
                losses = model.get_current_losses()
                t = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, opt, losses)

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save_networks('latest')

            iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save_networks('latest')
            model.save_networks(epoch)

        if opt.validate_freq > 0 and epoch % opt.validate_freq == 0:
            model.eval()
            for i, data in enumerate(val_dataset):
                model.set_input(data)
                ABC_path = data['ABC_path'][0]
                file_name = ABC_path.split('/')[-1].split('.')[0]
                real_in, fake_out_B, real_out_B, fake_out, real_out = model.test()
                images = [real_out, fake_out]
                names = ['ground_truth', 'encoded']

                img_path = str(epoch) + '_' + file_name
                save_images(images, names, img_path, opt=validate_opt, aspect_ratio=1.0, width=validate_opt.fineSize)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()
