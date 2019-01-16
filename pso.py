import random
import copy
import time

from data import CreateDataLoader
from models import create_model
from options.train_options import TrainOptions
from util.visualizer import Visualizer, save_images

from pso_helper import get_range_list, convert_hp_to_dict, print_options


class Particle:
    def __init__(self, particle_id, range_list, dim):
        self.particle_id = particle_id
        self.dim = dim
        self.position = []
        self.velocity = []
        self.range_list = range_list

        assert(len(self.range_list) == self.dim)

        for i in range(self.dim):
            self.position.append(random.randrange(self.range_list[i]))
            self.velocity.append(
                random.randint(-self.range_list[i]+1, self.range_list[i]-1))

        self.fitness = float('inf')
        self.fit_model(0)
        self.best_position = copy.deepcopy(self.position)
        self.best_fitness = self.fitness

    def fit_model(self, iter_id):
        # model train
        # model evaluation
        name = str(iter_id) + '_' + str(self.particle_id)
        checkpoints_dir = 'checkpoints/pso'
        opt = TrainOptions().parse()
        opt.name = name
        opt.checkpoints_dir = checkpoints_dir

        hp_opt_dict = convert_hp_to_dict(self.position, self.dim)
        for key, value in hp_opt_dict.items():
            setattr(opt, key, value)

        print_options(opt, hp_opt_dict)

        data_loader = CreateDataLoader(opt)
        dataset = data_loader.load_data()
        dataset_size = len(data_loader)
        print('#training images = %d' % dataset_size)

        validate_opt = copy.deepcopy(opt)
        validate_opt.phase = 'val'
        validate_opt.serial_batches = True  # no shuffle
        val_data_loader = CreateDataLoader(validate_opt)
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

            for i, data in enumerate(dataset):
                iter_start_time = time.time()
                if total_steps % opt.print_freq == 0:
                    t_data = iter_start_time - iter_data_time
                total_steps += opt.batch_size
                epoch_iter += opt.batch_size

                model.set_input(data)
                if not model.is_train():
                    continue

                model.optimize_parameters()

                if total_steps % opt.print_freq == 0:
                    losses = model.get_current_losses()
                    t = (time.time() - iter_start_time) / opt.batch_size
                    visualizer.print_current_losses(
                        epoch, epoch_iter, losses, t, t_data)

                if total_steps % opt.save_latest_freq == 0:
                    print('saving the latest model (epoch %d, total_steps %d)' %
                          (epoch, total_steps))
                    print("experiment name:", opt.name)
                    model.save_networks('latest')

                iter_data_time = time.time()

            if epoch % opt.save_epoch_freq == 0:
                print('saving the model at the end of epoch %d, iters %d' %
                      (epoch, total_steps))
                model.save_networks('latest')
                model.save_networks(epoch)
            print('End of epoch %d / %d \t Time Taken: %d sec' %
                  (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
            model.update_learning_rate()

        model.eval()
        validation_loss = 0.0
        for i, data in enumerate(val_dataset):
            model.set_input(data)
            real_in, fake_out_B, real_out_B, fake_out, real_out, val_loss_B, val_loss_C = model.validate()
            validation_loss += val_loss_C
            ABC_path = data['ABC_path']
            # print("ABC_path len", len(ABC_path))
            # last batch will be smaller than batch size
            for i in range(len(ABC_path)):
                ABC_path_i = ABC_path[i]
                file_name = ABC_path_i.split('/')[-1].split('.')[0]
                real_out_i = real_out[i].unsqueeze(0)
                fake_out_i = fake_out[i].unsqueeze(0)
                real_out_B_i = real_out_B[i].unsqueeze(0)
                fake_out_B_i = fake_out_B[i].unsqueeze(0)
                images = [real_out_i, fake_out_i, real_out_B_i, fake_out_B_i]
                names = ['real', 'fake', 'real_B', 'fake_B']
                img_path = str(epoch) + '_' + file_name
                save_images(images, names, img_path, opt=validate_opt, aspect_ratio=1.0,
                            width=validate_opt.fineSize)

        self.fitness = validation_loss.item()
        return self.fitness


class PSO:
    def __init__(self, dim, size, iter_num, best_fitness_value=float('inf'), C1=2, C2=2, W=1):
        self.C1 = C1  # 加速度系数
        self.C2 = C2  # 加速度系数
        self.W = W
        self.dim = dim  # 粒子维度，超参数的个数
        self.size = size  # 粒子个数
        self.iter_num = iter_num

        self.range_list = get_range_list()

        self.best_fitness = best_fitness_value  # 最优适应度值
        self.best_postition = []
        self.fitness_list = []

        self.particles = [Particle(i, self.range_list, self.dim)
                          for i in range(self.size)]
        self.get_best_fitness_position()

    def get_best_fitness_position(self):
        for particle in self.particles:
            if particle.fitness < self.best_fitness:
                self.best_fitness = particle.fitness
                self.best_postition = copy.deepcopy(particle.position)

    def update_velocity(self, particle):
        for i in range(self.dim):
            vel_value = self.W * particle.velocity[i] \
                + self.C1*random.random()*(particle.best_position[i] - particle.position[i]) \
                + self.C2*random.random() * \
                (self.best_postition[i]-particle.position[i])
            vel_value = round(vel_value)
            particle.velocity[i] = vel_value

    def update_position(self, iter_id, particle):
        for i in range(self.dim):
            pos_value = particle.position[i] + particle.velocity[i]
            if pos_value < 0:
                pos_value = 0
            elif pos_value >= self.range_list[i]:
                pos_value = self.range_list[i]-1
            particle.position[i] = pos_value
        fitness = particle.fit_model(iter_id)
        if fitness < particle.best_fitness:
            particle.best_position = copy.deepcopy(particle.position)
            particle.best_fitness = fitness
        if fitness < self.best_fitness:
            self.best_postition = copy.deepcopy(particle.position)
            self.best_fitness = fitness

    def iterate(self):
        print("Iterating pso")
        for i in range(1, self.iter_num):
            for particle in self.particles:
                self.update_velocity(particle)
                self.update_position(i, particle)
            self.fitness_list.append(self.best_fitness)
            print("End of iter %d / %d" % (i, self.iter_num))
            print("Current best fitness:", self.best_fitness)

        return self.fitness_list, self.best_postition
