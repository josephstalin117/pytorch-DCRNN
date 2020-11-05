import numpy as np
import torch
import math
import time
import json
import yaml
import os
from torch.utils.tensorboard import SummaryWriter
import datetime

import argparse
import collections
import torch
import dcrnn_metrics as module_metric
import dcrnn_model as dcrnn_model
#from parse_config import ConfigParser
import math
import dcrnn_utils as utils

from abc import abstractmethod
from numpy import inf


class DCRNNConfig():

    def __init__(self, args, config, options='', timestamp=True):
        self.base_dir = config['base_dir']

        # model
        self.n_in = config['model']['n_in']
        self.n_out = config['model']['n_out']
        self.num_rnn_layers = config['model']['num_rnn_layers']
        self.rnn_units = config['model']['rnn_units']
        self.input_dim = config['model']['input_dim']
        self.output_dim = config['model']['output_dim']
        # site nums
        self.num_nodes = config['model']['num_nodes']
        self.enc_input_dim = config['model']['enc_input_dim']
        self.dec_input_dim = config['model']['dec_input_dim']
        self.max_diffusion_step = config['model']['max_diffusion_step']
        self.cl_decay_steps = config["model"]["cl_decay_steps"]

        # train
        self.n_gpu = config['train']['n_gpu']
        self.device, self.device_ids = prepare_device(self.n_gpu)
        self.epochs = config['train']['epochs']
        self.save_period = config['train']['save_period']
        self.monitor = config['train']['monitor']
        self.save_dir = config['train']['save_dir']
        self.log_dir = config['log']['log_dir']
        self.base_lr = config['train']['base_lr']
        self.epsilon = config['train']['epsilon']
        self.max_grad_norm = config["train"]["max_grad_norm"]
        self.lr_milestones = config['train']['lr_milestones']
        self.lr_decay_ratio = config['train']['lr_decay_ratio']
        self.results_dir = config['train']['results_dir']

        # data
        self.dataset_dir = config['data']['dataset_dir']
        self.graph_pkl_filename = config['data']['graph_pkl_filename']
        self.batch_size = config['data']['batch_size']
        self.data_name = config['data']['data_name']

        self.exp_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.graph_name = "dcrnn_%s_nin%d_nout%d_batch%d_epoch%d_time%s" % \
                 (self.data_name, self.n_in, self.n_out, self.batch_size, self.epochs, self.exp_time)


def prepare_device(n_gpu_use):
    """
    setup GPU device if available, move model into configured device
    """
    gpu_num = torch.cuda.device_count()
    if n_gpu_use > 0 and gpu_num == 0:
        self.logger.warning("Warning: There\'s no GPU available on this machine,"
                            "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > gpu_num:
        self.logger.warning("Warning: The number of GPU\'s configured to use is {}, but only {} are available "
                            "on this machine.".format(n_gpu_use, n_gpu))
        n_gpu_use = gpu_num
    device = torch.device('cuda:'+ str(n_gpu_use) if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids

    


class DCRNNTrainer():
    """
    DCRNN trainer class
    """
    def __init__(self, model, loss, optimizer, config, data_loader, logger, 
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None, val_len_epoch=None):

        self.config = config
        #self.logger = config.get_logger('trainer', config['trainer']['verbosity'])
        self.logger = logger
        self.n_in = config.n_in
        self.n_out = config.n_out

        # setup GPU device if available, move model into configured device
        #self.device, device_ids = self._prepare_device(config['train']['n_gpu'])
        #self.device, device_ids = self._prepare_device(config.n_gpu)
        self.device = config.device
        device_ids = config.device_ids
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.loss = loss
        # a list of metric functions defined in dcrnn_metric.py
        self.metrics = ["masked_mae_np",  "masked_rmse_np", "masked_mape_np"]
        self.optimizer = optimizer

        #self.epochs = config['train']['epochs']
        self.epochs = config.epochs
        #self.save_period = cfg_trainer['save_period']  # should be 1
        # should be 1
        #self.save_period = config['train']['save_period']
        self.save_period = config.save_period

        #self.monitor = cfg_trainer.get('monitor', 'off')
        self.monitor = config.monitor

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            # TODO: early_stop
            self.early_stop = cfg_trainer.get('early_stop', inf)

        self.start_epoch = 1
        self.base_dir = config.base_dir
        self.checkpoint_dir = config.save_dir

        # setup visualization writer instance                
        # self.writer = TensorboardWriter(config['log']['log_dir'], self.logger, config['train']['tensorboard'])
        self.writer = SummaryWriter(os.path.join(config.base_dir, config.log_dir))

        #if config['train']['resume'] is not None:
        #    self._resume_checkpoint(config['train']['resume'])

        self.config = config
        self.data_loader = data_loader
        self.len_epoch = len_epoch
        self.val_len_epoch = val_len_epoch
        self.cl_decay_steps = config.cl_decay_steps

        self.max_grad_norm = config.max_grad_norm
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(20)
        # self.log_step = int(np.sqrt(data_loader.batch_size))  # sqrt(128)  sqrt(64)


    def _eval_metrics(self, output, target):
        #acc_metrics = np.zeros(len(self.metrics))
        acc_list=[]
        # mae
        acc_list.append(module_metric.masked_mae_np(output, target))
        # rmse
        acc_list.append(module_metric.masked_rmse_np(output, target))
        # mape
        acc_list.append(module_metric.masked_mape_np(output, target))
        acc_metrics=np.array(acc_list)
        for i, metric in enumerate(self.metrics):
            #acc_metrics[i] += metric(output, target)
            self.writer.add_scalar('{}'.format(metric), acc_metrics[i])
        return acc_metrics


    def _eval_metrics2(self, output, target):
        # mae
        acc_mae = module_metric.masked_mae_np(output, target)
        # rmse
        acc_rmse = module_metric.masked_rmse_np(output, target)
        # mape
        acc_mape = module_metric.masked_mape_np(output, target)
        self.writer.add_scalar('mae', acc_mae)
        self.writer.add_scalar('rmse', acc_rmse)
        self.writer.add_scalar('mape', acc_mape)
        return acc_mae, acc_rmse, acc_mape


    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        #filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        if not os.path.exists(os.path.join(self.base_dir, self.checkpoint_dir)):
            os.mkdir(os.path.join(self.base_dir, self.checkpoint_dir))

        filename = os.path.join(self.base_dir, self.checkpoint_dir, 'checkpoint-epoch{}.pth'.format(epoch))

        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            #best_path = str(self.checkpoint_dir / 'model_best.pth')
            best_path = os.path.join(self.base_dir, self.checkpoint_dir, 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")


    def _prepare_device(self, n_gpu_use):
        """
        setup GPU device if available, move model into configured device
        """
        gpu_num = torch.cuda.device_count()
        if n_gpu_use > 0 and gpu_num == 0:
            self.logger.warning("Warning: There\'s no GPU available on this machine,"
                                "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > gpu_num:
            self.logger.warning("Warning: The number of GPU\'s configured to use is {}, but only {} are available "
                                "on this machine.".format(n_gpu_use, n_gpu))
            n_gpu_use = gpu_num
        device = torch.device('cuda:'+ str(n_gpu_use) if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids


    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))


    def train(self):
        """
        Full training logic
        """
        training_time = 0
        logs = []
        for epoch in range(self.start_epoch, self.epochs + 1):
            result, train_epoch_time = self._train_epoch(epoch)
            training_time += train_epoch_time
            # save logged information into log dict
            log = {'epoch': epoch}
            for key, value in result.items():
                if key == 'metrics':
                    #log.update({mtr.__name__: value[i] for i, mtr in enumerate(self.metrics)})
                    # TODO: need edit
                    log.update({mtr: value[i] for i, mtr in enumerate(self.metrics)})
                elif key == 'val_metrics':
                    #log.update({'val_' + mtr.__name__: value[i] for i, mtr in enumerate(self.metrics)})
                    log.update({'val_' + mtr: value[i] for i, mtr in enumerate(self.metrics)})
                else:
                    log[key] = value

            # There is a chance that the training loss will explode, the temporary workaround
            # is to restart from the last saved model before the explosion, or to decrease
            # the learning rate earlier in the learning rate schedule.
            if log['loss'] > 1e5:
                self.logger.warning('Gradient explosion detected. Ending...')
                break
            # save log
            logs.append(log)

            # print logged information to the screen
            for key, value in log.items():
                self.logger.info('{:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False
                    not_improved_count = 0

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)
        average_trianing_time = training_time / self.epochs
        self.logger.info("Average training time: {:.4f}s".format(average_trianing_time))
        return logs


    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.model.train()
        start_time = time.time()
        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))
        for batch_idx, (data, target) in enumerate(self.data_loader.get_iterator()):
            data = torch.FloatTensor(data)
            target = torch.FloatTensor(target)
            # (..., 1)  supposed to be numpy array
            label = target[..., :self.model.output_dim]
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()

            # compute sampling ratio, which gradually decay to 0 during training
            global_step = (epoch - 1) * self.len_epoch + batch_idx
            teacher_forcing_ratio = self._compute_sampling_threshold(global_step, self.cl_decay_steps)

            output = self.model(data, target, teacher_forcing_ratio)
            #self.logger.info(f"output.shape: {output.shape}")
            # back to (50, 12, 207, 1)
            # back to (batch_size, n_out, num_nodes, output_dim)
            output = torch.transpose(output.view(self.n_out, self.model.batch_size, self.model.num_nodes,
                                                 self.model.output_dim), 0, 1)

            # loss is self-defined, need cpu input
            loss = self.loss(output.cpu(), label)              
            loss.backward()
            # add max grad clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            training_time = time.time() - start_time

            #self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.writer.add_scalar('loss', loss.item())
            total_loss += loss.item()
            total_metrics += self._eval_metrics(output.detach().numpy(), label.numpy())

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))

            if batch_idx == self.len_epoch:
                break

        log = {
            'loss': total_loss / self.len_epoch,
            'metrics': (total_metrics / self.len_epoch).tolist(),
        }

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(val_log)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        log.update({'Time': "{:.4f}s".format(training_time)})
        return log, training_time


    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_val_loss = 0
        total_val_metrics = np.zeros(len(self.metrics))
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader.get_iterator()):
                data = torch.FloatTensor(data)
                target = torch.FloatTensor(target)
                # (..., 1)  supposed to be numpy array
                label = target[..., :self.model.output_dim]
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data, target, 0)
                # back to (50, 12, 207, 1)
                #output = torch.transpose(output.view(12, self.model.batch_size, self.model.num_nodes,
                #                                     self.model.output_dim), 0, 1)

                # back to (batch_size, n_out, num_nodes, output_dim)
                output = torch.transpose(output.view(self.n_out, self.model.batch_size, self.model.num_nodes,
                                                     self.model.output_dim), 0, 1)


                loss = self.loss(output.cpu(), label)

                #self.writer.set_step((epoch - 1) * self.val_len_epoch + batch_idx, 'valid')
                self.writer.add_scalar('loss', loss.item())
                total_val_loss += loss.item()
                total_val_metrics += self._eval_metrics(output.detach().numpy(), label.numpy())
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')

        return {
            'val_loss': total_val_loss / self.val_len_epoch,
            'val_metrics': (total_val_metrics / self.val_len_epoch).tolist()
        }

    
    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)


    @staticmethod
    def _compute_sampling_threshold(global_step, k):
        """
        Computes the sampling probability for scheduled sampling using inverse sigmoid.
        :param global_step:
        :param k:
        :return:
        """
        return k / (k + math.exp(global_step / k))


class DCRNNTester():
    """
    DCRNN tester class
    """
    def __init__(self, model, loss, inverse, config, data_loader, test_data_loader, logger, test_len_epoch=None):

        self.config = config
        self.logger = logger
        self.loss = loss
        self.inverse = inverse
        self.n_in = config.n_in
        self.n_out = config.n_out
        
        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(config.n_gpu)
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        # a list of metric functions defined in dcrnn_metric.py
        self.metrics = ["masked_mae_np",  "masked_rmse_np", "masked_mape_np"]

        self.base_dir = config.base_dir

        self.config = config
        self.data_loader = data_loader
        self.test_len_epoch = test_len_epoch

        self.test_data_loader = test_data_loader
        self.log_step = int(20)


    def _eval_metrics(self, output, target):
        #acc_metrics = np.zeros(len(self.metrics))
        acc_list=[]
        # mae
        acc_list.append(module_metric.masked_mae_np(output, target))
        # rmse
        acc_list.append(module_metric.masked_rmse_np(output, target))
        # mape
        acc_list.append(module_metric.masked_mape_np(output, target))
        acc_metrics=np.array(acc_list)
        return acc_metrics
    

    def _prepare_device(self, n_gpu_use):
        """
        setup GPU device if available, move model into configured device
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning("Warning: There\'s no GPU available on this machine,"
                                "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning("Warning: The number of GPU\'s configured to use is {}, but only {} are available "
                                "on this machine.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids


    def predict(self):
        """
        Test

        :return: A log that contains information about test

        Note:
            The test metrics in log must have the key 'test_metrics'.
        """
        self.model.eval()
        total_test_loss = 0
        outputs = []
        targets = []
        total_test_metrics = np.zeros(len(self.metrics))
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.test_data_loader.get_iterator()):
                data = torch.FloatTensor(data)
                target = torch.FloatTensor(target)
                # (..., 1)  supposed to be numpy array
                label = target[..., :self.model.output_dim]
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data, target, 0)
                # back to (50, 12, 207, 1)
                # back to (batch_size, n_out, num_nodes, out_dim)
                #output = torch.transpose(output.view(12, self.model.batch_size, self.model.num_nodes,
                #                                     self.model.output_dim), 0, 1)
                output = torch.transpose(output.view(self.n_out, self.model.batch_size, self.model.num_nodes,
                                                     self.model.output_dim), 0, 1)

                loss = self.loss(output.cpu(), label)


                total_test_loss += loss.item()
                total_test_metrics += self._eval_metrics(output.detach().numpy(), label.numpy())
                acc_metrics = self._eval_metrics(output.detach().numpy(), label.numpy())
                # inverse scaler
                output, label = self.inverse(output.cpu(), label)

                for i in range(len(output)-1):
                    outputs.append(output[i].numpy().tolist())
                    targets.append(label[i].numpy().tolist())

        test_loss = total_test_loss / self.test_len_epoch
        test_metrics = (total_test_metrics / self.test_len_epoch).tolist()
        test_mae = test_metrics[0]
        test_rmse = test_metrics[1]
        test_mape = test_metrics[2]
        self.logger.info(f"test_loss: {test_loss}")
        self.logger.info(f"test_mae: {test_mae}")
        self.logger.info(f"test_rmse: {test_rmse}")
        self.logger.info(f"test_mape: {test_mape}")

        return test_loss, test_metrics, outputs, targets
    


def main(config):
    # set logger
    if not os.path.exists(os.path.join(config.base_dir, config.log_dir)):
        os.mkdir(os.path.join(config.base_dir, config.log_dir))
    logger = utils.get_logger(os.path.join(config.base_dir, config.log_dir), "test", log_filename=config.graph_name + ".log")

    # load dataset
    #graph_pkl_filename = '../dat/adj_mx.pkl'
    #graph_pkl_filename = os.path.join(config['base_dir'], config['data']['graph_pkl_filename'])
    graph_pkl_filename = os.path.join(config.base_dir, config.dataset_dir, config.graph_pkl_filename)
    _, _, adj_mat = utils.load_graph_data(graph_pkl_filename)
    data = utils.load_dataset(dataset_dir= os.path.join(config.base_dir, config.dataset_dir),
                              batch_size=config.batch_size,
                              test_batch_size=config.batch_size)

    logger.info(f"data:")
    logger.info(f"x_train: {data['x_train'].shape}, y_train: {data['y_train'].shape}")
    logger.info(f"x_val: {data['x_val'].shape}, y_val: {data['y_val'].shape}")
    logger.info(f"x_test: {data['x_test'].shape}, y_test: {data['y_test'].shape}")
    
    train_data_loader = data['train_loader']
    val_data_loader = data['val_loader']
    test_data_loader = data['test_loader']

    num_train_sample = data['x_train'].shape[0]
    num_val_sample = data['x_val'].shape[0]
    num_test_sample = data['x_test'].shape[0]

    # get number of iterations per epoch for progress bar
    num_train_iteration_per_epoch = math.ceil(num_train_sample / config.batch_size)
    num_val_iteration_per_epoch = math.ceil(num_val_sample / config.batch_size)
    num_test_iteration_per_epoch = math.ceil(num_test_sample / config.batch_size)

    # setup data_loader instances
    # data_loader = config.initialize('data_loader', module_data)
    # valid_data_loader = data_loader.split_validation()

    # build model architecture, then print to console
    #adj_arg = {"adj_mat": adj_mat}
    logger.info(f"model architecture:")
    logger.info(f"num_rnn_layers: {config.num_rnn_layers}, run_units: {config.rnn_units}, max_diffusion_step: {config.max_diffusion_step}")
    logger.info(f"n_in: {config.n_in}, n_out: {config.n_out}, epochs: {config.epochs}")
    logger.info(f"gpu: {config.n_gpu}")
    logger.info(f"input_dim: {config.input_dim}, output_dim: {config.output_dim}, num_nodes: {config.num_nodes}, batch_size: {config.batch_size}")
    logger.info(f"enc_input_dim: {config.enc_input_dim}, dec_input_dim: {config.dec_input_dim}")

    model = dcrnn_model.DCRNNModel(adj_mat, config.batch_size, config.enc_input_dim, config.dec_input_dim, \
            config.max_diffusion_step, config.num_nodes, config.num_rnn_layers, config.rnn_units, \
            config.output_dim, config.device)

    # model = getattr(module_arch, config['arch']['type'])(config['arch']['args'], adj_arg)

    # get function handles of loss and metrics
    loss = module_metric.masked_mae_loss(data['scaler'], 0.0)
    #loss = config.initialize('loss', module_metric, **{"scaler": data['scaler']})
    # metrics = [getattr(module_metric, met) for met in config['metrics']]

    # get inverse preds & labels
    inverse = module_metric.inverse_scaler(data['scaler'], 0.0)
    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    #optimizer = config.initialize('optimizer', torch.optim, trainable_params)
    optimizer = torch.optim.Adam(params=trainable_params, lr=config.base_lr, weight_decay=0.0, eps=config.epsilon, amsgrad=True)

    #lr_scheduler = config.initialize('lr_scheduler', torch.optim.lr_scheduler, optimizer)
    #lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config['train']['lr_milestones'], gamma=config['train']['lr_decay_ratio'])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.lr_milestones, gamma=config.lr_decay_ratio)

    trainer = DCRNNTrainer(model, loss, optimizer,
                           config=config,
                           data_loader=train_data_loader,
                           logger=logger,
                           valid_data_loader=val_data_loader,
                           lr_scheduler=lr_scheduler,
                           len_epoch=num_train_iteration_per_epoch,
                           val_len_epoch=num_val_iteration_per_epoch)

    train_logs = trainer.train()
    epoch_loss = [i['loss'] for i in train_logs]
    val_loss = [i['val_loss'] for i in train_logs]

    tester = DCRNNTester(model, loss, inverse, config, data_loader=test_data_loader, logger=logger, 
            test_data_loader=test_data_loader, test_len_epoch=num_test_iteration_per_epoch)

    test_loss, test_metrics, test_outputs, test_targets = tester.predict()
    #test_outputs = test_outputs.numpy()
    #test_targets = test_targets.numpy()

    test_mae = test_metrics[0]
    test_rmse = test_metrics[1]
    test_mape = test_metrics[2]

    # TODO: need fixed
    # result
    #results = {"test": test_targets, "prediction": test_outputs, "true": test_outputs, "train_loss": epoch_loss,
    #           "val_loss": val_loss,"rmse": test_rmse, "steps_rmse": steps_rmse, "mae": test_mae,
    #           "mape": test_mape, "in_feats": config.in_feats, "out_feats": config.out_feats,
    #           "encode_hidden_size": config.encode_hidden_size,"decode_hidden_size": config.decode_hidden_size,
    #           "full_size": config.full_size, "frame": config.frame, "columns": config.columns}
    
    results = {"test": test_targets, "prediction": test_outputs, "train_loss": epoch_loss,
               "val_loss": val_loss,"rmse": test_rmse, "mae": test_mae,"mape": test_mape, 
               "input_dim": config.input_dim, "output_dim": config.output_dim, "n_in": config.n_in, "n_out": config.n_out,
               "num_rnn_layers:": config.num_rnn_layers, "run_units": config.rnn_units, "max_diffusion_step": config.max_diffusion_step,
               "enc_input_dim": config.enc_input_dim, "dec_input_dim": config.dec_input_dim,
               "num_nodes": config.num_nodes, "batch_size": config.batch_size}


    if not os.path.exists(os.path.join(config.base_dir, config.results_dir)):
        os.mkdir(os.path.join(config.base_dir, config.results_dir))

    with open(os.path.join(config.base_dir, config.results_dir) + "/{}.json".format(config.graph_name), 'w') as fout:
        fout.write(json.dumps(results))


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch DCRNN')
    #args.add_argument('-c', '--config', default=None, type=str,
    #                  help='config file path (default: None)')
    #args.add_argument('-r', '--resume', default=None, type=str,
    #                  help='path to latest checkpoint (default: None)')
    #args.add_argument('-d', '--device', default=None, type=str,
    #                  help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    #CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    #options = [
    #    CustomArgs(['--lr', '--learning_rate'], type=float, target=('optimizer', 'args', 'lr')),
    #    CustomArgs(['--bs', '--batch_size'], type=int, target=('data_loader', 'args', 'batch_size'))
    #]
    # load config
    with open('../conf_model.yaml') as f:
        config_file = yaml.load(f, Loader=yaml.FullLoader)

    config = DCRNNConfig(args, config_file)
    main(config)

