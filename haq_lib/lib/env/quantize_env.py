# Code for "[HAQ: Hardware-Aware Automated Quantization with Mixed Precision"
# Kuan Wang*, Zhijian Liu*, Yujun Lin*, Ji Lin, Song Han
# {kuanwang, zhijian, yujunlin, jilin, songhan}@mit.edu

import time
import math
import torch
import numpy as np
import torch.nn as nn
from copy import deepcopy
import torch.optim as optim
from progress.bar import Bar
from tqdm import tqdm
from haq_lib.lib.utils.utils import AverageMeter, accuracy, prGreen, measure_model
from haq_lib.lib.utils.quantize_utils import quantize_model, kmeans_update_model

from utils.general import to_categorical
from utils.general import seg_classes, seg_label_to_cat


class QuantizeEnv:
    def __init__(self, model, pretrained_model, train_loader, val_loader,
                 compress_ratio, args, float_bit=32, is_model_pruned=False,
                 num_category=16, num_part=50, num_point=1024, logger=None):
        # default setting
        self.quantizable_layer_types = [nn.Conv2d, nn.Linear]

        # logger
        self.logger = logger

        # save options
        self.model = model
        self.model_for_measure = deepcopy(model)
        self.cur_ind = 0
        self.strategy = []  # quantization strategy

        # init data loader
        self.train_loader, self.val_loader = train_loader, val_loader
        self.num_category = num_category
        self.num_part = num_part
        self.num_point = num_point

        self.finetune_lr = args.finetune_lr
        self.optimizer = optim.SGD(
            model.parameters(), lr=args.finetune_lr, momentum=0.9, weight_decay=1e-5)
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.pretrained_model = pretrained_model
        self.compress_ratio = compress_ratio
        self.is_model_pruned = is_model_pruned
        self.finetune_gamma = args.finetune_gamma
        self.finetune_lr = args.finetune_lr
        self.finetune_flag = args.finetune_flag
        self.finetune_epoch = args.finetune_epoch

        # options from args
        self.min_bit = args.min_bit
        self.max_bit = args.max_bit
        self.float_bit = float_bit * 1.
        self.last_action = self.max_bit

        # sanity check
        assert self.compress_ratio > self.min_bit * 1. / self.float_bit, \
            'Error! You can make achieve compress_ratio smaller than min_bit!'

        # init reward
        self.best_reward = -math.inf

        # build indexs
        self._build_index()
        self._get_weight_size()
        self.n_quantizable_layer = len(self.quantizable_idx)

        self.model.load_state_dict(self.pretrained_model, strict=True)
        self.org_miou = self._validate(self.val_loader, self.model)
        # build embedding (static part), same as pruning
        self._build_state_embedding()

        # restore weight
        self.reset()
        self.logger.info('=> original miou: {:.3f}% on split dataset'.format(
            self.org_miou))
        self.logger.info('=> original #param: {:.4f}, model size: {:.4f} MB'.format(sum(self.wsize_list) * 1. / 1e6,
                                                                         sum(self.wsize_list) * self.float_bit / 8e6))

    def adjust_learning_rate(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= self.finetune_gamma

    def step(self, action):
        # Pseudo prune and get the corresponding statistics. The real pruning happens till the end of all pseudo pruning
        action = self._action_wall(action)  # percentage to preserve

        self.strategy.append(action)  # save action to strategy

        # all the actions are made
        if self._is_final_layer():
            self._final_action_wall()
            assert len(self.strategy) == len(self.quantizable_idx)
            w_size = self._cur_weight()
            w_size_ratio = self._cur_weight() / self._org_weight()

            centroid_label_dict = quantize_model(self.model, self.quantizable_idx, self.strategy,
                                                 mode='cpu', quantize_bias=False, centroids_init='k-means++',
                                                 is_pruned=self.is_model_pruned, max_iter=3)
            if self.finetune_flag:
                train_acc = self._kmeans_finetune(self.train_loader, self.model, self.quantizable_idx,
                                                  centroid_label_dict, epochs=self.finetune_epoch, verbose=False)
                acc = self._validate(self.val_loader, self.model)
            else:
                acc = self._validate(self.val_loader, self.model)

            # reward = self.reward(acc, w_size_ratio)
            reward = self.reward(acc)

            info_set = {'w_ratio': w_size_ratio,
                        'accuracy': acc, 'w_size': w_size}

            if reward > self.best_reward:
                self.best_reward = reward
                prGreen('New best policy: {}, reward: {:.3f}, acc: {:.3f}, w_ratio: {:.3f}'.format(
                    self.strategy, self.best_reward, acc, w_size_ratio))

            # actually the same as the last state
            obs = self.layer_embedding[self.cur_ind, :].copy()
            done = True
            return obs, reward, done, info_set

        w_size = self._cur_weight()
        info_set = {'w_size': w_size}
        reward = 0
        done = False
        self.cur_ind += 1  # the index of next layer
        self.layer_embedding[self.cur_ind][-1] = action
        # build next state (in-place modify)
        obs = self.layer_embedding[self.cur_ind, :].copy()
        return obs, reward, done, info_set

    # for quantization
    def reward(self, acc, w_size_ratio=None):
        if w_size_ratio is not None:
            return (acc - self.org_miou + 1. / w_size_ratio) * 0.1
        return (acc - self.org_miou) * 0.1

    def reset(self):
        # restore env by loading the pretrained model
        self.model.load_state_dict(self.pretrained_model, strict=True)
        self.optimizer = optim.SGD(self.model.parameters(
        ), lr=self.finetune_lr, momentum=0.9, weight_decay=4e-5)
        self.cur_ind = 0
        self.strategy = []  # quantization strategy
        obs = self.layer_embedding[0].copy()
        return obs

    def _is_final_layer(self):
        return self.cur_ind == len(self.quantizable_idx) - 1

    def _final_action_wall(self):
        target = self.compress_ratio * self._org_weight()
        min_weight = 0
        for i, n_bit in enumerate(self.strategy):
            min_weight += self.wsize_list[i] * self.min_bit
        while min_weight < self._cur_weight() and target < self._cur_weight():
            for i, n_bit in enumerate(reversed(self.strategy)):
                if n_bit > self.min_bit:
                    self.strategy[-(i+1)] -= 1
                if target >= self._cur_weight():
                    break
        self.logger.info('=> Final action list: {}'.format(self.strategy))

    # def _action_wall(self, action):
    #     assert len(self.strategy) == self.cur_ind
    #     # limit the action to certain range
    #     action = float(action)
    #     min_bit, max_bit = self.bound_list[self.cur_ind]
    #     lbound, rbound = min_bit - 0.5, max_bit + \
    #         0.5  # same stride length for each bit
    #     action = (rbound - lbound) * action + lbound
    #     action = int(np.round(action, 0))
    #     self.last_action = action
    #     return action  # not constrained here

    def _action_wall(self, action):
        assert len(self.strategy) == self.cur_ind
        # limit the action to certain range
        action = float(action)
        min_bit, max_bit = self.bound_list[self.cur_ind]
        out_action = max_bit if action > 0.5 else min_bit
        self.last_action = out_action
        return out_action  # not constrained here

    def _cur_weight(self):
        cur_weight = 0.
        # quantized
        for i, n_bit in enumerate(self.strategy):
            cur_weight += n_bit * self.wsize_list[i]
        return cur_weight

    def _cur_reduced(self):
        # return the reduced weight
        reduced = self.org_bitops - self._cur_bitops()
        return reduced

    def _org_weight(self):
        org_weight = 0.
        org_weight += sum(self.wsize_list) * self.float_bit
        return org_weight

    def _build_index(self):
        self.quantizable_idx = []
        self.layer_type_list = []
        self.bound_list = []
        # for i, layers in enumerate(self.model.modules()):
        #     if hasattr(self.model, 'modules'):
        #         layers = layers.modules()
        #     else:
        #         layers = [layers]
        #     for m in layers:
        #         if type(m) in self.quantizable_layer_types:
        #             self.quantizable_idx.append(i)
        #             self.layer_type_list.append(type(m))
        #             self.bound_list.append((self.min_bit, self.max_bit))
        for i, m in enumerate(self.model.modules()):
            if type(m) in self.quantizable_layer_types:
                self.quantizable_idx.append(i)
                self.layer_type_list.append(type(m))
                self.bound_list.append((self.min_bit, self.max_bit))
        self.logger.info('=> Final bound list: {}'.format(self.bound_list))

    def _get_weight_size(self):
        # get the param size for each layers to prune, size expressed in number of params
        self.wsize_list = []
        for i, m in enumerate(self.model.modules()):
            if i in self.quantizable_idx:
                if not self.is_model_pruned:
                    self.wsize_list.append(m.weight.data.numel())
                else:  # the model is pruned, only consider non-zeros items
                    self.wsize_list.append(torch.sum(m.weight.data.ne(0)))
        self.wsize_dict = {i: s for i, s in zip(
            self.quantizable_idx, self.wsize_list)}

    def _get_latency_list(self):
        # use simulator to get the latency
        raise NotImplementedError

    def _get_energy_list(self):
        # use simulator to get the energy
        raise NotImplementedError

    def _build_state_embedding(self):
        # measure model for cifar 32x32 input
        measure_model(self.model_for_measure, 1024, 22)
        # build the static part of the state embedding
        layer_embedding = []
        module_list = list(self.model_for_measure.modules())
        for i, ind in enumerate(self.quantizable_idx):
            m = module_list[ind]
            this_state = []
            if type(m) == nn.Conv2d:
                # layer type, 1 for conv_dw
                this_state.append([int(m.in_channels == m.groups)])
                this_state.append([m.in_channels])  # in channels
                this_state.append([m.out_channels])  # out channels
                this_state.append([m.stride[0]])  # stride
                this_state.append([m.kernel_size[0]])  # kernel size
                this_state.append([np.prod(m.weight.size())])  # weight size
                this_state.append([m.in_w*m.in_h])  # input feature_map_size
            elif type(m) == nn.Linear:
                this_state.append([0.])  # layer type, 0 for fc
                this_state.append([m.in_features])  # in channels
                this_state.append([m.out_features])  # out channels
                this_state.append([0.])  # stride
                this_state.append([1.])  # kernel size
                this_state.append([np.prod(m.weight.size())])  # weight size
                this_state.append([m.in_w*m.in_h])  # input feature_map_size

            this_state.append([i])  # index
            this_state.append([4.])  # bits
            layer_embedding.append(np.hstack(this_state))

        # normalize the state
        layer_embedding = np.array(layer_embedding, 'float')
        self.logger.info('=> shape of embedding (n_layer * n_dim): {}'.format(layer_embedding.shape))
        assert len(layer_embedding.shape) == 2, layer_embedding.shape
        for i in range(layer_embedding.shape[1]):
            fmin = min(layer_embedding[:, i])
            fmax = max(layer_embedding[:, i])
            if fmax - fmin > 0:
                layer_embedding[:, i] = (
                    layer_embedding[:, i] - fmin) / (fmax - fmin)

        self.layer_embedding = layer_embedding

    def _kmeans_finetune(self, train_loader, model, idx, centroid_label_dict, epochs=1, verbose=True):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        model.train()

        end = time.time()
        bar = Bar('train:', max=len(train_loader))
        best_acc = 0.
        criterion = torch.nn.CrossEntropyLoss()

        for _ in range(epochs):
            seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
            mean_correct = []

            for cat in seg_classes.keys():
                for label in seg_classes[cat]:
                    seg_label_to_cat[label] = cat

            for i, (points, label, target) in enumerate(train_loader):
                cur_batch_size, NUM_POINT, _ = points.size()
                points, label, target = points.float().cuda(
                ), label.long().cuda(), target.long().cuda()
                seg_pred = model(torch.cat([points, to_categorical(
                    label, self.num_category).repeat(1, points.shape[1], 1)], -1))
                seg_pred = seg_pred.contiguous().view(-1, self.num_part)
                target = target.view(-1, 1)[:, 0]
                pred_choice = seg_pred.data.max(1)[1]

                correct = pred_choice.eq(target.data).cpu().sum()
                mean_correct.append(
                    correct.item() / (train_loader.batch_size * self.num_point))
                loss = criterion(seg_pred, target)
                loss.backward()
                self.optimizer.step()

                kmeans_update_model(model, self.quantizable_idx,
                                    centroid_label_dict, free_high_bit=True)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # plot progress
                if i % 1 == 0:
                    bar.suffix = \
                        '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | ' \
                        .format(
                            batch=i + 1,
                            size=len(train_loader),
                            data=data_time.avg,
                            bt=batch_time.avg,
                            total=bar.elapsed_td,
                            eta=bar.eta_td
                        )
                    bar.next()

            bar.finish()
            train_instance_acc = np.mean(mean_correct)

            if train_instance_acc > best_acc:
                best_acc = train_instance_acc
            self.adjust_learning_rate()
        
        self.logger.info('* Train class_avg_accuracy: %.3f' % best_acc)

        return best_acc

    def _validate(self, val_loader, model, verbose=False):
        batch_time = AverageMeter()
        data_time = AverageMeter()

        with torch.no_grad():
            # switch to evaluate mode
            model.eval()

            end = time.time()
            bar = Bar('valid:', max=len(val_loader))

            total_correct = 0
            total_seen = 0
            total_seen_class = [0 for _ in range(self.num_part)]
            total_correct_class = [0 for _ in range(self.num_part)]
            shape_ious = {cat: [] for cat in seg_classes.keys()}
            seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
            test_metrics = {}

            for cat in seg_classes.keys():
                for label in seg_classes[cat]:
                    seg_label_to_cat[label] = cat

            for i, (points, label, target) in enumerate(val_loader):
                cur_batch_size, NUM_POINT, _ = points.size()
                points, label, target = points.float().cuda(
                ), label.long().cuda(), target.long().cuda()
                seg_pred = model(torch.cat([points, to_categorical(
                    label, self.num_category).repeat(1, points.shape[1], 1)], -1))

                cur_pred_val = seg_pred.cpu().data.numpy()
                cur_pred_val_logits = cur_pred_val
                cur_pred_val = np.zeros(
                    (cur_batch_size, NUM_POINT)).astype(np.int32)
                target = target.cpu().data.numpy()

                for j in range(cur_batch_size):
                    cat = seg_label_to_cat[target[j, 0]]
                    logits = cur_pred_val_logits[j, :, :]
                    cur_pred_val[j, :] = np.argmax(
                        logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]

                correct = np.sum(cur_pred_val == target)
                total_correct += correct
                total_seen += (cur_batch_size * NUM_POINT)

                for l in range(self.num_part):
                    total_seen_class[l] += np.sum(target == l)
                    total_correct_class[l] += (
                        np.sum((cur_pred_val == l) & (target == l)))

                for i in range(cur_batch_size):
                    segp = cur_pred_val[i, :]
                    segl = target[i, :]
                    cat = seg_label_to_cat[segl[0]]
                    part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
                    for l in seg_classes[cat]:
                        if (np.sum(segl == l) == 0) and (
                                np.sum(segp == l) == 0):  # part is not present, no prediction as well
                            part_ious[l - seg_classes[cat][0]] = 1.0
                        else:
                            part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                                np.sum((segl == l) | (segp == l)))
                    shape_ious[cat].append(np.mean(part_ious))
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                # plot progress
                if i % 1 == 0:
                    bar.suffix = \
                        '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | ' \
                        .format(
                            batch=i + 1,
                            size=len(val_loader),
                            data=data_time.avg,
                            bt=batch_time.avg,
                            total=bar.elapsed_td,
                            eta=bar.eta_td
                        )
                    bar.next()

            all_shape_ious = []
            for cat in shape_ious.keys():
                for iou in shape_ious[cat]:
                    all_shape_ious.append(iou)
                shape_ious[cat] = np.mean(shape_ious[cat])
            mean_shape_ious = np.mean(list(shape_ious.values()))
            test_metrics['accuracy'] = total_correct / float(total_seen)
            test_metrics['class_avg_accuracy'] = np.mean(
                np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))
            for cat in sorted(shape_ious.keys()):
                self.logger.info('* Eval mIoU of {} {}'.format(
                    cat + ' ' * (14 - len(cat)), shape_ious[cat]))
            test_metrics['class_avg_iou'] = mean_shape_ious
            test_metrics['inctance_avg_iou'] = np.mean(all_shape_ious)
            bar.finish()

        self.logger.info('* Test accuracy: %.3f  class_avg_accuracy: %.3f  '
              'class_avg_iou: %.3f  inctance_avg_iou: %.3f' %
              (test_metrics['accuracy'], test_metrics['class_avg_accuracy'],
               test_metrics['class_avg_iou'], test_metrics['inctance_avg_iou']))

        return test_metrics['accuracy']
