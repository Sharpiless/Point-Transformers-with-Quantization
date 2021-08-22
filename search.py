import math
import logging
import torch.backends.cudnn as cudnn
from copy import deepcopy
from haq_lib.lib.rl.ddpg import DDPG
from haq_lib.lib.env.linear_quantize_env import LinearQuantizeEnv
from haq_lib.lib.env.quantize_env import QuantizeEnv
import hydra
from utils.dataset import PartNormalDataset
import numpy as np
import shutil
import importlib
import torch
import os
import yaml
import warnings

warnings.filterwarnings('ignore')

# training models
# rl search


def create_attr_dict(yaml_config):
    from ast import literal_eval
    for key, value in yaml_config.items():
        if type(value) is dict:
            yaml_config[key] = value = AttrDict(value)
        if isinstance(value, str):
            try:
                value = literal_eval(value)
            except BaseException:
                pass
        if isinstance(value, AttrDict):
            create_attr_dict(yaml_config[key])
        else:
            yaml_config[key] = value


class AttrDict(dict):
    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        if key in self.__dict__:
            self.__dict__[key] = value
        else:
            self[key] = value


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(), ]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y


def init_agent(model, pretrained, train_loader,
               val_loader, num_category,
               num_class, num_point, logger):
    with open('config/agent.yaml', 'r') as f:
        args = AttrDict(yaml.safe_load(f.read()))
    create_attr_dict(args)
    base_folder_name = 'output'
    if args.suffix is not None:
        base_folder_name = base_folder_name + '_' + args.suffix
    args.output = os.path.join(args.output, base_folder_name)
    print('==> Output path: {}...'.format(args.output))

    assert torch.cuda.is_available(), 'CUDA is needed for CNN'

    if args.seed > 0:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    print('    Total params: %.2fM' % (sum(p.numel()
                                           for p in model.parameters())/1000000.0))
    cudnn.benchmark = True

    if args.linear_quantization:
        env = LinearQuantizeEnv(model, pretrained,  train_loader, val_loader,
                                compress_ratio=args.preserve_ratio, args=args,
                                float_bit=args.float_bit, is_model_pruned=args.is_pruned,
                                num_category=num_category, num_part=num_class,
                                num_point=num_point, logger=logger)
    else:
        env = QuantizeEnv(model, pretrained,  train_loader, val_loader,
                          compress_ratio=args.preserve_ratio, args=args,
                          float_bit=args.float_bit, is_model_pruned=args.is_pruned,
                          num_category=num_category, num_part=num_class,
                          num_point=num_point, logger=logger)

    nb_states = env.layer_embedding.shape[1]
    nb_actions = 1  # actions for weight and activation quantization
    args.rmsize = args.rmsize * len(env.quantizable_idx)  # for each layer
    print('** Actual replay buffer size: {}'.format(args.rmsize))
    agent = DDPG(nb_states, nb_actions, args)
    return agent, env


def main():
    with open('config/partseg.yaml', 'r') as f:
        args = AttrDict(yaml.safe_load(f.read()))
    create_attr_dict(args)
    print(args.model)
    # HYPER PARAMETER
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    work_dir = args.work_dir
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    log_file =  os.path.join(work_dir, 'search.log')
    logger = logging.getLogger('search')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_file, 'w')
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    console = logging.StreamHandler()
    logger.addHandler(console)

    root = hydra.utils.to_absolute_path(
        'data/shapenetcore_partanno_segmentation_benchmark_v0_normal/')

    TRAIN_DATASET = PartNormalDataset(
        root=root, npoints=args.num_point, split='trainval', normal_channel=args.normal)
    train_loader = torch.utils.data.DataLoader(
        TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    TEST_DATASET = PartNormalDataset(
        root=root, npoints=args.num_point, split='test', normal_channel=args.normal)
    val_loader = torch.utils.data.DataLoader(
        TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # MODEL LOADING
    args.input_dim = (6 if args.normal else 3) + 16
    args.num_class = 50
    shutil.copy(hydra.utils.to_absolute_path('models/model.py'), '.')

    model = getattr(importlib.import_module('models.model'),
                    'PointTransformerSeg')(args).cuda()

    pretrained = torch.load("best_model.pth")
    if 'model_state_dict' in pretrained:
        pretrained = pretrained['model_state_dict']
    # pretrained = model.state_dict()
    # for m in model.modules():
    #     print(type(m))
    #     pass
    agent, env = init_agent(model, pretrained, train_loader,
                            val_loader, args.num_category,
                            args.num_class, args.num_point, logger)

    best_reward = -math.inf
    best_policy = []
    agent.is_training = True
    step = episode = episode_steps = 0
    episode_reward = 0.
    observation = None
    T = []  # trajectory
    if not os.path.exists(args.output):
        os.mkdir(args.output)
    while episode < args.train_episode:  # counting based on episode
        # reset if it is the start of episode
        if observation is None:
            observation = deepcopy(env.reset())
            agent.reset(observation)

        # agent pick action ...
        if episode <= args.warmup:
            action = agent.random_action()
        else:
            action = agent.select_action(observation, episode=episode)

        # env response with next_observation, reward, terminate_info
        observation2, reward, done, info = env.step(action)
        observation2 = deepcopy(observation2)

        T.append([reward, deepcopy(observation),
                  deepcopy(observation2), action, done])

        # [optional] save intermideate model
        if episode % int(args.train_episode / 10) == 0:
            agent.save_model(args.output)

        # update
        step += 1
        episode_steps += 1
        episode_reward += reward
        observation = deepcopy(observation2)

        if done:  # end of episode

            logger.info('#{}: episode_reward:{:.4f} acc: {:.4f}, weight: {:.4f} %'.format(episode, episode_reward,
                                                                                           info['accuracy'],
                                                                                           info['w_ratio'] * 100))

            final_reward = T[-1][0]
            # agent observe and update policy
            for r_t, s_t, s_t1, a_t, done in T:
                agent.observe(final_reward, s_t, s_t1, a_t, done)
                if episode > args.warmup:
                    for _ in range(args.n_update):
                        agent.update_policy()

            agent.memory.append(
                observation,
                agent.select_action(observation, episode=episode),
                0., False
            )

            # reset
            observation = None
            episode_steps = 0
            episode_reward = 0.
            episode += 1
            T = []

            if final_reward > best_reward:
                best_reward = final_reward
                best_policy = env.strategy

            logger.info('best reward: {}\n'.format(best_reward))
            logger.info('best policy: {}\n'.format(best_policy))

    return best_policy, best_reward


if __name__ == '__main__':

    main()
