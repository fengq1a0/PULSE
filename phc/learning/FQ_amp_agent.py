from phc.utils.running_mean_std import RunningMeanStd
from phc.learning.loss_functions import kl_multi
import learning.replay_buffer as replay_buffer
import learning.amp_datasets as amp_datasets
from rl_games.algos_torch import torch_ext, a2c_continuous, central_value
from rl_games.common import a2c_common

from isaacgym.torch_utils import *

import numpy as np
import torch
from torch import nn, optim

import time
import copy
import os
import os.path as osp
import wandb
import sys
sys.path.append(os.getcwd())





class AMPAgent(a2c_continuous.A2CAgent):

    def __init__(self, base_name, config):
        # Call the orginal function directly
        # A2CBase --> ContinuousA2CBase  -->  A2CAgent  --> AMPAgent
        a2c_common.A2CBase.__init__(self, base_name, config)
        self.cfg = config
        self.exp_name = self.cfg['train_dir'].split('/')[-1]

        self._load_config_params(config)

        self.is_discrete = False
        self._setup_action_space()
        self.bounds_loss_coef = config.get('bounds_loss_coef', None)
        
        self.clip_actions = config.get('clip_actions', True)
        self._save_intermediate = config.get('save_intermediate', False)

        net_config = self._build_net_config()
        
        if self.normalize_input:
            if "vec_env" in self.__dict__:
                obs_shape = torch_ext.shape_whc_to_cwh(self.vec_env.env.task.get_running_mean_size())
            else:
                obs_shape = self.obs_shape
            self.running_mean_std = RunningMeanStd(obs_shape).to(self.ppo_device)
            
        net_config['mean_std'] = self.running_mean_std
        self.model = self.network.build(net_config)
        self.model.to(self.ppo_device)
        self.states = None

        self.init_rnn_from_model(self.model)
        self.last_lr = float(self.last_lr)

        self.optimizer = optim.Adam(self.model.parameters(), float(self.last_lr), eps=1e-08, weight_decay=self.weight_decay)

        if self.has_central_value:
            cv_config = {
                'state_shape': torch_ext.shape_whc_to_cwh(self.state_shape),
                'value_size': self.value_size,
                'ppo_device': self.ppo_device,
                'num_agents': self.num_agents,
                'horizon_length': self.horizon_length,
                'num_actors': self.num_actors,
                'num_actions': self.actions_num,
                'seq_len': self.seq_len,
                'model': self.central_value_config['network'],
                'config': self.central_value_config,
                'writter': self.writer,
                'multi_gpu': self.multi_gpu
            }
            self.central_value_net = central_value.CentralValueTrain(**cv_config).to(self.ppo_device)

        self.use_experimental_cv = self.config.get('use_experimental_cv', True)
        self.dataset = amp_datasets.AMPDataset(self.batch_size, self.minibatch_size, self.is_discrete, self.is_rnn, self.ppo_device, self.seq_len)
        self.algo_observer.after_init(self)


        if self.config.get('use_seq_rl', False):
            # Use the is_rnn to force the dataset to have sequencal format. 
            self.dataset = amp_datasets.AMPDataset(self.batch_size, self.minibatch_size, self.is_discrete, True, self.ppo_device, self.seq_len)
        else:
            self.dataset = amp_datasets.AMPDataset(self.batch_size, self.minibatch_size, self.is_discrete, self.is_rnn, self.ppo_device, self.seq_len)


        if self.normalize_value:
            self.value_mean_std = RunningMeanStd((1,)).to(self.ppo_device)  # Override and get new value

        if self._normalize_amp_input:
            self._amp_input_mean_std = RunningMeanStd(self._amp_observation_space.shape).to(self.ppo_device)

        norm_disc_reward = config.get('norm_disc_reward', False)
        if (norm_disc_reward):
            self._disc_reward_mean_std = RunningMeanStd((1,)).to(self.ppo_device)
        else:
            self._disc_reward_mean_std = None

        self.save_kin_info = self.vec_env.env.task.cfg.env.get("save_kin_info", False)
        self.only_kin_loss = self.vec_env.env.task.cfg.env.get("only_kin_loss", False)
        self.temp_running_mean = self.vec_env.env.task.temp_running_mean # use temp running mean to make sure the obs used for training is the same as calc gradient.

        kin_lr = float(self.vec_env.env.task.kin_lr)
        
        if self.save_kin_info:
            self.kin_dict_info = None
            self.kin_optimizer = torch.optim.Adam(self.model.a2c_network.parameters(), kin_lr)

        # ZL Hack
        if self.vec_env.env.task.fitting:
            print("#################### Fitting and freezing!! ####################")
            checkpoint = torch_ext.load_checkpoint(self.vec_env.env.task.models_path[0])
            
            self.set_stats_weights(checkpoint)  # loads mean std. essential for distilling knowledge. will not load if has a shape mismatch.
            self.freeze_state_weights()  # freeze the mean stds.
            # def load_my_state_dict(target, saved_dict):
            #load_my_state_dict(self.model.state_dict(), checkpoint['model'])  # loads everything (model, std, ect.). that can be load from the last model.
            target = self.model.state_dict()
            saved_dict = checkpoint['model']
            for name, param in saved_dict.items():
                if name not in target:
                    continue
                if target[name].shape == param.shape:
                    target[name].copy_(param)
            # self.value_mean_std # not freezing value function though.
        
        return
    
    def set_stats_weights(self, weights):
        if self.normalize_input:
            if weights['running_mean_std']['running_mean'].shape == self.running_mean_std.state_dict()['running_mean'].shape:
                self.running_mean_std.load_state_dict(weights['running_mean_std'])
            else:
                print("shape mismatch, can not load input mean std")
                
        if self.normalize_value:
            self.value_mean_std.load_state_dict(weights['reward_mean_std'])

        if self.has_central_value:
            self.central_value_net.set_stats_weights(weights['assymetric_vf_mean_std'])
 
        if self.mixed_precision and 'scaler' in weights:
            self.scaler.load_state_dict(weights['scaler'])
            
        if self._normalize_amp_input:
            if weights['amp_input_mean_std']['running_mean'].shape == self._amp_input_mean_std.state_dict()['running_mean'].shape:
                self._amp_input_mean_std.load_state_dict(weights['amp_input_mean_std'])
            else:
                print("shape mismatch, can not load AMP mean std")
            

        if (self._norm_disc_reward()):
            self._disc_reward_mean_std.load_state_dict(weights['disc_reward_mean_std'])
            
    def get_full_state_weights(self):
        state = super().get_full_state_weights()
        
        if "kin_optimizer" in self.__dict__:
            print("!!!saving kin_optimizer!!! Remove this message asa p!!")
            state['kin_optimizer'] = self.kin_optimizer.state_dict()

        return state

    def set_full_state_weights(self, weights):
        super().set_full_state_weights(weights)
        if "kin_optimizer" in weights:
            print("!!!loading kin_optimizer!!! Remove this message asa p!!")
            self.kin_optimizer.load_state_dict(weights['kin_optimizer'])
        

    def freeze_state_weights(self):
        if self.normalize_input:
            self.running_mean_std.freeze()
        if self.normalize_value:
            self.value_mean_std.freeze()
        if self.has_central_value:
            raise NotImplementedError()
        if self.mixed_precision:
            raise NotImplementedError()

    def unfreeze_state_weights(self):
        if self.normalize_input:
            self.running_mean_std.unfreeze()
        if self.normalize_value:
            self.value_mean_std.unfreeze()
        if self.has_central_value:
            raise NotImplementedError()
        if self.mixed_precision:
            raise NotImplementedError()

    def init_tensors(self):
        super().init_tensors()
        self.experience_buffer.tensor_dict['next_obses'] = torch.zeros_like(self.experience_buffer.tensor_dict['obses'])
        self.experience_buffer.tensor_dict['next_values'] = torch.zeros_like(self.experience_buffer.tensor_dict['values'])

        self.tensor_list += ['next_obses']

        self._build_amp_buffers()

        if self.save_kin_info:
            B, S, _ = self.experience_buffer.tensor_dict['obses'].shape
            kin_dict = self.vec_env.env.task.kin_dict
            kin_dict_size = np.sum([v.reshape(v.shape[0], -1).shape[-1] for k, v in kin_dict.items()])
            self.experience_buffer.tensor_dict['kin_dict'] = torch.zeros((B, S, kin_dict_size)).to(self.experience_buffer.tensor_dict['obses'])
            self.tensor_list += ['kin_dict']
            
        if self.vec_env.env.task.z_type == "vae":
            B, S, _ = self.experience_buffer.tensor_dict['obses'].shape
            self.experience_buffer.tensor_dict['z_noise'] = torch.zeros(B, S, self.model.a2c_network.embedding_size).to(self.experience_buffer.tensor_dict['obses'])
            self.tensor_list += ['z_noise']
            
        return

    def set_eval(self):
        super().set_eval()
        if self._normalize_amp_input:
            self._amp_input_mean_std.eval()

        if (self._norm_disc_reward()):
            self._disc_reward_mean_std.eval()

        return

    def set_train(self):
        super().set_train()
        if self._normalize_amp_input:
            self._amp_input_mean_std.train()

        if (self._norm_disc_reward()):
            self._disc_reward_mean_std.train()

        return

    def get_stats_weights(self):
        state = super().get_stats_weights()
        if self._normalize_amp_input:
            state['amp_input_mean_std'] = self._amp_input_mean_std.state_dict()

        if (self._norm_disc_reward()):
            state['disc_reward_mean_std'] = self._disc_reward_mean_std.state_dict()

        return state
    

    def play_steps_rnn(self):
        self.set_eval()
        mb_rnn_states = []
        epinfos = []
        self.experience_buffer.tensor_dict['values'].fill_(0)
        self.experience_buffer.tensor_dict['rewards'].fill_(0)
        self.experience_buffer.tensor_dict['dones'].fill_(1)
        step_time = 0.0

        update_list = self.update_list

        batch_size = self.num_agents * self.num_actors
        mb_rnn_masks = None

        mb_rnn_masks, indices, steps_mask, steps_state, play_mask, mb_rnn_states = self.init_rnn_step(batch_size, mb_rnn_states) # mb_rnn_states means "memory bank" rnn states

        ### ZL
        done_indices = []
        terminated_flags = torch.zeros(self.num_actors, device=self.device)
        reward_raw = torch.zeros(1, device=self.device)

        for n in range(self.horizon_length):
            
            
            
            self.obs = self.env_reset(done_indices)
            
            # self.rnn_states[0][:, :, -1] = n; print('debugg!!!!')
            # self.rnn_states[0][:, :, -2] = torch.arange(self.num_actors)
            
            seq_indices, full_tensor = self.process_rnn_indices(mb_rnn_masks, indices, steps_mask, steps_state, mb_rnn_states)  # this should upate mb_rnn_states
            if full_tensor:
                break
            
            if self.has_central_value:
                self.central_value_net.pre_step_rnn(self.last_rnn_indices, self.last_state_indices)

            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else:
                res_dict = self.get_action_values(self.obs)
            
            self.rnn_states = res_dict['rnn_states']
            self.experience_buffer.update_data_rnn('obses', indices, play_mask, self.obs['obs'])

            for k in update_list:
                self.experience_buffer.update_data_rnn(k, indices, play_mask, res_dict[k])

            if self.has_central_value:
                self.experience_buffer.update_data_rnn('states', indices[::self.num_agents], play_mask[::self.num_agents] // self.num_agents, self.obs['states'])

            if self.only_kin_loss:
                # pure behavior cloning, kinemaitc loss. 
                self.obs, rewards, self.dones, infos = self.env_step(res_dict['mus'])
            else:
                self.obs, rewards, self.dones, infos = self.env_step(res_dict['actions'])
            
                
            shaped_rewards = self.rewards_shaper(rewards)

            if self.value_bootstrap and 'time_outs' in infos:
                shaped_rewards += self.gamma * res_dict['values'] * self.cast_obs(infos['time_outs']).unsqueeze(1).float()
            self.experience_buffer.update_data_rnn('rewards', indices, play_mask, shaped_rewards)
            self.experience_buffer.update_data_rnn('next_obses', indices, play_mask, self.obs['obs'])
            self.experience_buffer.update_data_rnn('dones', indices, play_mask, self.dones.byte())
            self.experience_buffer.update_data_rnn('amp_obs', indices, play_mask, infos['amp_obs'])

            ### ZL
            terminated = infos['terminate'].float()
            terminated_flags += terminated
            reward_raw_mean = infos['reward_raw'].mean(dim=0)

            if reward_raw.shape != reward_raw_mean.shape:
                reward_raw = reward_raw_mean
            else:
                reward_raw += reward_raw_mean

            terminated = terminated.unsqueeze(-1)
            input_dict = {"obs": self.obs['obs'], "rnn_states": self.rnn_states}
            next_vals = self._eval_critic(input_dict)  # ZL this has issues? (maybe not, since we are passing the states in.)
            next_vals *= (1.0 - terminated)
            self.experience_buffer.update_data_rnn('next_values', indices, play_mask, next_vals)

            self.current_rewards += rewards
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            done_indices = all_done_indices[::self.num_agents]

            self.process_rnn_dones(all_done_indices, indices, seq_indices)

            if self.has_central_value:
                self.central_value_net.post_step_rnn(all_done_indices)

            self.algo_observer.process_infos(infos, done_indices)

            fdones = self.dones.float()
            not_dones = 1.0 - self.dones.float()

            self.game_rewards.update(self.current_rewards[done_indices])
            self.game_lengths.update(self.current_lengths[done_indices])

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones
            
            if self.only_kin_loss:
                self.experience_buffer.update_data_rnn('kin_dict', indices, play_mask, torch.cat([v.reshape(v.shape[0], -1) for k, v in infos['kin_dict'].items()], dim = -1))
                if self.kin_dict_info is None:
                    self.kin_dict_info = {k: (v.shape, v.reshape(v.shape[0], -1).shape) for k, v in infos['kin_dict'].items()}

            if (self.vec_env.env.task.viewer):
                self._amp_debug(infos)

            done_indices = done_indices[:, 0]
            

        mb_fdones = self.experience_buffer.tensor_dict['dones'].float()
        mb_values = self.experience_buffer.tensor_dict['values']
        mb_next_values = self.experience_buffer.tensor_dict['next_values']

        mb_rewards = self.experience_buffer.tensor_dict['rewards']
        mb_amp_obs = self.experience_buffer.tensor_dict['amp_obs']
        amp_rewards = self._calc_amp_rewards(mb_amp_obs)
        mb_rewards = self._combine_rewards(mb_rewards, amp_rewards)
        

        mb_advs = self.discount_values(mb_fdones, mb_values, mb_rewards, mb_next_values)
        mb_returns = mb_advs + mb_values
        
        # self.experience_buffer.tensor_dict['actions']: is num_env, Batch, feat. That's why we swap and flatten, mb_rnn_states is already in that format. 
        batch_dict = self.experience_buffer.get_transformed_list(a2c_common.swap_and_flatten01, self.tensor_list) # swap to step, num_envs, feat
        batch_dict['returns'] = a2c_common.swap_and_flatten01(mb_returns)
        batch_dict['rnn_states'] = mb_rnn_states
        
        batch_dict['rnn_masks'] = mb_rnn_masks # ZL: this should be swap and flattened, but it's all ones for now
        batch_dict['terminated_flags'] = terminated_flags
        batch_dict['reward_raw'] =reward_raw / self.horizon_length
        
        batch_dict['played_frames'] = n * self.num_actors * self.num_agents
        batch_dict['step_time'] = step_time
        

        for k, v in amp_rewards.items():
            batch_dict[k] = a2c_common.swap_and_flatten01(v)

        batch_dict['mb_rewards'] = a2c_common.swap_and_flatten01(mb_rewards)
        
        return batch_dict

    def play_steps(self):
        self.set_eval()
        humanoid_env = self.vec_env.env.task

        epinfos = []
        done_indices = []
        update_list = self.update_list
        terminated_flags = torch.zeros(self.num_actors, device=self.device)
        reward_raw = torch.zeros(1, device=self.device)
        for n in range(self.horizon_length):

            self.obs = self.env_reset(done_indices)
            self.experience_buffer.update_data('obses', n, self.obs['obs'])

            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else:
                res_dict = self.get_action_values(self.obs)
                
            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k])

            if self.has_central_value:
                self.experience_buffer.update_data('states', n, self.obs['states'])
            
            if self.only_kin_loss and self.save_kin_info:
                self.obs, rewards, self.dones, infos = self.env_step(res_dict['mus'])
            else:
                self.obs, rewards, self.dones, infos = self.env_step(res_dict['actions'])
                
            shaped_rewards = self.rewards_shaper(rewards)
            self.experience_buffer.update_data('rewards', n, shaped_rewards)
            self.experience_buffer.update_data('next_obses', n, self.obs['obs'])
            self.experience_buffer.update_data('dones', n, self.dones)
            self.experience_buffer.update_data('amp_obs', n, infos['amp_obs'])
            
            if self.save_kin_info:
                self.experience_buffer.update_data('kin_dict', n, torch.cat([v.reshape(v.shape[0], -1) for k, v in infos['kin_dict'].items()], dim = -1))
                
                if self.kin_dict_info is None:
                    self.kin_dict_info = {k: (v.shape, v.reshape(v.shape[0], -1).shape) for k, v in infos['kin_dict'].items()}

                
            terminated = infos['terminate'].float()
            terminated_flags += terminated

            reward_raw_mean = infos['reward_raw'].mean(dim=0)
            if reward_raw.shape != reward_raw_mean.shape:
                reward_raw = reward_raw_mean
            else:
                reward_raw += reward_raw_mean
            terminated = terminated.unsqueeze(-1)

            next_vals = self._eval_critic(self.obs)
            next_vals *= (1.0 - terminated)
            self.experience_buffer.update_data('next_values', n, next_vals)
            
            self.current_rewards += rewards
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            done_indices = all_done_indices[::self.num_agents]
            self.game_rewards.update(self.current_rewards[done_indices])
            self.game_lengths.update(self.current_lengths[done_indices])
            self.algo_observer.process_infos(infos, done_indices)

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

            if (self.vec_env.env.task.viewer):
                self._amp_debug(infos)

            done_indices = done_indices[:, 0]

        mb_fdones = self.experience_buffer.tensor_dict['dones'].float()
        mb_values = self.experience_buffer.tensor_dict['values']
        mb_next_values = self.experience_buffer.tensor_dict['next_values']

        mb_rewards = self.experience_buffer.tensor_dict['rewards']
        mb_amp_obs = self.experience_buffer.tensor_dict['amp_obs']
        amp_rewards = self._calc_amp_rewards(mb_amp_obs)
        mb_rewards = self._combine_rewards(mb_rewards, amp_rewards)
        mb_advs = self.discount_values(mb_fdones, mb_values, mb_rewards, mb_next_values)
        mb_returns = mb_advs + mb_values

        batch_dict = self.experience_buffer.get_transformed_list(a2c_common.swap_and_flatten01, self.tensor_list)
        batch_dict['returns'] = a2c_common.swap_and_flatten01(mb_returns)
        batch_dict['terminated_flags'] = terminated_flags
        batch_dict['reward_raw'] =reward_raw / self.horizon_length
        batch_dict['played_frames'] = self.batch_size
        
        for k, v in amp_rewards.items():
            batch_dict[k] = a2c_common.swap_and_flatten01(v)
        batch_dict['mb_rewards'] = a2c_common.swap_and_flatten01(mb_rewards)
        
        return batch_dict

    def prepare_dataset(self, batch_dict):
        obses = batch_dict['obses']
        returns = batch_dict['returns']
        dones = batch_dict['dones']
        values = batch_dict['values']
        actions = batch_dict['actions']
        neglogpacs = batch_dict['neglogpacs']
        mus = batch_dict['mus']
        sigmas = batch_dict['sigmas']
        rnn_states = batch_dict.get('rnn_states', None)
        rnn_masks = batch_dict.get('rnn_masks', None)

        advantages = self._calc_advs(batch_dict)

        if self.normalize_value:
            values = self.value_mean_std(values)
            returns = self.value_mean_std(returns)

        dataset_dict = {}
        dataset_dict['old_values'] = values
        dataset_dict['old_logp_actions'] = neglogpacs
        dataset_dict['advantages'] = advantages
        dataset_dict['returns'] = returns
        dataset_dict['actions'] = actions
        dataset_dict['obs'] = obses
        dataset_dict['rnn_states'] = rnn_states
        dataset_dict['rnn_masks'] = rnn_masks
        dataset_dict['mu'] = mus
        dataset_dict['sigma'] = sigmas
        
        if self.has_central_value:
            dataset_dict = {}
            dataset_dict['old_values'] = values
            dataset_dict['advantages'] = advantages
            dataset_dict['returns'] = returns
            dataset_dict['actions'] = actions
            dataset_dict['obs'] = batch_dict['states']
            dataset_dict['rnn_masks'] = rnn_masks
            self.central_value_net.update_dataset(dataset_dict)
            
        self.dataset.update_values_dict(dataset_dict)
        
        dataset_dict['amp_obs'] = batch_dict['amp_obs']
        dataset_dict['amp_obs_demo'] = batch_dict['amp_obs_demo']
        dataset_dict['amp_obs_replay'] = batch_dict['amp_obs_replay']

        if self.save_kin_info:
            dataset_dict['kin_dict'] = batch_dict['kin_dict']
        
        if self.vec_env.env.task.z_type == "vae":
            dataset_dict['z_noise'] = batch_dict['z_noise']
            
        self.dataset.update_values_dict(dataset_dict, rnn_format = True, horizon_length = self.horizon_length, num_envs = self.num_actors)
        # self.dataset.update_values_dict(dataset_dict)

        return

    def train_epoch(self):
        self.pre_epoch(self.epoch_num)
        play_time_start = time.time()

        ### ZL: do not update state weights during play

        with torch.no_grad():
            if self.is_rnn:
                batch_dict = self.play_steps_rnn()
            else:
                batch_dict = self.play_steps()

        play_time_end = time.time()
        update_time_start = time.time()
        rnn_masks = batch_dict.get('rnn_masks', None)

        self._update_amp_demos()
        num_obs_samples = batch_dict['amp_obs'].shape[0]
        amp_obs_demo = self._amp_obs_demo_buffer.sample(num_obs_samples)['amp_obs']
        batch_dict['amp_obs_demo'] = amp_obs_demo

        if (self._amp_replay_buffer.get_total_count() == 0):
            batch_dict['amp_obs_replay'] = batch_dict['amp_obs']
        else:
            batch_dict['amp_obs_replay'] = self._amp_replay_buffer.sample(num_obs_samples)['amp_obs']

        self.set_train()

        self.curr_frames = batch_dict.pop('played_frames')
        
        self.prepare_dataset(batch_dict)
        self.algo_observer.after_steps()

        if self.has_central_value:
            self.train_central_value()

        train_info = None

        # if self.is_rnn:
        # frames_mask_ratio = rnn_masks.sum().item() / (rnn_masks.nelement())

        for _ in range(0, self.mini_epochs_num):
            ep_kls = []
            for i in range(len(self.dataset)):
                curr_train_info = self.train_actor_critic(self.dataset[i])

                if self.schedule_type == 'legacy':
                    if self.multi_gpu:
                        curr_train_info['kl'] = self.hvd.average_value(curr_train_info['kl'], 'ep_kls')
                    self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0, curr_train_info['kl'].item())
                    self.update_lr(self.last_lr)

                if (train_info is None):
                    train_info = dict()
                    for k, v in curr_train_info.items():
                        train_info[k] = [v]
                else:
                    for k, v in curr_train_info.items():
                        train_info[k].append(v)

            av_kls = torch_ext.mean_list(train_info['kl'])

            if self.schedule_type == 'standard':
                if self.multi_gpu:
                    av_kls = self.hvd.average_value(av_kls, 'ep_kls')
                self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0, av_kls.item())
                self.update_lr(self.last_lr)

        if self.schedule_type == 'standard_epoch':
            if self.multi_gpu:
                av_kls = self.hvd.average_value(torch_ext.mean_list(kls), 'ep_kls')
            self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0, av_kls.item())
            self.update_lr(self.last_lr)
            
        update_time_end = time.time()
        play_time = play_time_end - play_time_start
        update_time = update_time_end - update_time_start
        total_time = update_time_end - play_time_start

        self._store_replay_amp_obs(batch_dict['amp_obs'])

        train_info['play_time'] = play_time
        train_info['update_time'] = update_time
        train_info['total_time'] = total_time
        train_info['terminated_flags'] = batch_dict['terminated_flags']
        train_info['reward_raw'] = batch_dict['reward_raw']
        train_info['mb_rewards'] = batch_dict['mb_rewards']
        train_info['returns'] = batch_dict['returns']
        self._record_train_batch_info(batch_dict, train_info)
        self.post_epoch(self.epoch_num)
        
        if self.save_kin_info:
            print_str = "Kin: " + " \t".join([f"{k}: {torch.mean(torch.tensor(train_info[k])):.4f}" for k, v in train_info.items() if k.startswith("kin")])
            print(print_str)
        
        return train_info

    def pre_epoch(self, epoch_num):
        # print("freeze running mean/std")

        if self.vec_env.env.task.humanoid_type in ["smpl", "smplh", "smplx"]:
            humanoid_env = self.vec_env.env.task
            if (epoch_num > 1) and epoch_num % humanoid_env.shape_resampling_interval == 1: # + 1 to evade the evaluations. 
            # if (epoch_num > 0) and epoch_num % humanoid_env.shape_resampling_interval == 0 and not (epoch_num % (self.save_freq)): # Remove the resampling for this. 
                # Different from AMP, always resample motion no matter the motion type.
                print("Resampling Shape")
                humanoid_env.resample_motions()
                # self.current_rewards # Fixing these values such that they do not get whacked by the
                # self.current_lengths
            if humanoid_env.getup_schedule:
                humanoid_env.update_getup_schedule(epoch_num, getup_udpate_epoch=humanoid_env.getup_udpate_epoch)
                if epoch_num > humanoid_env.getup_udpate_epoch:  # ZL fix janky hack
                    self._task_reward_w = 0.5
                    self._disc_reward_w = 0.5
                else:
                    self._task_reward_w = 0
                    self._disc_reward_w = 1

        self.running_mean_std_temp = copy.deepcopy(self.running_mean_std)  # Freeze running mean/std, so that the actor does not use the updated mean/std
        self.running_mean_std_temp.freeze()

    def post_epoch(self, epoch_num):
        self.running_mean_std_temp = copy.deepcopy(self.running_mean_std)  # Unfreeze running mean/std
        self.running_mean_std_temp.freeze()
        

    def _preproc_obs(self, obs_batch, use_temp=False):
        if type(obs_batch) is dict:
            for k, v in obs_batch.items():
                obs_batch[k] = self._preproc_obs(v, use_temp = use_temp)
        else:
            if obs_batch.dtype == torch.uint8:
                obs_batch = obs_batch.float() / 255.0

        if self.normalize_input:
            obs_batch_proc = obs_batch[:, :self.running_mean_std.mean_size]
            if use_temp:
                obs_batch_out = self.running_mean_std_temp(obs_batch_proc)
                obs_batch_orig = self.running_mean_std(obs_batch_proc)  # running through mean std, but do not use its value. use temp
            else:
                obs_batch_out = self.running_mean_std(obs_batch_proc)  # running through mean std, but do not use its value. use temp
            obs_batch_out = torch.cat([obs_batch_out, obs_batch[:, self.running_mean_std.mean_size:]], dim=-1)

        return obs_batch_out

    def calc_gradients(self, input_dict):
        
        self.set_train()
        humanoid_env = self.vec_env.env.task

        value_preds_batch = input_dict['old_values']
        old_action_log_probs_batch = input_dict['old_logp_actions']
        advantage = input_dict['advantages']
        old_mu_batch = input_dict['mu']
        old_sigma_batch = input_dict['sigma']
        return_batch = input_dict['returns']
        actions_batch = input_dict['actions']
        obs_batch = input_dict['obs']
        obs_batch_processed = self._preproc_obs(obs_batch, use_temp=self.temp_running_mean)
        input_dict['obs_processed'] = obs_batch_processed

        amp_obs = input_dict['amp_obs'][0:self._amp_minibatch_size]
        amp_obs = self._preproc_amp_obs(amp_obs)
        
        amp_obs_replay = input_dict['amp_obs_replay'][0:self._amp_minibatch_size]
        amp_obs_replay = self._preproc_amp_obs(amp_obs_replay)

        amp_obs_demo = input_dict['amp_obs_demo'][0:self._amp_minibatch_size]
        amp_obs_demo = self._preproc_amp_obs(amp_obs_demo)
        amp_obs_demo.requires_grad_(True)

        lr = self.last_lr
        kl = 1.0
        lr_mul = 1.0
        curr_e_clip = lr_mul * self.e_clip
        
        self.train_result = {}
        if self.only_kin_loss:
            # pure behavior cloning, kinemaitc loss.
            batch_dict = {}
            batch_dict['obs_orig'] = obs_batch
            batch_dict['obs'] = input_dict['obs_processed']
            batch_dict['kin_dict'] = input_dict['kin_dict']
            
            # if humanoid_env.z_type == "vae":
            #     batch_dict['z_noise'] = input_dict['z_noise']
            
            rnn_len = self.horizon_length
            rnn_len = 1
            if self.is_rnn:
                batch_dict['rnn_states'] = input_dict['rnn_states']
                batch_dict['seq_length'] = rnn_len

            kin_loss_info = self._optimize_kin(batch_dict)
            self.train_result.update( {'entropy': torch.tensor(0).float(), 'kl': torch.tensor(0).float(), 'last_lr': self.last_lr, 'lr_mul': torch.tensor(0).float()})
            
        else:
            batch_dict = {'is_train': True, 'amp_steps': self.vec_env.env.task._num_amp_obs_steps, \
                'prev_actions': actions_batch, 'obs': obs_batch_processed, 'amp_obs': amp_obs, 'amp_obs_replay': amp_obs_replay, 'amp_obs_demo': amp_obs_demo, \
                    "obs_orig": obs_batch
                    }
        
            rnn_masks = None
            rnn_len = self.horizon_length
            rnn_len = 1
            if self.is_rnn:
                rnn_masks = input_dict['rnn_masks']
                batch_dict['rnn_states'] = input_dict['rnn_states']
                batch_dict['seq_length'] = rnn_len
                
                
            with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                res_dict = self.model(batch_dict) # current model if RNN, has BPTT enabled. 
                
                action_log_probs = res_dict['prev_neglogp']
                values = res_dict['values']
                entropy = res_dict['entropy']
                mu = res_dict['mus']
                sigma = res_dict['sigmas']
                disc_agent_logit = res_dict['disc_agent_logit']
                disc_agent_replay_logit = res_dict['disc_agent_replay_logit']
                disc_demo_logit = res_dict['disc_demo_logit']

                if not rnn_masks is None:
                    rnn_mask_bool = rnn_masks.squeeze().bool()
                    old_action_log_probs_batch, action_log_probs, advantage, values, entropy, mu, sigma, return_batch, old_mu_batch, old_sigma_batch = \
                        old_action_log_probs_batch[rnn_mask_bool], action_log_probs[rnn_mask_bool], advantage[rnn_mask_bool], values[rnn_mask_bool], \
                            entropy[rnn_mask_bool], mu[rnn_mask_bool], sigma[rnn_mask_bool], return_batch[rnn_mask_bool], old_mu_batch[rnn_mask_bool], old_sigma_batch[rnn_mask_bool]
                    
                    # flatten values for computing loss
                    
                a_info = self._actor_loss(old_action_log_probs_batch, action_log_probs, advantage, curr_e_clip)
                a_loss = a_info['actor_loss']

                c_info = self._critic_loss(value_preds_batch, values, curr_e_clip, return_batch, self.clip_value)
                c_loss = c_info['critic_loss']

                b_loss = self.bound_loss(mu)

                a_loss = torch.mean(a_loss)
                c_loss = torch.mean(c_loss)
                b_loss = torch.mean(b_loss)
                entropy = torch.mean(entropy)

                disc_agent_cat_logit = torch.cat([disc_agent_logit, disc_agent_replay_logit], dim=0)
                
                disc_info = self._disc_loss(disc_agent_cat_logit, disc_demo_logit, amp_obs_demo)
                disc_loss = disc_info['disc_loss']

                loss = a_loss + self.critic_coef * c_loss - self.entropy_coef * entropy + self.bounds_loss_coef * b_loss \
                    + self._disc_coef * disc_loss
                
                
                a_clip_frac = torch.mean(a_info['actor_clipped'].float())

                a_info['actor_loss'] = a_loss
                a_info['actor_clip_frac'] = a_clip_frac
                c_info['critic_loss'] = c_loss

                if self.multi_gpu:
                    self.optimizer.zero_grad()
                else:
                    for param in self.model.parameters():
                        param.grad = None

            self.scaler.scale(loss).backward()
            
            with torch.no_grad():
                reduce_kl = not self.is_rnn
                kl_dist = torch_ext.policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce_kl)
                if self.is_rnn:
                    kl_dist = kl_dist.mean()
            
                    
            #TODO: Refactor this ugliest code of the year
            if self.truncate_grads:
                if self.multi_gpu:
                    self.optimizer.synchronize()
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                    with self.optimizer.skip_synchronize():
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                else:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            
            self.train_result.update( {'entropy': entropy, 'kl': kl_dist, 'last_lr': self.last_lr, 'lr_mul': lr_mul, 'b_loss': b_loss})
            self.train_result.update(a_info)
            self.train_result.update(c_info)
            self.train_result.update(disc_info)
            
        if self.save_kin_info:
            self.train_result.update(kin_loss_info)

        return

    def _assamble_kin_dict(self, kin_dict_flat):
        B = kin_dict_flat.shape[0]
        len_acc = 0
        kin_dict = {}
        for k, v in self.kin_dict_info.items():
            kin_dict[k] = kin_dict_flat[:, len_acc:(len_acc + v[1][-1])].view(B, *v[0][1:])
            len_acc += v[1][-1]
        return kin_dict

    def _optimize_kin(self, batch_dict):
        info_dict = {}
        humanoid_env = self.vec_env.env.task
        if humanoid_env.distill: 
            kin_dict = self._assamble_kin_dict(batch_dict['kin_dict'])
            gt_action = kin_dict['gt_action']

            kin_body_rot_geo_loss, kin_vel_loss_l2 = 0.0, 0.0
            if humanoid_env.z_type == "vae":
                pred_action, pred_action_sigma, extra_dict = self.model.a2c_network.eval_actor(batch_dict, return_extra = True)
                # kin_body_loss = (pred_action - gt_action).pow(2).mean() * 10  ## MSE
                kin_action_loss = torch.norm(pred_action - gt_action, dim=-1).mean()  ## RMSE
                
                vae_mu, vae_log_var = extra_dict['vae_mu'], extra_dict['vae_log_var']
                if humanoid_env.use_vae_prior or humanoid_env.use_vae_fixed_prior:
                    prior_mu, prior_log_var = self.model.a2c_network.compute_prior(batch_dict)
                    KLD = kl_multi(vae_mu, vae_log_var, prior_mu, prior_log_var).mean()
                else:
                    KLD = -0.5 * torch.sum(1 + vae_log_var - vae_mu.pow(2) - vae_log_var.exp()) / vae_mu.shape[0]
                    
                ar1_prior, regu_prior = 0, 0 
                if humanoid_env.use_ar1_prior:
                    time_zs = vae_mu.view(self.minibatch_size // self.horizon_length, self.horizon_length, -1)
                    phi = 0.99
                    
                    error = time_zs[:, 1:] - time_zs[:, :-1] * phi
                    
                    idxes = kin_dict['progress_buf'].view(self.minibatch_size // self.horizon_length, self.horizon_length, -1)
                    
                    not_consecs = ((idxes[:, 1:] - idxes[:, :-1]) != 1).view(-1)
                    error = error.view(-1, error.shape[-1])
                    error[not_consecs] = 0
                    
                    starteres = ((idxes <= 2)[:, 1:] + (idxes <= 2)[:, :-1]).view(-1) # make sure the "drop" is not affected. 
                    error[starteres] = 0
                    
                    ar1_prior = torch.norm(error, dim=-1).mean() 
                    info_dict["kin_ar1"] = ar1_prior
                    
                if humanoid_env.use_vae_prior_regu:
                    prior_mean_regu = ((prior_mu ** 2).mean() + (vae_mu ** 2).mean()) * 0.001 # penalize large prior values
                    prior_var_regu = ((prior_log_var ** 2).mean() + (vae_log_var ** 2).mean()) * 0.001 # penalize large variance values
                    regu_prior = prior_mean_regu + prior_var_regu
                    info_dict["kin_prior_regu"] = regu_prior
                
                kin_loss = kin_action_loss +  KLD * humanoid_env.kld_coefficient + ar1_prior * humanoid_env.ar1_coefficient + regu_prior * 0.005
                
                
                info_dict["kin_action_loss"] = kin_action_loss
                info_dict["kin_KLD"] = KLD
                
                if KLD > 100:
                    import ipdb; ipdb.set_trace()
                    print("KLD is too large, clipping to 10")
                
                ######### KLD annealing #######
                if humanoid_env.kld_anneal:
                    anneal_start_epoch = 2500
                    anneal_end_epoch = 5000
                    min_val = humanoid_env.kld_coefficient_min
                    if self.epoch_num > anneal_start_epoch:
                        humanoid_env.kld_coefficient = (0.01 - min_val) * max((anneal_end_epoch -self.epoch_num) / (anneal_end_epoch - anneal_start_epoch), 0) + min_val
                    info_dict["kin_kld_w"] = humanoid_env.kld_coefficient
                ######### KLD annealing #######
                
                
                    
                    
            else:
                raise NotImplementedError()    
                
            self.kin_optimizer.zero_grad()
            kin_loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
            self.kin_optimizer.step()
            
            info_dict.update({"kin_loss": kin_loss})
            
        return info_dict



    def _load_config_params(self, config):
        self.last_lr = config['learning_rate']

        self._task_reward_w = config['task_reward_w']
        self._disc_reward_w = config['disc_reward_w']

        self._amp_observation_space = self.env_info['amp_observation_space']
        self._amp_batch_size = int(config['amp_batch_size'])
        self._amp_minibatch_size = int(config['amp_minibatch_size'])
        assert (self._amp_minibatch_size <= self.minibatch_size)

        self._disc_coef = config['disc_coef']
        self._disc_logit_reg = config['disc_logit_reg']
        self._disc_grad_penalty = config['disc_grad_penalty']
        self._disc_weight_decay = config['disc_weight_decay']
        self._disc_reward_scale = config['disc_reward_scale']
        self._normalize_amp_input = config.get('normalize_amp_input', True)
        return

    def _build_net_config(self):
        obs_shape = torch_ext.shape_whc_to_cwh(self.obs_shape)
        config = {
            'actions_num': self.actions_num,
            'input_shape': obs_shape,
            'num_seqs': self.num_actors * self.num_agents,
            'value_size': self.env_info.get('value_size', 1),
        }
        config['amp_input_shape'] = self._amp_observation_space.shape
        
        config['task_obs_size_detail'] = self.vec_env.env.task.get_task_obs_size_detail()
        if self.vec_env.env.task.has_task:
            config['self_obs_size'] = self.vec_env.env.task.get_self_obs_size()
            config['task_obs_size'] = self.vec_env.env.task.get_task_obs_size()

        return config

    def _init_train(self):
        self._init_amp_demo_buf()
        return


    def _oracle_loss(self, obs):
        oracle_a, _ = self.oracle_model.a2c_network.eval_actor({"obs": obs})
        model_a, _ = self.model.a2c_network.eval_actor({"obs": obs})
        oracle_loss = (oracle_a - model_a).pow(2).mean(dim=-1) * 50
        return {'oracle_loss': oracle_loss}

    def _disc_loss(self, disc_agent_logit, disc_demo_logit, obs_demo):
        '''
        disc_agent_logit: replay and current episode logit (fake examples)
        disc_demo_logit: disc_demo_logit logit 
        obs_demo: gradient penalty demo obs (real examples)
        '''
        # prediction loss
        disc_loss_agent = self._disc_loss_neg(disc_agent_logit)
        disc_loss_demo = self._disc_loss_pos(disc_demo_logit)
        
        disc_loss = 0.5 * (disc_loss_agent + disc_loss_demo)

        # logit reg
        logit_weights = self.model.a2c_network.get_disc_logit_weights()
        disc_logit_loss = torch.sum(torch.square(logit_weights)) # make weight small??
        disc_loss += self._disc_logit_reg * disc_logit_loss

        # grad penalty
        disc_demo_grad = torch.autograd.grad(disc_demo_logit, obs_demo, grad_outputs=torch.ones_like(disc_demo_logit), create_graph=True, retain_graph=True, only_inputs=True)
        disc_demo_grad = disc_demo_grad[0]

        ### ZL Hack for zeroing out gradient penalty on the shape (406,)
        # if self.vec_env.env.task.__dict__.get("smpl_humanoid", False):
        #     humanoid_env = self.vec_env.env.task
        #     B, feat_dim = disc_demo_grad.shape
        #     shape_obs_dim = 17
        #     if humanoid_env.has_shape_obs:
        #         amp_obs_dim = int(feat_dim / humanoid_env._num_amp_obs_steps)
        #         for i in range(humanoid_env._num_amp_obs_steps):
        #             disc_demo_grad[:,
        #                            ((i + 1) * amp_obs_dim -
        #                             shape_obs_dim):((i + 1) * amp_obs_dim)] = 0

        disc_demo_grad = torch.sum(torch.square(disc_demo_grad), dim=-1)

        disc_grad_penalty = torch.mean(disc_demo_grad)
        disc_loss += self._disc_grad_penalty * disc_grad_penalty

        # weight decay
        if (self._disc_weight_decay != 0):
            disc_weights = self.model.a2c_network.get_disc_weights()
            disc_weights = torch.cat(disc_weights, dim=-1)
            disc_weight_decay = torch.sum(torch.square(disc_weights))
            disc_loss += self._disc_weight_decay * disc_weight_decay

        disc_agent_acc, disc_demo_acc = self._compute_disc_acc(disc_agent_logit, disc_demo_logit)

        # print(f"agent_loss: {disc_loss_agent.item():.3f}  | disc_loss_demo {disc_loss_demo.item():.3f}")
        disc_info = {
            'disc_loss': disc_loss,
            'disc_grad_penalty': disc_grad_penalty.detach(),
            'disc_logit_loss': disc_logit_loss.detach(),
            'disc_agent_acc': disc_agent_acc.detach(),
            'disc_demo_acc': disc_demo_acc.detach(),
            'disc_agent_logit': disc_agent_logit.detach(),
            'disc_demo_logit': disc_demo_logit.detach()
        }
        return disc_info
    
    def _disc_loss_neg(self, disc_logits):
        bce = torch.nn.BCEWithLogitsLoss()
        loss = bce(disc_logits, torch.zeros_like(disc_logits))
        return loss

    def _disc_loss_pos(self, disc_logits):
        bce = torch.nn.BCEWithLogitsLoss()
        loss = bce(disc_logits, torch.ones_like(disc_logits))
        return loss

    def _compute_disc_acc(self, disc_agent_logit, disc_demo_logit):
        agent_acc = disc_agent_logit < 0
        agent_acc = torch.mean(agent_acc.float())
        demo_acc = disc_demo_logit > 0
        demo_acc = torch.mean(demo_acc.float())
        return agent_acc, demo_acc

    def _fetch_amp_obs_demo(self, num_samples):
        amp_obs_demo = self.vec_env.env.fetch_amp_obs_demo(num_samples)
        return amp_obs_demo

    def _build_amp_buffers(self):
        batch_shape = self.experience_buffer.obs_base_shape
        self.experience_buffer.tensor_dict['amp_obs'] = torch.zeros(batch_shape + self._amp_observation_space.shape, device=self.ppo_device)
        amp_obs_demo_buffer_size = int(self.config['amp_obs_demo_buffer_size'])
        self._amp_obs_demo_buffer = replay_buffer.ReplayBuffer(amp_obs_demo_buffer_size, self.ppo_device)  # Demo is the data from the dataset. Real samples

        self._amp_replay_keep_prob = self.config['amp_replay_keep_prob']
        replay_buffer_size = int(self.config['amp_replay_buffer_size'])
        self._amp_replay_buffer = replay_buffer.ReplayBuffer(replay_buffer_size, self.ppo_device)

        self.tensor_list += ['amp_obs']
        return

    def _init_amp_demo_buf(self):
        buffer_size = self._amp_obs_demo_buffer.get_buffer_size()
        num_batches = int(np.ceil(buffer_size / self._amp_batch_size))

        for i in range(num_batches):
            curr_samples = self._fetch_amp_obs_demo(self._amp_batch_size)
            self._amp_obs_demo_buffer.store({'amp_obs': curr_samples})

        return

    def _update_amp_demos(self):
        new_amp_obs_demo = self._fetch_amp_obs_demo(self._amp_batch_size)
        self._amp_obs_demo_buffer.store({'amp_obs': new_amp_obs_demo})
        return

    def _norm_disc_reward(self):
        return self._disc_reward_mean_std is not None

    def _preproc_amp_obs(self, amp_obs):
        if self._normalize_amp_input:
            amp_obs = self._amp_input_mean_std(amp_obs)
        return amp_obs

    def _combine_rewards(self, task_rewards, amp_rewards):
        disc_r = amp_rewards['disc_rewards']

        combined_rewards = self._task_reward_w * task_rewards + \
                         + self._disc_reward_w * disc_r
        return combined_rewards

    def _eval_disc(self, amp_obs):
        proc_amp_obs = self._preproc_amp_obs(amp_obs)
        return self.model.a2c_network.eval_disc(proc_amp_obs)

    def _calc_amp_rewards(self, amp_obs):
        disc_r = self._calc_disc_rewards(amp_obs)
        output = {'disc_rewards': disc_r}
        return output

    def _calc_disc_rewards(self, amp_obs):
        with torch.no_grad():
            disc_logits = self._eval_disc(amp_obs)
            prob = 1 / (1 + torch.exp(-disc_logits))
            disc_r = -torch.log(torch.maximum(1 - prob, torch.tensor(0.0001, device=self.ppo_device)))

            if (self._norm_disc_reward()):
                self._disc_reward_mean_std.train()
                norm_disc_r = self._disc_reward_mean_std(disc_r.flatten())
                disc_r = norm_disc_r.reshape(disc_r.shape)
                disc_r = 0.5 * disc_r + 0.25

            disc_r *= self._disc_reward_scale

        return disc_r

    def _store_replay_amp_obs(self, amp_obs):
        buf_size = self._amp_replay_buffer.get_buffer_size()
        buf_total_count = self._amp_replay_buffer.get_total_count()
        if (buf_total_count > buf_size):
            keep_probs = to_torch(np.array([self._amp_replay_keep_prob] * amp_obs.shape[0]), device=self.ppo_device)
            keep_mask = torch.bernoulli(keep_probs) == 1.0
            amp_obs = amp_obs[keep_mask]

        if (amp_obs.shape[0] > buf_size):
            rand_idx = torch.randperm(amp_obs.shape[0])
            rand_idx = rand_idx[:buf_size]
            amp_obs = amp_obs[rand_idx]

        self._amp_replay_buffer.store({'amp_obs': amp_obs})
        return

    def _record_train_batch_info(self, batch_dict, train_info):
        train_info['disc_rewards'] = batch_dict['disc_rewards']
        return
    
    def _assemble_train_info(self, train_info, frame):
        train_info_dict = {
            "update_time": train_info['update_time'],
            "play_time": train_info['play_time'],
            "last_lr": train_info['last_lr'][-1] * train_info['lr_mul'][-1],
            "lr_mul": train_info['lr_mul'][-1],
            "e_clip": self.e_clip * train_info['lr_mul'][-1],
        }
        
        if "actor_loss" in train_info:
            train_info_dict.update(
                {
                    "a_loss": torch_ext.mean_list(train_info['actor_loss']).item(),
                    "c_loss": torch_ext.mean_list(train_info['critic_loss']).item(),
                    "bounds_loss": torch_ext.mean_list(train_info['b_loss']).item(),
                    "entropy": torch_ext.mean_list(train_info['entropy']).item(),
                    "clip_frac": torch_ext.mean_list(train_info['actor_clip_frac']).item(),
                    "kl": torch_ext.mean_list(train_info['kl']).item(),
                }
            )
        
        if "disc_loss" in train_info:
            disc_reward_std, disc_reward_mean = torch.std_mean(train_info['disc_rewards'])
            train_info_dict.update({
                "disc_loss": torch_ext.mean_list(train_info['disc_loss']).item(),
                "disc_agent_acc": torch_ext.mean_list(train_info['disc_agent_acc']).item(),
                "disc_demo_acc": torch_ext.mean_list(train_info['disc_demo_acc']).item(),
                "disc_agent_logit": torch_ext.mean_list(train_info['disc_agent_logit']).item(),
                "disc_demo_logit": torch_ext.mean_list(train_info['disc_demo_logit']).item(),
                "disc_grad_penalty": torch_ext.mean_list(train_info['disc_grad_penalty']).item(),
                "disc_logit_loss": torch_ext.mean_list(train_info['disc_logit_loss']).item(),
                "disc_reward_mean": disc_reward_mean.item(),
                "disc_reward_std": disc_reward_std.item(),
            })
        
        if "returns" in train_info:
            train_info_dict['returns'] = train_info['returns'].mean().item()
            
        if "mb_rewards" in train_info:
            train_info_dict['mb_rewards'] = train_info['mb_rewards'].mean().item()
        
        # if 'terminated_flags' in train_info:
        #     train_info_dict["success_rate"] =  1 - torch.mean((train_info['terminated_flags'] > 0).float()).item()
        
        if "reward_raw" in train_info:
            for idx, v in enumerate(train_info['reward_raw'].cpu().numpy().tolist()):
                train_info_dict[f"ind_reward.{idx}"] =  v
        
        if "sym_loss" in train_info:
            train_info_dict['sym_loss'] = torch_ext.mean_list(train_info['sym_loss']).item()
        return train_info_dict

    def _amp_debug(self, info):
        with torch.no_grad():
            amp_obs = info['amp_obs']
            amp_obs = amp_obs[0:1]
            disc_pred = self._eval_disc(amp_obs)
            amp_rewards = self._calc_amp_rewards(amp_obs)
            disc_reward = amp_rewards['disc_rewards']

            disc_pred = disc_pred.detach().cpu().numpy()[0, 0]
            disc_reward = disc_reward.cpu().numpy()[0, 0]
            # print("disc_pred: ", disc_pred, disc_reward)
        return


    def _setup_action_space(self):
        action_space = self.env_info['action_space']
        
        self.actions_num = action_space.shape[0]

        # todo introduce device instead of cuda()
        self.actions_low = torch.from_numpy(action_space.low.copy()).float().to(self.ppo_device)
        self.actions_high = torch.from_numpy(action_space.high.copy()).float().to(self.ppo_device)
        return

    def _log_train_info(self, train_info, frame):
        
        for k, v in train_info.items():
            self.writer.add_scalar(k, v, self.epoch_num)
        
        if not wandb.run is None:
            wandb.log(train_info, step=self.epoch_num)
       
        return 
    
    def discount_values(self, mb_fdones, mb_values, mb_rewards, mb_next_values):
        lastgaelam = 0
        mb_advs = torch.zeros_like(mb_rewards)

        for t in reversed(range(self.horizon_length)):
            not_done = 1.0 - mb_fdones[t]
            not_done = not_done.unsqueeze(1)

            delta = mb_rewards[t] + self.gamma * mb_next_values[t] - mb_values[t]
            lastgaelam = delta + self.gamma * self.tau * not_done * lastgaelam
            mb_advs[t] = lastgaelam

        return mb_advs
    
    def env_reset(self, env_ids=None):
        obs = self.vec_env.reset(env_ids)
        obs = self.obs_to_tensors(obs)
        return obs
    
   
    def get_action_values(self, obs):
        obs_orig = obs['obs']
        processed_obs = self._preproc_obs(obs['obs'])
        self.model.eval()
        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'obs' : processed_obs,
            "obs_orig": obs_orig,
            'rnn_states' : self.rnn_states
        }

        with torch.no_grad():
            res_dict = self.model(input_dict)
            if self.has_central_value:
                states = obs['states']
                input_dict = {
                    'is_train': False,
                    'states' : states,
                    #'actions' : res_dict['action'],
                    #'rnn_states' : self.rnn_states
                }
                value = self.get_central_value(input_dict)
                res_dict['values'] = value
        if self.normalize_value:
            res_dict['values'] = self.value_mean_std(res_dict['values'], True)
        return res_dict



    def train(self):
        self.init_tensors()
        self.last_mean_rewards = -100500
        start_time = time.time()
        total_time = 0
        rep_count = 0
        self.frame = 0
        self.obs = self.env_reset()
        self.curr_frames = self.batch_size_envs

        model_output_file = osp.join(self.network_path, self.config['name'])

        if self.multi_gpu:
            self.hvd.setup_algo(self)

        self._init_train()

        while True:
            epoch_start = time.time()

            epoch_num = self.update_epoch()
            train_info = self.train_epoch()

            sum_time = train_info['total_time']
            total_time += sum_time
            frame = self.frame
            if self.multi_gpu:
                self.hvd.sync_stats(self)

            if self.rank == 0:
                scaled_time = sum_time
                scaled_play_time = train_info['play_time']
                curr_frames = self.curr_frames
                self.frame += curr_frames
                fps_step = curr_frames / scaled_play_time
                fps_total = curr_frames / scaled_time

                self.writer.add_scalar('performance/total_fps', curr_frames / scaled_time, frame)
                self.writer.add_scalar('performance/step_fps', curr_frames / scaled_play_time, frame)
                self.writer.add_scalar('info/epochs', epoch_num, frame)
                train_info_dict = self._assemble_train_info(train_info, frame)
                self.algo_observer.after_print_stats(frame, epoch_num, total_time)
                if self.save_freq > 0:
                    
                    if epoch_num % min(50, self.save_best_after) == 0:
                        self.save(model_output_file)
                    
                    if (self._save_intermediate) and (epoch_num % (self.save_freq) == 0):
                        # Save intermediate model every save_freq  epoches
                        int_model_output_file = model_output_file + '_' + str(epoch_num).zfill(8)
                        self.save(int_model_output_file)
                        
                if self.game_rewards.current_size > 0:
                    mean_rewards = self._get_mean_rewards()
                    mean_lengths = self.game_lengths.get_mean()

                    for i in range(self.value_size):
                        self.writer.add_scalar('rewards{0}/frame'.format(i), mean_rewards[i], frame)
                        self.writer.add_scalar('rewards{0}/iter'.format(i), mean_rewards[i], epoch_num)
                        self.writer.add_scalar('rewards{0}/time'.format(i), mean_rewards[i], total_time)

                    self.writer.add_scalar('episode_lengths/frame', mean_lengths, frame)
                    self.writer.add_scalar('episode_lengths/iter', mean_lengths, epoch_num)

                    if (self._save_intermediate) and (epoch_num % (self.save_freq) == 0):
                        eval_info = self.eval()
                        train_info_dict.update(eval_info)
                    
                    train_info_dict.update({"episode_lengths": mean_lengths, "mean_rewards": np.mean(mean_rewards)})
                    self._log_train_info(train_info_dict, frame)

                    epoch_end = time.time()
                    log_str = f"{self.exp_name}-Ep: {self.epoch_num}\trwd: {np.mean(mean_rewards):.1f}\tfps_step: {fps_step:.1f}\tfps_total: {fps_total:.1f}\tep_time:{epoch_end - epoch_start:.1f}\tframe: {self.frame}\teps_len: {mean_lengths:.1f}"
                    
                    print(log_str)

                    if self.has_self_play_config:
                        self.self_play_manager.update(self)

                if epoch_num > self.max_epochs:
                    self.save(model_output_file)
                    print('MAX EPOCHS NUM!')
                    return self.last_mean_rewards, epoch_num

                update_time = 0
        return


    def bound_loss(self, mu):
        if self.bounds_loss_coef is not None:
            soft_bound = 1.0
            mu_loss_high = torch.clamp_min(mu - soft_bound, 0.0)**2
            mu_loss_low = torch.clamp_max(mu + soft_bound, 0.0)**2
            b_loss = (mu_loss_low + mu_loss_high).sum(axis=-1)
        else:
            b_loss = 0
        return b_loss

    def eval(self):
        raise NotImplementedError("evaluation routine not implemented")
        print("evaluation routine not implemented")
        return {}
    

    def _actor_loss(self, old_action_log_probs_batch, action_log_probs, advantage, curr_e_clip):
        ratio = torch.exp(old_action_log_probs_batch - action_log_probs)
        surr1 = advantage * ratio
        surr2 = advantage * torch.clamp(ratio, 1.0 - curr_e_clip, 1.0 + curr_e_clip)
        a_loss = torch.max(-surr1, -surr2)

        clipped = torch.abs(ratio - 1.0) > curr_e_clip
        clipped = clipped.detach()

        info = {'actor_loss': a_loss, 'actor_clipped': clipped.detach()}
        return info
    
    def _calc_advs(self, batch_dict):
        returns = batch_dict['returns']
        values = batch_dict['values']

        advantages = returns - values
        advantages = torch.sum(advantages, axis=1)

        if self.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages
    

    def _critic_loss(self, value_preds_batch, values, curr_e_clip, return_batch, clip_value):
        if clip_value:
            value_pred_clipped = value_preds_batch + \
                    (values - value_preds_batch).clamp(-curr_e_clip, curr_e_clip)
            value_losses = (values - return_batch)**2
            value_losses_clipped = (value_pred_clipped - return_batch)**2
            c_loss = torch.max(value_losses, value_losses_clipped)
        else:
            c_loss = (return_batch - values)**2

        info = {'critic_loss': c_loss}
        return info
    

    def _eval_critic(self, obs_dict):
        self.model.eval()
        obs_dict['obs'] = self._preproc_obs(obs_dict['obs'])
        if self.model.is_rnn():
            value, state = self.model.a2c_network.eval_critic(obs_dict)
        else:
            value = self.model.a2c_network.eval_critic(obs_dict)

        if self.normalize_value:
            value = self.value_mean_std(value, True)
        return value
    
    def _get_mean_rewards(self):
        return self.game_rewards.get_mean()
