import os.path as osp
import re
from collections import OrderedDict
from types import MethodType

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import HOOKS
from mmcv.runner.checkpoint import load_checkpoint, save_checkpoint
from mmcv.runner.dist_utils import master_only
from mmcv.runner.hooks import Hook
from torch.nn import Conv2d
from torch.nn.modules.batchnorm import _BatchNorm

# These grad_fn pattern are flags of specific a nn.Module
CONV = ('ThnnConv2DBackward', 'CudnnConvolutionBackward')
FC = ('ThAddmmBackward', 'AddmmBackward', 'MmBackward')
BN = ('ThnnBatchNormBackward', 'CudnnBatchNormBackward')
# the modules which contains NON_PASS grad_fn need to change the parameter size
# according to channels after pruning
NON_PASS = CONV + FC


@HOOKS.register_module()
class FisherPruningHook(Hook):
    """Use fisher information to pruning the model, must register after
    optimizer hook.

    Args:
        pruning (bool): When True, the model in pruning process,
            when False, the model is in finetune process.
            Default: True
        delta (str): "acts" or "flops", prune the model by
            "acts" or flops. Default: "acts"
        batch_size (int): The batch_size when pruning model.
            Default: 2
        interval (int): The interval of  pruning two channels.
            Default: 10
        deploy_from (str): Path of checkpoint containing the structure
            of pruning model. Defaults to None and only effective
            when pruning is set True.
        save_flops_thr  (list): Checkpoint would be saved when
            the flops reached specific value in the list:
            Default: [0.75, 0.5, 0.25]
        save_acts_thr (list): Checkpoint would be saved when
            the acts reached specific value in the list:
            Default: [0.75, 0.5, 0.25]
    """
    def __init__(
        self,
        pruning=True,
        delta='acts',
        batch_size=2,
        interval=10,
        deploy_from=None,
        save_flops_thr=[0.75, 0.5, 0.25],
        save_acts_thr=[0.75, 0.5, 0.25],
    ):

        assert delta in ('acts', 'flops')
        self.pruning = pruning
        self.delta = delta
        self.interval = interval
        self.batch_size = batch_size
        # The key of self.input is conv module, and value of it
        # is list of conv' input_features in forward process
        self.conv_inputs = {}
        # The key of self.flops is conv module, and value of it
        # is the summation of conv's flops in forward process
        self.flops = {}
        # The key of self.acts is conv module, and value of it
        # is number of all the out feature's activations(N*C*H*W)
        # in forward process
        self.acts = {}
        # The key of self.temp_fisher_info is conv module, and value
        # is a temporary variable used to estimate fisher.
        self.temp_fisher_info = {}

        # The key of self.batch_fishers is conv module, and value
        # is the estimation of fisher by single batch.
        self.batch_fishers = {}

        # The key of self.accum_fishers is conv module, and value
        # is the estimation of parameter's fisher by all the batch
        # during number of self.interval iterations.
        self.accum_fishers = {}
        self.channels = 0
        self.delta = delta
        self.deploy_from = deploy_from

        for i in range(len(save_acts_thr) - 1):
            assert save_acts_thr[i] > save_acts_thr[i + 1]
        for i in range(len(save_flops_thr) - 1):
            assert save_flops_thr[i] > save_flops_thr[i + 1]

        self.save_flops_thr = save_flops_thr
        self.save_acts_thr = save_acts_thr
        if self.pruning:
            assert torch.__version__.startswith('1.3'), (
                'Due to the frequent changes of the autograd '
                'interface, we only guarantee it works well in pytorch==1.3.')

    def after_build_model(self, model):
        """Remove all pruned channels in finetune stage.

        We add this function to ensure that this happens before DDP's
        optimizer's initialization
        """

        if not self.pruning:
            for name, module in model.named_modules():
                add_pruning_attrs(module)
            assert self.deploy_from, 'You have to give a ckpt' \
                'containing the structure information of the pruning model'
            load_checkpoint(model, self.deploy_from)
            deploy_pruning(model)

    def before_run(self, runner):
        """Initialize the relevant variables(fisher, flops and acts) for
        calculating the importance of the channel, and use the layer-grouping
        algorithm to make the coupled module shared the mask of input
        channel."""

        self.conv_names = OrderedDict()
        self.bn_names = OrderedDict()
        self.logger = runner.logger

        for n, m in runner.model.named_modules():
            if self.pruning:
                add_pruning_attrs(m, pruning=self.pruning)
            if isinstance(m, Conv2d):
                m.name = n
                self.conv_names[m] = n
            if isinstance(m, _BatchNorm):
                m.name = n
                self.bn_names[m] = n

        model = runner.model

        if self.pruning:
            # divide the conv to several group and all convs in same
            # group used same input at least once in model's
            # forward process.
            model.eval()
            self.set_group_masks(model)
            model.train()
            self.construct_outchannel_masks()
            for conv, name in self.conv_names.items():
                self.conv_inputs[conv] = []
                self.temp_fisher_info[conv] = conv.weight.data.new_zeros(
                    self.batch_size, conv.in_channels)
                self.accum_fishers[conv] = conv.weight.data.new_zeros(
                    conv.in_channels)
            for group_id in self.groups:
                module = self.groups[group_id][0]
                self.temp_fisher_info[group_id] = conv.weight.data.new_zeros(
                    self.batch_size, module.in_channels)
                self.accum_fishers[group_id] = conv.weight.data.new_zeros(
                    module.in_channels)
            self.register_hooks()
            self.init_flops_acts()
            self.init_temp_fishers()
        self.print_model(runner, print_flops_acts=False)

    def after_train_iter(self, runner):
        if not self.pruning:
            return
        self.group_fishers()
        if runner.world_size > 1:
            self.reduce_fishers()
        self.accumulate_fishers()
        self.init_temp_fishers()
        if self.every_n_iters(runner, self.interval):
            self.channel_prune()
            self.init_accum_fishers()
            self.print_model(runner, print_channel=False)
        self.init_flops_acts()

    @master_only
    def print_model(self, runner, print_flops_acts=True, print_channel=True):
        """Print the related information of the current model.

        Args:
            runner (Runner): Runner in mmcv
            print_flops_acts (bool): Print the remained percentage of
                flops and acts
            print_channel (bool): Print information about
                the number of reserved channels.
        """

        if print_flops_acts:
            flops, acts = self.compute_flops_acts()
            runner.logger.info('Flops: {:.2f}%, Acts: {:.2f}%'.format(
                flops * 100, acts * 100))
            if len(self.save_flops_thr):
                flops_thr = self.save_flops_thr[0]
                if flops < flops_thr:
                    self.save_flops_thr.pop(0)
                    path = osp.join(
                        runner.work_dir, 'flops_{:.0f}_acts_{:.0f}.pth'.format(
                            flops * 100, acts * 100))
                    save_checkpoint(runner.model, filename=path)
            if len(self.save_acts_thr):
                acts_thr = self.save_acts_thr[0]
                if acts < acts_thr:
                    self.save_acts_thr.pop(0)
                    path = osp.join(
                        runner.work_dir, 'acts_{:.0f}_flops_{:.0f}.pth'.format(
                            acts * 100, flops * 100))
                    save_checkpoint(runner.model, filename=path)
        if print_channel:
            for module, name in self.conv_names.items():
                chans_i = int(module.in_mask.sum().cpu().numpy())
                chans_o = int(module.out_mask.sum().cpu().numpy())
                runner.logger.info(
                    '{}: input_channels: {}/{}, out_channels: {}/{}'.format(
                        name, chans_i, module.in_channels, chans_o,
                        module.out_channels))

    def compute_flops_acts(self):
        """Computing the flops and activation remains."""
        flops = 0
        max_flops = 0
        acts = 0
        max_acts = 0
        for module, name in self.conv_names.items():
            max_flop = self.flops[module]
            i_mask = module.in_mask
            o_mask = module.out_mask
            flops += max_flop / (i_mask.numel() * o_mask.numel()) * (
                i_mask.sum() * o_mask.sum())
            max_flops += max_flop
            max_act = self.acts[module]
            acts += max_act / o_mask.numel() * o_mask.sum()
            max_acts += max_act
        return flops.cpu().numpy() / max_flops, acts.cpu().numpy() / max_acts

    def init_accum_fishers(self):
        """Clear accumulated fisher info."""
        for module, name in self.conv_names.items():
            self.accum_fishers[module].zero_()
        for group in self.groups:
            self.accum_fishers[group].zero_()

    def find_pruning_channel(self, module, fisher, in_mask, info):
        """Find the the channel of a model to pruning.

        Args:
            module (nn.Conv | int ): Conv module of model or idx of self.group
            fisher(Tensor): the fisher information of module's in_mask
            in_mask (Tensor): the squeeze in_mask of modules
            info (dict): store the channel of which module need to pruning
                module: the module has channel need to pruning
                channel: the index of channel need to pruning
                min : the value of fisher / delta

        Returns:
            dict: store the current least important channel
                module: the module has channel need to be pruned
                channel: the index of channel need be to pruned
                min : the value of fisher / delta
        """
        module_info = {}
        if fisher.sum() > 0 and in_mask.sum() > 0:
            nonzero = in_mask.nonzero().view(-1)
            fisher = fisher[nonzero]
            min_value, argmin = fisher.min(dim=0)
            if min_value < info['min']:
                module_info['module'] = module
                module_info['channel'] = nonzero[argmin]
                module_info['min'] = min_value
        return module_info

    def single_prune(self, info, exclude=None):
        """Find the channel with smallest fisher / delta in modules not in
        group.

        Args:
            info (dict): Store the channel of which module need
                to pruning
                module: the module has channel need to pruning
                channel: the index of channel need to pruning
                min : the value of fisher / delta
            exclude (list): List contains all modules in group.
                Default: None

        Returns:
            dict: store the channel of which module need to be pruned
                module: the module has channel need to be pruned
                channel: the index of channel need be to pruned
                min : the value of fisher / delta
        """

        for module, name in self.conv_names.items():
            if exclude is not None and module in exclude:
                continue
            fisher = self.accum_fishers[module]
            in_mask = module.in_mask.view(-1)
            ancestors = self.conv2ancest[module]
            if self.delta == 'flops':
                # delta_flops is a value indicate how much flops is
                # reduced in entire forward process after we set a
                # zero in `in_mask` of a specific conv_module.
                delta_flops = self.flops[module] * module.out_mask.sum() / (
                    module.in_channels * module.out_channels)
                for ancestor in ancestors:
                    delta_flops += self.flops[ancestor] * ancestor.in_mask.sum(
                    ) / (ancestor.in_channels * ancestor.out_channels)
                fisher /= (float(delta_flops) / 1e9)
            if self.delta == 'acts':
                delta_acts = 0
                for ancestor in ancestors:
                    delta_acts += self.acts[ancestor] / ancestor.out_channels
                fisher /= (float(max(delta_acts, 1.)) / 1e6)
            info.update(
                self.find_pruning_channel(module, fisher, in_mask, info))
        return info

    def channel_prune(self):
        """Select the channel in model with smallest fisher / delta set
        corresponding in_mask 0."""

        info = {'module': None, 'channel': None, 'min': 1e9}
        info.update(self.single_prune(info, self.group_modules))
        for group in self.groups:
            in_mask = self.groups[group][0].in_mask.view(-1)
            fisher = self.accum_fishers[group].double()
            if self.delta == 'flops':
                fisher /= float(self.flops[group] / 1e9)
            elif self.delta == 'acts':
                fisher /= float(self.acts[group] / 1e6)
            info.update(self.find_pruning_channel(group, fisher, in_mask,
                                                  info))
        module, channel = info['module'], info['channel']
        if isinstance(module, int):
            for m in self.groups[module]:
                m.in_mask[0, channel] = 0
        elif module is not None:
            module.in_mask[0, channel] = 0

    def accumulate_fishers(self):
        """Accumulate all the fisher during self.interval iterations."""

        for module, name in self.conv_names.items():
            self.accum_fishers[module] += self.batch_fishers[module]
        for group in self.groups:
            self.accum_fishers[group] += self.batch_fishers[group]

    def reduce_fishers(self):
        """Collect fisher from all rank."""
        for module, name in self.conv_names.items():
            dist.all_reduce(self.batch_fishers[module])
        for group in self.groups:
            dist.all_reduce(self.batch_fishers[group])

    def group_fishers(self):
        """Accumulate all module.in_mask's fisher and flops in same group."""
        for group in self.groups:
            self.flops[group] = 0
            self.acts[group] = 0
            for module in self.groups[group]:
                module_fisher = self.temp_fisher_info[module]
                self.temp_fisher_info[group] += module_fisher
                delta_flops = self.flops[module] // module.in_channels // \
                    module.out_channels * module.out_mask.sum()
                self.flops[group] += delta_flops

            self.batch_fishers[group] = (
                self.temp_fisher_info[group]**2).sum(0)

            for module in self.ancest[group]:
                delta_flops = self.flops[module] // module.out_channels // \
                        module.in_channels * module.in_mask.sum()
                self.flops[group] += delta_flops
                acts = self.acts[module] // module.out_channels
                self.acts[group] += acts
        for module, name in self.conv_names.items():
            self.batch_fishers[module] = (
                self.temp_fisher_info[module]**2).sum(0)

    def init_flops_acts(self):
        """Clear the flops and acts of model in last iter."""
        for module, name in self.conv_names.items():
            self.flops[module] = 0
            self.acts[module] = 0

    def init_temp_fishers(self):
        """Clear fisher info of single conv and group."""
        for module, name in self.conv_names.items():
            self.temp_fisher_info[module].zero_()
        for group in self.groups:
            self.temp_fisher_info[group].zero_()

    def register_hooks(self):
        """Register forward and backward hook to Conv module."""
        for module, name in self.conv_names.items():
            module.register_forward_hook(self.save_input_forward_hook)
            module.register_backward_hook(self.compute_fisher_backward_hook)

    def save_input_forward_hook(self, module, inputs, outputs):
        """Save the input and flops and acts for computing fisher and flops or
        acts.

        Args:
            module (nn.Module): the module of register hook
            inputs (tuple): input of module
            outputs (tuple): out of module
        """

        n, oc, oh, ow = outputs.shape
        ic = module.in_channels // module.groups
        kh, kw = module.kernel_size
        self.flops[module] += np.prod([n, oc, oh, ow, ic, kh, kw])
        self.acts[module] += np.prod([n, oc, oh, ow])
        # a conv module may has several inputs in graph,for example
        # head in Retinanet
        if inputs[0].requires_grad:
            self.conv_inputs[module].append(inputs)

    def compute_fisher_backward_hook(self, module, grad_input, *args):
        """
        Args:
            module (nn.Module): module register hooks
            grad_input (tuple): tuple contains grad of input and parameters,
                grad_input[0]is the grad of input in Pytorch 1.3, it seems
                has changed in Higher version
        """
        def compute_fisher(input, grad_input, layer_name):
            grads = input * grad_input
            if layer_name == 'Conv2d':
                grads = grads.sum(-1).sum(-1)
            return grads

        if module in self.conv_names and grad_input[0] is not None:
            layer_name = type(module).__name__
            feature = self.conv_inputs[module].pop(-1)[0]
            grad_feature = grad_input[0]
            # avoid that last batch is't full,
            # but actually it's always full in mmdetection.
            self.temp_fisher_info[module][:grad_input[0].size(0)] += \
                compute_fisher(feature, grad_feature, layer_name)

    def construct_outchannel_masks(self):
        """Register the `input_mask` of one conv to it's nearest ancestor conv,
        and name it as `out_mask`, which means the actually number of output
        feature map after pruning."""

        for conv, name in self.conv_names.items():
            assigned = False
            for m, ancest in self.conv2ancest.items():
                if conv in ancest:
                    conv.out_mask = m.in_mask
                    assigned = True
                    break
            # may be the last conv of network
            if not assigned:
                conv.register_buffer(
                    'out_mask',
                    torch.ones((1, conv.out_channels, 1, 1),
                               dtype=torch.bool).cuda())

        for bn, name in self.bn_names.items():
            conv_module = self.bn2ancest[bn][0]
            bn.out_mask = conv_module.out_mask

    def make_groups(self):
        """The modules (convolutions and BNs) connected to the same conv need
        to change the channels simultaneously when pruning.

        This function divides all modules into different groups according to
        the connections.
        """

        idx = -1
        groups, groups_ancest = {}, {}
        for module, name in reversed(self.conv_names.items()):
            added = False
            for group in groups:
                module_ancest = set(self.conv2ancest[module])
                group_ancest = set(groups_ancest[group])
                if len(module_ancest.intersection(group_ancest)) > 0:
                    groups[group].append(module)
                    groups_ancest[group] = list(
                        module_ancest.union(group_ancest))
                    added = True
                    break
            if not added:
                idx += 1
                groups[idx] = [module]
                groups_ancest[idx] = self.conv2ancest[module]
        # key is the ids the group, and value contains all conv
        # of this group
        self.groups = {}
        # key is the ids the group, and value contains all nearest
        # ancestor of this group
        self.ancest = {}
        idx = 0
        # filter the group with only one conv
        for group in groups:
            modules = groups[group]
            if len(modules) > 1:
                self.groups[idx] = modules
                self.ancest[idx] = groups_ancest[group]
                idx += 1
        # the conv's name in same group, just for debug
        # TODO remove this
        self.conv_names_group = [[item.name for item in v]
                                 for idx, v in self.groups.items()]

    def set_group_masks(self, model):
        """the modules(convolutions and BN) connect to same convolutions need
        change the out channels at same time when pruning, divide the modules
        into different groups according to the connection.

        Args:
            model(nn.Module): the model contains all modules
        """

        # TODO: support two stage model
        net_names = ['backbone', 'neck', 'bbox_head']

        def onestage_wrapper(x):
            out = 0.
            for levels in x:
                out += sum([level.sum() for level in levels])
            return out

        inputs = torch.zeros(1, 3, 256, 256).cuda()

        for name in net_names:
            net = dict(model.named_modules())[f'module.{name}']
            inputs = net(inputs)
        loss = onestage_wrapper(inputs)
        # self.conv2ancest is a dict, key is all conv instance in
        # model, value is a list which contains all nearest  ancestor
        # conv, and self.conv_link describe the connection using module
        # name
        self.conv2ancest = self.find_module_ancestors(loss, NON_PASS)
        self.conv_link = {
            k.name: [item.name for item in v]
            for k, v in self.conv2ancest.items()
        }
        # key is bn module, and value is a list of nearest
        # ancestor convs
        self.bn2ancest = self.find_module_ancestors(loss, BN)
        loss.sum().backward()
        self.make_groups()

        # list contains all the convs which are contained in a
        # group (if more the one conv has same ancestor,
        # they will be in same group)
        self.group_modules = []
        for group in self.groups:
            self.group_modules.extend(self.groups[group])

    def find_module_ancestors(self, loss, pattern):
        """find the nearest Convolution of the module
        matching the pattern
        Args:
            loss(Tensor): the output of the network
            pattern(Tuple[str]): the pattern name

        Returns:
            dict: the key is the module match the pattern(Conv or Fc),
             and value is the list of it's nearest ancestor Convolution
        """

        # key is the op (indicate a Conv or Fc) and value is a list
        # contains all the nearest ops (indicate a Conv or Fc)
        op2parents = {}
        self.traverse(loss.grad_fn, op2parents, pattern)

        var2module = {}
        if pattern is BN:
            module_names = self.bn_names
        else:
            module_names = self.conv_names
        for module, name in module_names.items():
            var2module[id(module.weight)] = module

        # same module may appear several times in computing graph,
        # so same module can correspond to several op, for example,
        # different feature pyramid level share heads.
        # op2module select one op as the flag of module.
        op2module = {}
        for op, parents in op2parents.items():
            # TODO bfs to get variable
            var_id = id(op.next_functions[1][0].variable)
            module = var2module[var_id]
            exist = False
            # may several op link to same module
            for temp_op, temp_module in op2module.items():
                # temp_op(has visited in loop) and op
                # link to same module, so their should share
                # all parents, so we need extend the value of
                # op to value of temp_op
                if temp_module is module:
                    op2parents[temp_op].extend(op2parents[op])
                    exist = True
                    break
            if not exist:
                op2module[op] = module
        if not hasattr(self, 'conv_module'):
            # save for find bn's ancestor convolutions
            self.conv_module = op2module
        return {
            module: [
                self.conv_module[parent] for parent in op2parents[op]
                if parent in self.conv_module
            ]
            for op, module in op2module.items()
        }

    def traverse(self, op, op2parents, pattern=NON_PASS):
        """to get a dict which can describe the computer Graph,

        Args:
            op (grad_fn): as a root of DFS
            op2parents (dict): key is the grad_fn match the patter,and
                value is first grad_fn match NON_PASS when DFS from Key
            pattern (Tuple[str]): the patter of grad_fn to match
        """

        if op is not None:
            parents = op.next_functions
            if parents is not None:
                if self.match(op, pattern):
                    op2parents[op] = self.dfs(parents[0][0], [])
                for parent in parents:
                    parent = parent[0]
                    if parent not in op2parents:
                        self.traverse(parent, op2parents, pattern)

    def dfs(self, op, visited):
        """DFS from a op,return all op when find a op match the patter
        NON_PASS.

        Args:
            op (grad_fn): the root of DFS
            visited (list[grad_fn]): contains all op has been visited

        Returns:
            list : all the ops  match the patter NON_PASS
        """

        ret = []
        if op is not None:
            visited.append(op)
            if self.match(op, NON_PASS):
                return [op]
            parents = op.next_functions
            if parents is not None:
                for parent in parents:
                    parent = parent[0]
                    if parent not in visited:
                        ret.extend(self.dfs(parent, visited))
        return ret

    def match(self, op, op_to_match):
        """Match an operation to a group of operations; In pytorch graph, there
        may be an additional '0' or '1' (e.g. Addbackward1) after the ops
        listed above.

        Args:
            op (grad_fn): the grad_fn to match the pattern
            op_to_match (list[str]): the pattern need to match

        Returns:
            bool: return True when match the pattern else False
        """

        for to_match in op_to_match:
            if re.match(to_match + '[0-1]?$', type(op).__name__):
                return True
        return False


def add_pruning_attrs(module, pruning=False):
    """When module is conv, add `finetune` attribute, register `mask` buffer
    and change the origin `forward` function. When module is BN, add `out_mask`
    attribute to module.

    Args:
        conv (nn.Conv2d):  The instance of `torch.nn.Conv2d`
        pruning (bool): Indicating the state of model which
            will make conv's forward behave differently.
    """
    # TODO: mask  change to bool
    if type(module).__name__ == 'Conv2d':
        module.register_buffer(
            'in_mask', module.weight.new_ones((1, module.in_channels, 1, 1), ))
        module.register_buffer(
            'out_mask', module.weight.new_ones(
                (1, module.out_channels, 1, 1), ))
        module.finetune = not pruning

        def modified_forward(self, feature):
            if not self.finetune:
                feature = feature * self.in_mask
            return F.conv2d(feature, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)

        module.forward = MethodType(modified_forward, module)
    if 'BatchNorm' in type(module).__name__:
        module.register_buffer(
            'out_mask', module.weight.new_ones(
                (1, len(module.weight), 1, 1), ))


def deploy_pruning(model):
    """To speed up the finetune process, We change the shape of parameter
    according to the `in_mask` and `out_mask` in it."""

    for name, module in model.named_modules():
        if type(module).__name__ == 'Conv2d':
            module.finetune = True
            requires_grad = module.weight.requires_grad
            out_mask = module.out_mask.view(-1).bool()

            if hasattr(module, 'bias') and module.bias is not None:
                module.bias = nn.Parameter(module.bias.data[out_mask])
                module.bias.requires_grad = requires_grad
            temp_weight = module.weight.data[out_mask]
            in_mask = module.in_mask.view(-1).bool()
            module.weight = nn.Parameter(temp_weight[:, in_mask].data)
            module.weight.requires_grad = requires_grad

        elif 'BatchNorm2d' in type(module).__name__:
            out_mask = module.out_mask.view(-1).bool()
            requires_grad = module.weight.requires_grad
            module.weight = nn.Parameter(module.weight.data[out_mask].data)
            module.bias = nn.Parameter(module.bias.data[out_mask].data)
            module.running_mean = module.running_mean[out_mask]
            module.running_var = module.running_var[out_mask]
            module.weight.requires_grad = requires_grad
            module.bias.requires_grad = requires_grad
