import torch
import random
import copy

def get_weights(net):
    """ Extract parameters from net, and return a list of tensors"""
    return [p.data for p in net.parameters()]

def set_weights(net, weights, directions=None, step=None):
    """
        Overwrite the network's weights with a specified list of tensors
        or change weights along directions with a step size.
    """
    if directions is None:
        # You cannot specify a step length without a direction.
        for (p, w) in zip(net.parameters(), weights):
            p.data.copy_(w.type(type(p.data)))
    else:
        assert step is not None, 'If a direction is specified then step must be specified as well'

        if len(directions) == 2:
            dx = directions[0]
            dy = directions[1]
            changes = [d0*step[0] + d1*step[1] for (d0, d1) in zip(dx, dy)]
        else:
            changes = [d*step for d in directions[0]]

        for (p, w, d) in zip(net.parameters(), weights, changes):
            p.data = w + torch.Tensor(d).type(type(w))

            
def get_random_weights(weights):
    """
        Produce a random direction that is a list of random Gaussian tensors
        with the same shape as the network's weights, so one direction entry per weight.
    """
    return [torch.randn(w.size()) for w in weights]


def get_random_states(states):
    """
        Produce a random direction that is a list of random Gaussian tensors
        with the same shape as the network's state_dict(), so one direction entry
        per weight, including BN's running_mean/var.
    """
    return [torch.randn(w.size()) for k, w in states.items()]


def get_diff_weights(weights, weights2):
    """ Produce a direction from 'weights' to 'weights2'."""
    return [w2 - w for (w, w2) in zip(weights, weights2)]


def get_diff_states(states, states2):
    """ Produce a direction from 'states' to 'states2'."""
    return [v2 - v for (k, v), (k2, v2) in zip(states.items(), states2.items())]

def add_weights_eps(weights,eps=1e-3,rand=True,isuniform=False):
    if(rand):
        if(isuniform):
            return [w +random.uniform(-eps,eps) for w in weights]
        else:
            return [w +random.gauss(0,eps) for w in weights]
    else:
        return [w + eps for w in weights]

def add_net_eps(net,eps=1e-3,rand=False, directions=None, step=None):
    w=add_weights_eps(get_weights(net))
    nnet= copy.deepcopy(net)
    set_weights(nnet,w,directions,step)
    return nnet


class Netweight():
        def __init__(self,net:torch.nn):
            self.net=net
            
        def get(self):
            """ Extract parameters from net, and return a list of tensors"""
            return [p.data for p in self.net.parameters()]
        
        def set(self, weights, directions=None, step=None):
            """
                Overwrite the network's weights with a specified list of tensors
                or change weights along directions with a step size.
            """
            if directions is None:
                # You cannot specify a step length without a direction.
                for (p, w) in zip(self.net.parameters(), weights):
                    p.data.copy_(w.type(type(p.data)))
            else:
                assert step is not None, 'If a direction is specified then step must be specified as well'

                if len(directions) == 2:
                    dx,dy = directions
                    changes = [d0*step[0] + d1*step[1] for (d0, d1) in zip(dx, dy)]
                else:
                    changes = [d*step for d in directions[0]]

                for (p, w, d) in zip(self.net.parameters(), weights, changes):
                    p.data = w + torch.Tensor(d).type(type(w))

        def _add_weights_eps(weights,eps=1e-3,rand=True,isuniform=False):
            if(rand):
                if(isuniform):
                    return [w +random.uniform(-eps,eps) for w in weights]
                else:
                    return [w +random.gauss(0,eps) for w in weights]
            else:
                return [w + eps for w in weights]
            
        def add_eps(self,eps=1e-3,rand=False, directions=None, step=None):
            self.set(self._add_weights_eps(self.get()))

        def copy_with_eps(self,eps=1e-3,rand=False, directions=None, step=None):
            nnet= copy.deepcopy(self.net)
            w=self._add_weights_eps(self.get())
            set_weights(nnet,w,directions,step)
            return Netweight(nnet)
        