import gym
import tensorflow as tf

from utils import distributions
from baselines.common.mpi_running_mean_std import RunningMeanStd

from utils import scope_utils
from utils import layer_utils
from utils import array_utils
from utils import function_utils


class Policy(object):
    def __init__(self, name, *args, **kargs):
        with tf.variable_scope(name) as scope:
            self._scope = scope
            self._initialize_policy(*args, **kargs)

    def act(self, stochastic, observation):
        raise NotImplemented()

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self._scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self._scope)

    def get_initial_state(self):
        return []


class MlpPolicy(Policy):
    def _initialize_policy(self,
                           ob_space, ac_space,
                           num_units, num_layers,
                           gaussian_fixed_var=True):

        if not isinstance(ob_space, gym.spaces.Box):
            raise TypeError("expected `ob_space` to be ",
                            "gym.spaces.Box, found ", type(ob_space))

        with tf.variable_scope("obfilter"):
            ob_rms = RunningMeanStd(shape=ob_space.shape)
        sequence_length = None
        ob = scope_utils.get_placeholder(
            name="ob", dtype=tf.float32,
            shape=[sequence_length] + list(ob_space.shape))
        obz = tf.clip_by_value(
            (ob - ob_rms.mean) / ob_rms.std,
            clip_value_min=-5.0, clip_value_max=5.0)


        last_out_vf = obz
        last_out_pol = obz
        for i in range(num_layers):
            last_out_vf = layer_utils.dense(
                inputs=last_out_vf,
                units=num_units,
                activation=tf.nn.tanh,
                name="vffc%i" % (i + 1),
                kernel_initializer=layer_utils.normc_initializer(1.0))
            last_out_pol = layer_utils.dense(
                inputs=last_out_pol,
                units=num_units,
                activation=tf.nn.tanh,
                name="polfc%i" % (i + 1),
                weight_init=layer_utils.normc_initializer(1.0))

        vpred = layer_utils.dense(
            inputs=last_out_vf, units=1, name="vffinal",
            kernel_initializer=layer_utils.normc_initializer(1.0))
        vpred = vpred[:, 0]

        # this should create a DiagGaussianPd
        pdtype = distributions.make_pdtype(ac_space)
        if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
            mean = layer_utils.dense(
                inputs=last_out_pol,
                units=pdtype.param_shape()[0] // 2,
                name="polfinal",
                kernel_initializer=layer_utils.normc_initializer(0.01))

            logstd = tf.get_variable(name="logstd",
                shape=[1, pdtype.param_shape()[0] // 2],
                initializer=tf.zeros_initializer())

            pdparam = array_utils.concatenate(
                [mean, mean * 0.0 + logstd], axis=1)
        else:
            pdparam = layer_utils.dense(
                inputs=last_out_pol,
                units=pdtype.param_shape()[0],
                name="polfinal",
                kernel_initializer=layer_utils.normc_initializer(0.01))

        # create the distribution
        prob_dist = pdtype.pdfromflat(pdparam)
        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        action = function_utils.switch(
            condition=stochastic,
            then_expression=prob_dist.sample(),
            else_expression=prob_dist.mode())
        action = function_utils.function(
            inputs=[stochastic, ob],
            outputs=[action, self.vpred])


        self._state_in = []
        self._state_out = []

        self._vpred = vpred
        self._ob_rms = ob_rms

        self._pdtype = pdtype
        self._prob_dist = prob_dist
        self._action = action

    def act(self, stochastic, observation):
        new_action, new_vpred = self._action(stochastic, observation[None])
        


