from logging import getLogger
from sklearn import linear_model
logger = getLogger(__name__)

import numpy as np
import tensorflow as tf


class NAF(object):
  def __init__(self, sess,
               env, strategy, pred_network, target_network, stat,
               discount, batch_size, learning_rate,
               max_steps, update_repeat, max_episodes, im_rollouts=None):
    self.sess = sess
    self.env = env
    self.IM_ROLLOUT=im_rollouts
    self.strategy = strategy
    self.pred_network = pred_network
    self.target_network = target_network
    self.stat = stat

    self.discount = discount
    self.batch_size = batch_size
    self.learning_rate = learning_rate
    self.action_size = env.action_space.shape[0]

    self.max_steps = max_steps
    self.update_repeat = update_repeat
    self.max_episodes = max_episodes

    self.prestates = []
    self.actions = []
    self.rewards = []
    self.poststates = []
    self.terminals = []
    self.f_prestates = []
    self.f_actions = []
    self.f_rewards = []
    self.f_poststates = []
    self.f_terminals = []
    self.ir_model =  linear_model.LinearRegression()
    self.ir_model_rew =  linear_model.LinearRegression()


    with tf.name_scope('optimizer'):
      self.target_y = tf.placeholder(tf.float32, [None], name='target_y')
      self.loss = tf.reduce_mean(tf.squared_difference(self.target_y, tf.squeeze(self.pred_network.Q)), name='loss')

      self.optim = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
    self.stat.load_model()
    self.target_network.hard_copy_from(self.pred_network)

  def run(self, monitor=False, display=False, is_train=True, first=False, n_episodes=20):

    rewards = []
    if is_train:
        max_episodes = self.max_episodes
    else:
        max_episodes=n_episodes
    for self.idx_episode in range(max_episodes):
      state = self.env.reset()
      for t in range(0, self.max_steps):
        if display: self.env.render()

        # 1. predict
        if state.shape[0] == 1:
            state=state[0]
        action = self.predict(state)
        # 2. step
        self.prestates.append(state)
        input_action = action
        if action.shape == (1,):
            input_action = action.reshape(1,1)
        state, reward, terminal, _ = self.env.step(input_action)
        reward = reward[0]
        self.rewards.append(reward)
        self.actions.append(action)
        terminal = terminal[0] 
        if state.shape[0] == 1:
            state=state[0]
        self.poststates.append(state)
        terminal = True if t == self.max_steps - 1 else terminal
        # 3. perceive
        if is_train:
          q, v, a, l = self.perceive(state, reward, action, terminal)

          if self.stat:
            self.stat.on_step(action, reward, terminal, q, v, a, l)
        if terminal:
          self.strategy.reset()
          break
       
      rewards.append(reward)
    #Imaginary rollouts
    if self.IM_ROLLOUT:
        self.fit_dynamics()
        self.im_rollouts(state)    
    
    success_rate = np.mean(np.array(rewards) >= 0)


    return None, success_rate
  def im_rollouts(self, initial_state, n_ir_episodes=5):
    state = initial_state
    for i in range(n_ir_episodes):
      for t in range(self.max_steps):
        # 1. predict
        action = self.predict(state)

        # 2. step
        input_action = action
        if action.shape == (1,):
          input_action = action.reshape(1,1)
        try:
            poststate, reward = self.predict_fitted(state, input_action)
            self.f_prestates.append(state)
            self.f_poststates.append(poststate)
            self.f_rewards.append(reward)
            self.f_actions.append(action)
        except:
            print("Not enough information for IM rollout")
        terminal = True if t == self.max_steps - 1 else False
        if terminal:
          self.strategy.reset()
          break

  def run2(self, monitor=False, display=False, is_train=True):
    target_y = tf.placeholder(tf.float32, [None], name='target_y')
    loss = tf.reduce_mean(tf.squared_difference(target_y, tf.squeeze(self.pred_network.Q)), name='loss')

    optim = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

    self.stat.load_model()
    self.target_network.hard_copy_from(self.pred_network)

    # replay memory
    prestates = []
    actions = []
    rewards = []
    poststates = []
    terminals = []
    f_prestates = []
    f_actions = []
    f_rewards = []
    f_poststates = []
    f_terminals = []

    # the main learning loop
    total_reward = 0
    for i_episode in range(self.max_episodes):
      observation = self.env.reset()
      episode_reward = 0

      for t in range(self.max_steps):
        if display:
          self.env.render()

        # predict the mean action from current observation
        x_ = np.array([observation])
        u_ = self.pred_network.mu.eval({self.pred_network.x: x_})[0]

        action = u_ + np.random.randn(1) / (i_episode + 1)

        prestates.append(observation)
        actions.append(action)

        observation, reward, done, info = self.env.step(action)
        episode_reward += reward

        rewards.append(reward); poststates.append(observation); terminals.append(done)
        if done:
          break

      print("average loss:", loss_/k)
      print("Episode {} finished after {} timesteps, reward {}".format(i_episode + 1, t + 1, episode_reward))
      total_reward += episode_reward



    


    print("Average reward per episode {}".format(total_reward / self.episodes))

  def predict(self, state):
    if len(state.shape) == 2:
       u = self.pred_network.predict(state)
    else:
       u = self.pred_network.predict([state])[0]

    return self.strategy.add_noise(u, {'idx_episode': self.idx_episode})

  def predict_fitted(self, state, action):
      state= self.ir_model.predict(np.hstack([state, action]).reshape(1,-1))[0]
      reward =self.ir_model_rew.predict(np.hstack([state, action]).reshape(1,-1))[0] 
      return state, reward

  def fit_dynamics(self):
    #update ir model
    try:
      self.ir_model_rew.fit(np.hstack([self.prestates, self.actions]),self.rewards)
    except:
      import ipdb; ipdb.set_trace()
    self.ir_model.fit(np.hstack([self.prestates, self.actions]),self.poststates)



  def perceive(self, state, reward, action, terminal):
    if len(self.f_prestates) > 140 and self.IM_ROLLOUT:
        self.q_learning_minibatch(imaginary=True)
    return self.q_learning_minibatch()

  def q_learning_minibatch(self, imaginary=False):
    q_list = []
    v_list = []
    a_list = []
    l_list = []
    update_repeat = self.update_repeat 
    if imaginary:
        update_repeat = 3
    for iteration in range(update_repeat):
      if not imaginary:
        if len(self.rewards) >= self.batch_size:
          indexes = np.random.choice(len(self.rewards), size=self.batch_size)
        else:
          indexes = np.arange(len(self.rewards))
      else:
        if len(self.f_rewards) >= self.batch_size:
          indexes = np.random.choice(len(self.f_rewards), size=self.batch_size)
        else:
          indexes = np.arange(len(self.rewards))

      if not imaginary:
        x_t = np.array(self.prestates)[indexes]
        x_t_plus_1 = np.array(self.poststates)[indexes]
        r_t = np.array(self.rewards)[indexes]
        u_t = np.array(self.actions)[indexes]
      else:
        x_t = np.array(self.f_prestates)[indexes]
        x_t_plus_1 = np.array(self.f_poststates)[indexes]
        r_t = np.array(self.f_rewards)[indexes]
        u_t = np.array(self.f_actions)[indexes]


      v = self.target_network.predict_v(x_t_plus_1, u_t)
      target_y = self.discount * np.squeeze(v) + r_t

      _, l, q, v, a = self.sess.run([
        self.optim, self.loss,
        self.pred_network.Q, self.pred_network.V, self.pred_network.A,
      ], {
        self.target_y: target_y,
        self.pred_network.x: x_t,
        self.pred_network.u: u_t,
        self.pred_network.is_train: True,
      })

      q_list.extend(q)
      v_list.extend(v)
      a_list.extend(a)
      l_list.append(l)

      self.target_network.soft_update_from(self.pred_network)

      logger.debug("q: %s, v: %s, a: %s, l: %s" \
        % (np.mean(q), np.mean(v), np.mean(a), np.mean(l)))

    return np.sum(q_list), np.sum(v_list), np.sum(a_list), np.sum(l_list)
    
#didn't want to deal with import issues
import datetime
import dateutil.tz

def get_timestamp():
  now = datetime.datetime.now(dateutil.tz.tzlocal())
  return now.strftime('%Y_%m_%d_%H_%M_%S')
