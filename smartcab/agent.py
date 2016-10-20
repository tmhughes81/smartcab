import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.alpha = 0.5 # Learning Rate
        self.gamma = 0.5 # Discount Rate
        self.epsilon = 0.02 # Exploration Rate
        
        # Table to store expected rewards for various states
        self.Q_table = {}
        
        # Valid values for our state data
        waypoint = ["forward", "left", "right"]
        light = ['red', 'green']
        oncoming = [None, "forward", "left", "right"]
        left = [None, "forward", "left", "right"]
        right = [None, "forward", "left", "right"]
        action = [None, "forward", "left", "right"]
        
        # Initialize our Q table with zeroes, values will update as it learns
        for w in waypoint:
            for li in light:
                for o in oncoming:
                    for le in left:
                        for r in right:
                            for a in action:
                                self.Q_table[((w, li, o, le, r), a)] = 0
    
    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
    
    def set_state(self, inputs):
        """ Takes a set of inputs and defines a state that will be used by our
        algorithm later for policy decision making. """
        
        self.state = {'waypoint': self.next_waypoint,
                      'light': inputs['light'],
                      'oncoming': inputs['oncoming'],
                      'left': inputs['left'],
                      'right': inputs['right']
                       }
        return
    
    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # Update state
        self.set_state(inputs)
        
        # We're going to need our start state and end-state for our Q update
        start_state = self.state
        
        # Select action according to your policy
        action = self.policy(start_state)

        # Execute action and get reward
        reward = self.env.act(self, action)
        
        # Now that we've taken an action, we can get our end_state
        self.set_state(inputs)
        end_state = self.state
        
        # Learn policy based on start state, action, reward, end_state
        self.Q_hat(start_state, action, reward, end_state)

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]
    
    def policy(self, s):
        """ Returns estimated best action based on a state 's' """
        # We don't need 'None' here because we initialize our viable actions with
        # the values for 'None'
        actions = ['forward', 'left', 'right']

        # Convert our state into a tuple for lookup purposes
        s = self.tuple_state(s)
        
        # Initialize our viable actions by assuming no action is the best action
        # until we see a better option.
        threshold = self.Q_table[(s, None)]
        viable_actions = [None]
        
        # Go through all options for actions, and check if they exceed our current
        # threshold.  If they are the same as the threshold, add it to the list of
        # viable actions.  If the action exceeds the threshold, eliminate all
        # prior viable actions, and use this as our new threshold.
        for a in actions:
            if self.Q_table[(s, a)] == threshold:
                viable_actions.append(a)
            elif self.Q_table[(s, a)] > threshold:
                viable_actions = [a]
                threshold = self.Q_table[(s,a)]
        
        # Take a random action from our list of viable actions.  This list will
        # only be longer than 1 in the event that there are multiple actions
        # in this state with the same estimated utility
        return viable_actions[random.randint(0, (len(viable_actions)-1))]
    
    def Q_hat(self, s, a, r, s_prime):
        """ This is our learning algorithm.  It takes a state 's', action 'a',
        reward 'r', and a new state 's_prime', and updates the estimated utility
        of taking action 'a' from state 's' """
        
        # Get the optimal action for the next state
        a_prime = self.policy(s_prime)
        
        # Convert our state inputs into tuples for table lookup
        s = self.tuple_state(s)
        s_prime = self.tuple_state(s_prime)
        
        # Update the Q_table with what we've learned
        self.Q_table[(s, a)] = (1-self.alpha)*self.Q_table[(s, a)]+self.alpha*(r+self.gamma*self.Q_table[(s_prime, a_prime)])
        
        return None
    
    def tuple_state(self, state):
        """ We sometimes need our state in a tuple format for looking up values
        in our Q_table dict """
        tupled_state = (state["waypoint"],
                        state["light"],
                        state["oncoming"],
                        state["left"],
                        state["right"])
        
        return tupled_state

def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.5, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
