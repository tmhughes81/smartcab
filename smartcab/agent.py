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
        # Initialize any additional variables here
        self.alpha = 0.9 # Learning Rate
        self.gamma = 0.1 # Discount Rate
        self.epsilon = 0 # Exploration Rate, as an integer percent from 0-99
        
        # Success measure
        self.success = 0
        self.no_penalty = True
        
        # For Q Updates
        self.prev_state = None
        self.prev_action = None
        self.prev_reward = None
        
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
        # Prepare for a new trip; reset any variables here, if required
        # Reset our no_penalty flag for success tracking
        self.no_penalty = True
        self.prev_state = None
    
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
        
        # Learn policy based on previous state, previous action, previous reward, and current state, so long as we have a prev_state
        if self.prev_state != None:
            self.Q_hat(self.prev_state, self.prev_action, self.prev_reward, self.state)
        
        # Select action according to your policy.  Set optimal to false to explore
        action = self.policy(self.state, optimal=False)

        # Execute action and get reward
        reward = self.env.act(self, action)
        
        # Now that we've taken an action, record everything from this action to pass to the next update
        self.prev_action = action
        self.prev_reward = reward
        self.prev_state = self.state
        
        
        # If we were penalized, we don't want to record this run as a success
        if reward < 0:
            self.no_penalty = False
        
        # Record success if we have triggered no penalties and reached our destination
        if self.no_penalty and self.env.agent_states[self]['location'] == self.env.agent_states[self]['destination']:
            self.success += 1
        
        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]
        # Uncomment this line if you want to get success count updates; bit spammy
        #print "LearningAgent.update(): successes = {}".format(self.success)
    
    def policy(self, s, optimal=True):
        # Consider all actions we might use
        actions = [None, 'forward', 'left', 'right']

        """ Returns estimated best action based on a state 's' """
        # Handle exploration first.  If we randomly come in below epsilon, just
        # go a random direction.  Only do this if 'optimal' is set to 'false' 
        if not optimal and random.randint(0, 99) < self.epsilon:
            return random.choice(actions)
        
        # Convert our state into a tuple for lookup purposes
        s = self.tuple_state(s)
        
        # Previous version had a threshold counter loop I was using; as per
        # Reviewer #1, I've used his list comprehension version of the code,
        # as it is less complicated than what I was doing.  Credit should
        # definitely be given to reviewer 1 for this.
        
        # get the maximum q-value for the state
        max_qval = max([self.Q_table[(s, a)] for a in actions])
        
        # find the actions that yield the maximum q-value
        best_actions = [a for a in actions if self.Q_table[(s, a)] == max_qval]
        
        # randomly pick one of the best actions
        action = random.choice(best_actions)
        
        return action
    
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
    sim = Simulator(e, update_delay=0.0, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line

if __name__ == '__main__':
    run()
