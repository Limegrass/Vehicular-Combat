
class QFunc:

    LEARNING_RATE = 0.5 #change this
    DISCOUNT_FACTOR = 0.5 #change this
    
    def __init__(self, state, action, features, reward):
        
        self.state = state
        self.action = action
        self.features = features

        self.weights = []

        for i in range(len(features)):
            self.weights[i] = random.uniform(0.0, 1.0)

        self.immediate_rew = reward
        self.value = q_val(self.weights, self.features)

    def q_val(weights, features):

        val = 0;
        
        for i in range(len(features)):
            val += (weights[i])*(features[i])

        return val

    def find_max_a():
        possible_actions = []

        max = 0
        #find max a if you try every action at current state
        for i in possible_actions:
            #compute features
            #call qval()
            #update max if necessary
            pass
        
    def update_weight(w, f, state):

        #updates one w using the formula:
        #w <- w + alpha( R(s) + delta*max(a')*Q(s',a') - Q(s,a) )x
        new_w = w + LEARNING_RATE * ( self.value + (DISCOUNT_FACTOR*find_max_a()) - self.value))*f

        pass

    #features will somehow need to be a list of functions
    def update_weights(state, action, features, reward):

        new_qval = 0;
        self.value += reward

        self.state = state
        self.action = action

        #update each weight and add (weight * feature) to the current q value
        for i in range(len(self.weights)):
            w = update_weight(self.weights[i], self.features[i])
            new_qval += w * self.features[i]

        self.value = new_qval
