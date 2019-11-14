import sys
sys.path.insert(0,'/Users/neelguha/Dropbox/NeelResearch/fairness/code/tensorflow_constrained_optimization/')
import tensorflow as tf
import tensorflow_constrained_optimization as tfco


def get_placeholder_name(attribute):
    str(hash(attribute))+ "_placeholder"

class LinearModel(object):
    def __init__(self, feature_names, protected_features, label_column, constraints, tpr_max_diff=0, random_seed=123):
        """ Initializes linear model 
        
        Args:
            feature_names (list): number of input features 
            protected_features (list): list of protected features
            label_column (string): name of column with label
            constraints (list): list of groups for which we want to constrain performance
            tpr_max_diff (float): maximum tolerated difference in TPR (default = 0.0)
            random_seed (int): random seed (default = 123)
        """
        
        tf.random.set_random_seed(random_seed)
        self.tpr_max_diff = tpr_max_diff
        self.feature_names = feature_names 
        self.protected_features = protected_features
        self.label_column = label_column
        self.constraints = constraints

        self.features_placeholder = tf.placeholder(
            tf.float32, shape=(None, len(self.feature_names)), name='features_placeholder')
        self.labels_placeholder = tf.placeholder(
            tf.float32, shape=(None, 1), name='labels_placeholder')
        
        # add placeholder for each protected attribute
        self.protected_placeholders_dict = {attribute: tf.placeholder(tf.float32, shape=(None, 1), name=get_placeholder_name(attribute)) for attribute in protected_features}
        self.protected_placeholders = [self.protected_placeholders_dict[attribute] for attribute in protected_features]
        
        # We use a linear model.
        self.predictions_tensor = tf.layers.dense(inputs=self.features_placeholder, units=1, activation=None)


    def build_train_op(self,
                       learning_rate,
                       unconstrained=False):
        ctx = tfco.rate_context(self.predictions_tensor, self.labels_placeholder)
        positive_slice = ctx.subset(self.labels_placeholder > 0) 
        overall_tpr = tfco.positive_prediction_rate(positive_slice)
        constraints = []

        # add constraints
        if not unconstrained:

            for constraint in self.constraints:
                
                print(constraint)
                if len(constraint) == 1:
                    placeholder = self.protected_placeholders_dict[constraint[0]]
                    slice_tpr = tfco.positive_prediction_rate(ctx.subset((placeholder > 0) & (self.labels_placeholder > 0)))
                elif len(constraint) == 2:
                    placeholder0 = self.protected_placeholders_dict[constraint[0]]
                    placeholder1 = self.protected_placeholders_dict[constraint[1]]
                    slice_tpr = tfco.positive_prediction_rate(ctx.subset((placeholder0 > 0) & (placeholder1 > 0) & (self.labels_placeholder > 0)))

                constraints.append(slice_tpr >= overall_tpr - self.tpr_max_diff)

        mp = tfco.RateMinimizationProblem(tfco.error_rate(ctx), constraints)
        opt = tfco.ProxyLagrangianOptimizer(tf.train.AdamOptimizer(learning_rate))
        self.train_op = opt.minimize(minimization_problem=mp)
        return self.train_op
  
    def feed_dict_helper(self, dataframe):
        feed_dict = {self.features_placeholder:
                  dataframe[self.feature_names],
              self.labels_placeholder:
                  dataframe[[self.label_column]],}
        for i, protected_attribute in enumerate(self.protected_features):
            feed_dict[self.protected_placeholders[i]] = dataframe[[protected_attribute]]
        return feed_dict