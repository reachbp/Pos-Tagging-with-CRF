import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from pystruct.models import ChainCRF
from pystruct.learners import OneSlackSSVM

X, y = [], []
current_file = os.path.abspath(os.path.dirname(__file__))

for f in ["Data/train-%d.txt" % i for i in range(1, 5001)]:             # Iterate over all the training sample files
    csv_filename = os.path.join(current_file, f)
    data = pd.read_csv(csv_filename, header=None, quoting=3)                       # Read each training sample file into 'data' variable
    labels = data[1]                                                    # Extract 'tag' field into 'labels'
    features = data.values[:, 2:].astype(np.int)                        # Extract feature fields into 'features'
    for f_idx in range(len(features)):                                  # Adjust features starting at 1 to start at 0
      f1 = features[f_idx]
      features[f_idx] = [f1[0]-1, f1[1], f1[2], f1[3]-1, f1[4]-1]
    y.append(labels.values - 1)                                         # Adjust labels to lie in {0,...,9}, and add to 'y'
    X.append(features)                                                  # Add feature vector to 'X'

# See: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
# [Note: if you get an error on the below line, it may be because you need to upgrade scikit-learn]
encoder = OneHotEncoder(n_values=[1,2,2,201,201],sparse=False).fit(np.vstack(X))                 
                                                                        # Represent features using one-of-K scheme: If a feature can take value in 
X_encoded = [encoder.transform(x) for x in X]                           # {0,...,K}, then introduce K binary features such that the value of only 
                                                                        # the i^th binary feature is non-zero when the feature takes value 'i'.
                                                                        # n_values specifies the number of states each feature can take.

                                                                        
X_small, y_small = X_encoded[:100], y[:100]                             # Pick the first 100 samples from the encoded training set.


# See: http://pystruct.github.io/generated/pystruct.learners.OneSlackSSVM.html
# See: http://pystruct.github.io/generated/pystruct.models.ChainCRF.html
# Rest of documentation can be found here: http://pystruct.github.io/references.html
ssvm = OneSlackSSVM(ChainCRF(n_states=10,inference_method='max-product',directed=True), max_iter=200,C=1)
                                                                        # Construct a directed ChainCRF with 10 states for each variable, 
                                                                        # and pass this CRF to OneSlackSSVM constructor to create an object 'ssvm'
ssvm.fit(X_small, y_small)                                              # Learn Structured SVM using X_small and y_small
weights = ssvm.w                                                        # Store learnt weights in 'weights'
print ssvm.score(X_small, y_small)                                      # Evaluate training accuracy on X_small, y_small
print ssvm.predict(X_small)                                             # Get predicted labels on X_small using the learnt model
