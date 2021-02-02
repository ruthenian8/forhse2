```python
import numpy as np
import pandas as pd
```


```python
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
```

## Task 1


```python
class Tree:
  '''Create a binary tree; keyword-only arguments `data`, `left`, `right`.

  Examples:
    l1 = Tree.leaf("leaf1")
    l2 = Tree.leaf("leaf2")
    tree = Tree(data="root", left=l1, right=Tree(right=l2))
  '''

  def leaf(data):
    '''Create a leaf tree
    '''
    return Tree(data=data)

  # pretty-print trees
  def __repr__(self):
    if self.is_leaf():
      return "Leaf(%r)" % self.data
    else:
      return "Tree(%r) { left = %r, right = %r }" % (self.data, self.left, self.right) 

  # all arguments after `*` are *keyword-only*!
  def __init__(self, *, data = None, left = None, right = None):
    self.data = data
    self.left = left
    self.right = right

  def is_leaf(self):
    '''Check if this tree is a leaf tree
    '''
    return self.left == None and self.right == None

  def children(self):
    '''List of child subtrees
    '''
    return [x for x in [self.left, self.right] if x]

  def depth(self):
    '''Compute the depth of a tree
    A leaf is depth-1, and a child is one deeper than the parent.
    '''
    return max([x.depth() for x in self.children()], default=0) + 1
```


```python
l1 = Tree.leaf('like')
l2 = Tree.leaf('like')
l3 = Tree.leaf('like')
l4 = Tree.leaf('nah')
l5 = Tree.leaf('nah')
```


```python
q4 = Tree(data="morning", left=l3, right=l5)
q3 = Tree(data='likedOtherSys', left=l4, right=l2)
q2 = Tree(data='TakenOtherSys', left=q4, right=q3)
q1 = Tree(data='isSystems', left=l1, right=q2)
```


```python
print(q1)
```

    Tree('isSystems') { left = Leaf('like'), right = Tree('TakenOtherSys') { left = Tree('morning') { left = Leaf('like'), right = Leaf('nah') }, right = Tree('likedOtherSys') { left = Leaf('nah'), right = Leaf('like') } } }
    

## Task 2


```python
# the comma-separated values were previously saved as a .txt file
data = pd.read_csv('data.txt', header=0)
new = data['rating'] >= 0
data['ok'] = new
```

## Task 3


```python
def single_feature_score(data: object, goal: str, feature: str, scorer='percent') -> float:
  assert type(goal) == str and type(feature) == str, "feature names should be passed as strings"
  assert type(data)==pd.DataFrame, "data should be passed as a Pandas DataFrame"
  sample_df = data[[feature, goal]]
  s_liked = sample_df.loc[sample_df[feature]==True]
  s_disliked = sample_df.loc[sample_df[feature]==False]
  all_liked = s_liked.shape[0]
  all_disliked = s_disliked.shape[0]

  like_liked = s_liked.loc[s_liked[goal]==True].shape[0]
  like_disliked = s_liked.loc[s_liked[goal]==False].shape[0]
  dislike_liked = s_disliked.loc[s_disliked[goal]==True].shape[0]
  dislike_disliked = s_disliked.loc[s_disliked[goal]==False].shape[0]
  if scorer == 'std':
    like_std = np.std(s_liked[goal])
    dislike_std = np.std(s_disliked[goal])
    return -(like_std + dislike_std)
  elif scorer == "percent":
    return (max([like_liked, like_disliked]) / all_liked) \
    + (max([dislike_liked, dislike_disliked]) / all_disliked)
```


```python
def best_feature(data, goal, features, scorer="percent"):
  # optional: avoid the lambda using `functools.partial`
  return max(features, key=lambda f: single_feature_score(data, goal, f, scorer))
```


```python
params = ["easy", "ai", "systems", "theory", "morning"]
```


```python
print(f"The best feature is: \n{best_feature(data, 'ok', params, 'percent')}")
```

    The best feature is: 
    systems
    


```python
print(f'The worst feature is: \n{min(params, key=lambda f: single_feature_score(data, "ok", f, "percent"))}')

```

    The worst feature is: 
    easy
    

## Task 4


```python
def DecisionTreeTrain(data: object, features: list, goal="ok") -> Tree:
  assert type(goal)==str, "goal feature name should be passed as a string"
  assert type(data)==pd.DataFrame, "data should be passed as a Pandas DataFrame"
  features = features.copy()
  ok_slice = data[goal]
  guess = np.max(ok_slice)
  if ok_slice.unique().shape[0] == 1:
    return Tree.leaf(guess)
  elif len(features) == 0:
    return Tree.leaf(guess)
  else:
    best = best_feature(data, goal, features, "percent")
    yes = data.loc[data[best] == True]
    no = data.loc[data[best] == False]
    features.pop(features.index(best))
    left = DecisionTreeTrain(no, features)
    right = DecisionTreeTrain(yes, features)
    return Tree(data=best, left=left, right=right)
```


```python
tr = DecisionTreeTrain(data, params)
```


```python
def DecisionTreeTest(tree, testp):
  assert type(testp) == dict or type(testp) == pd.DataFrame, "test point should be passed as a dict-like object"
  if tree.is_leaf() == True:
    return tree.data
  else:
    if testp[tree.data] == False:
      return DecisionTreeTest(tree.left, testp)
    else:
      return DecisionTreeTest(tree.right, testp)
```


```python
test_point = {"easy":False, "ai":False, "systems":True, "theory":False, "morning":False}
```


```python
DecisionTreeTest(tr, test_point)
```




    False



## Task 5


```python
def DecisionTree_with_depth(data, features, goal="ok", maxdepth=5):
  maxdepth -= 1
  assert type(goal)==str, "goal feature name should be passed as a string"
  features = features.copy()
  ok_slice = data[goal]
  guess = np.max(ok_slice)
  if ok_slice.unique().shape[0] == 1:
    return Tree.leaf(guess)
  elif len(features) == 0:
    return Tree.leaf(guess)
  elif maxdepth == 0:
    return Tree.leaf(guess)
  else:
    best = best_feature(data, goal, features, "percent")
    yes = data.loc[data[best] == True]
    no = data.loc[data[best] == False]
    features.pop(features.index(best))
    left = DecisionTree_with_depth(no, features, maxdepth=maxdepth)
    right = DecisionTree_with_depth(yes, features, maxdepth=maxdepth)
    return Tree(data=best, left=left, right=right)
```


```python
max_tr = DecisionTree_with_depth(data, params, maxdepth=3)
```


```python
def test_func(data) -> None:
  all_num = data.shape[0]
  random_half = np.random.randint(0, all_num, size=(all_num,))
  tests = []
  rights = []
  params = ["easy", "ai", "systems", "theory", "morning"]
  for i in random_half:
    test_pt = data.iloc[i,:].to_dict()
    tests.append(test_pt)
    rights.append(test_pt["ok"])
  
  scores = []
  for i in range(1, 6):
    results = []
    mx_tree = DecisionTree_with_depth(data, params, maxdepth=i)
    for item in tests:
      results.append(DecisionTreeTest(mx_tree, item))
    scores.append(f1_score(rights, results))
  fig = plt.scatter(range(1, 6), scores)
  plt.xlabel("Tree depth")
  plt.ylabel("F-1 score")
  plt.show()
```


```python
test_func(data=data)
```


![png](output_25_0.png)


### The plot shows the increase in the quality with the number of parameters, which, however, could lead to overfitting.
