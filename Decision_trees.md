## Task 1


```python
l1 = Tree.leaf('like')
l2 = Tree.leaf('like')
l3 = Tree.leaf('like')
l4 = Tree.leaf('nah')
l5 = Tree.leaf('nah')

q4 = Tree(data="morning", left=l3, right=l5)
q3 = Tree(data='likedOtherSys', left=l4, right=l2)
q2 = Tree(data='TakenOtherSys', left=q4, right=q3)
q1 = Tree(data='isSystems', left=l1, right=q2)

print(q1)
```

    Tree('isSystems') { left = Leaf('like'), right = Tree('TakenOtherSys') { left = Tree('morning') { left = Leaf('like'), right = Leaf('nah') }, right = Tree('likedOtherSys') { left = Leaf('nah'), right = Leaf('like') } } }
    

## Task 3


```python
def best_feature(data, goal, features, scorer="percent"):
  # optional: avoid the lambda using `functools.partial`
  return max(features, key=lambda f: single_feature_score(data, goal, f, scorer))

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
tr = DecisionTreeTrain(data, params)

test_point = {"easy":False, "ai":False, "systems":True, "theory":False, "morning":False}

DecisionTreeTest(tr, test_point)
```




    False



## Task 5


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


![png](output_10_0.png)


### The plot shows the increase in the quality with the number of parameters, which, however, could lead to overfitting.
