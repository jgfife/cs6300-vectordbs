
from puzzle import GOAL, pretty, neighbors, manhattan
from search import AStar

start = (1,2,3,4,5,6,7,0,8)  # one move to goal (R)
solver = AStar(manhattan, neighbors, cost_hooks=[])
res = solver.search(start, GOAL)
print("Depth:", res["depth"])
print("Nodes:", res["nodes_expanded"])
print("Moves:", [a for (_,a) in res["solution"] if a is not None])
