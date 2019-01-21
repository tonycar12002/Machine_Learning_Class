# -*- coding: utf-8 -*-

import collections
graph= collections.defaultdict(list) 

graph[1].append(500)
graph[1].append(3)
graph[1].append(88)
graph[1].append(79)
graph[1].append(123)

ans = graph[1]
ans.sort()
print(type(ans))


for i in ans:
    print(i)