# -*- coding: utf-8 -*-

#%% Import Library
import numpy as np
import os
import collections
import time

#%% 
class Graph:
    def __init__(self, file_input, file_output):
        
        self.V = 82168
        self.k_core = 1200 
        self.visited = [False] * self.V
        self.vertex_degree = [0] * self.V    
        self.graph= collections.defaultdict(list)                          
                           
        
        t_start = time.process_time()
        self.read_file(file_input)
        print("Read file done. Time = " + str(time.process_time()-t_start))
        self.remove_low_kcore()
        print("Preprocess done. Time = " + str(time.process_time()-t_start))
        self.print_kcore(file_output)
        print("Write ans done. Time = " + str(time.process_time()-t_start))
        
    def print_kcore(self, file_name):
        root = None
        for i in range(self.V):
            if self.visited[i] == False:
                root = i
                break
        ans = self.graph[root]
        ans.sort()
        with open(file_name, 'w+') as file:
            file.write(str(root)  + '\n')
            for i in ans:
                file.write(str(i)  + '\n')
    
    def remove_low_kcore(self):
        all_set = False
        while (all_set == False):
            all_set = True
            for i in range(self.V):
                if self.visited[i] == False and self.vertex_degree[i] < self.k_core:
                    all_set = False
                    self.visited[i] = True
                    self.vertex_degree[i] = 0
                    for link_node in self.graph[i]:
                        self.vertex_degree[link_node] -= 1
                        self.graph[link_node].remove(i)
                    self.graph[i] = [] 
        
    def read_file(self, file_name):
        with open(file_name, 'r') as file:
            lines = file.readlines()
            for line in lines:
                line = line.replace("\n", "")
                line_split = line.split(" ")
                self.add_edge(int(line_split[0]), int(line_split[1]))
    
    def add_edge(self, node1, node2):
        self.vertex_degree[node1] += 1
        self.vertex_degree[node2] += 1
        self.graph[node1].append(node2)
        self.graph[node2].append(node1)

if __name__ == '__main__':
    graph = Graph("hw6_dataset.txt", "ans.txt")
    
            