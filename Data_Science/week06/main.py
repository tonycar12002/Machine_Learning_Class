# -*- coding: utf-8 -*-

#%% Import Library
import numpy as np
import os

#%% 

class Graph:
    def __init__(self, file_name):
        
        self.V = 82168
        self.k_core = 1200 
        self.visited = [False] * self.V
        self.vertex_degree = [0] * self.V    
                             
                             
        self.read_file(file_name)
        
    def read_file(self, file_name):
        with open(file_name, 'r') as file:
            lines = file.readlines()
            for line in lines:
                line = line.replace("\n", "")
                line_split = line.split(" ")
                print(line_split)

if __name__ == '__main__':
    graph = Graph("hw6_dataset.txt")
    
            