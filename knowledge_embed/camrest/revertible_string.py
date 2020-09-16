import os, sys

class RevertibleString():
    def __init__(self, string):
        self.original_str = string
        self.str = string
        self.entities = []
    
    def to_origin(self):
        self.str = self.original_str
        self.entities = []
    
    def __repr__(self):
        return self.str    
    
    def __str__(self):
        return self.str
    
    def __len__(self):
        return len(self.str)