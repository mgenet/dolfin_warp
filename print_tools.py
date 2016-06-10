#coding=utf8

########################################################################
###                                                                  ###
### Created by Martin Genet, 2016                                    ###
###                                                                  ###
### École Polytechnique, Palaiseau, France                           ###
###                                                                  ###
########################################################################

import myVTKPythonLibrary as myVTK

########################################################################

#class Printer:
    #def __init__(self):
        #self.cur_level = 0

    #def inc(self):
        #self.cur_level += 1

    #def dec(self):
        #self.cur_level -= 1

    #def print_str(self, string, var_level=0):
        #print  " | "*(self.cur_level+var_level) + string

    #def print_var(self, name, val, var_level=0):
        #print " | "*(self.cur_level+var_level) + name + " = " + str(val)

    #def print_sci(self, name, val, var_level=0):
        #print " | "*(self.cur_level+var_level) + name.ljust(13) + " = " + format(val,".4e")

########################################################################

def print_str(tab, string):
    print  " | "*tab + string

def print_var(tab, name, val):
    print " | "*tab + name + " = " + str(val)

def print_sci(tab, name, val):
    print " | "*tab + name.ljust(13) + " = " + format(val,".4e")

