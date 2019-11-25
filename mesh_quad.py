from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import time
import collections
import itertools
import math
import networkx as nx
import matplotlib.pyplot as plt

with open('Original.dat','r') as file:
    list_lines = file.readlines()


class Element():

    def __init__(self,node1,node2,node3,node4):
        self.node1 = node1
        self.node2 = node2
        self.node3 = node3
        self.node4 = node4
        self.edge1 = [node1, node2]
        self.edge2 = [node2, node3]
        self.edge3 = [node3, node4]
        self.edge4 = [node4, node1]

    def normal_vector(self):

        p1 = np.array(grid_pos[str(self.node1)])
        p2 = np.array(grid_pos[str(self.node2)])
        p3 = np.array(grid_pos[str(self.node3)])

        v1 = p3 - p1
        v2 = p2 - p1

        cp = np.cross(v1,v2)

        a,b,c = cp
        d = np.dot(cp, p3)

        return a,b,c,d


'''
Dictionary formulation
'''
dict_cbeams={}
obj_cquads={}
##dict_grid={}
for lines in list_lines:
    out=[lines[i:i+8] for i in range(0, len(lines), 8)]
    line = [x.strip(' ') for x in out]
    if line[0]=='CBEAM':
        dict_cbeams[line[1]]=[int(line[3]),int(line[4])]
##
    if line[0]=='CQUAD4':
        obj_cquads[line[1]]=Element(int(line[3]),int(line[4]),int(line[5]),int(line[6]))

##    if line[0]=='GRID':
##        dict_grid.append([float(line[2]),float(line[3]),float(line[4])])


'''
List formulationa
'''

##dict_cbeams=[]
dict_cquads={}
dict_grid=[]
grid_pos={}
for lines in list_lines:
    out=[lines[i:i+8] for i in range(0, len(lines), 8)]
    line = [x.strip(' ') for x in out]
##    if line[0]=='CBEAM':
##        dict_cbeams.append([int(out[3]),int(out[4])])

    if line[0]=='CQUAD4':
        dict_cquads[line[1]]=[int(line[3]),int(line[4]),int(line[5]),int(line[6])]

    if line[0]=='GRID':
        #dict_grid.append(int(line[1]))
        grid_pos[line[1]]=(float(line[3]),float(line[4]),float(line[5]))
                                   

'''
#Plot 3d
'''
def network_plot_3D(G,angle):
    pos = nx.get_node_attributes(G, 'pos')

    # Get number of nodes
    n = G.number_of_nodes()

    # 3D network plot
    with plt.style.context(('ggplot')):
        
        fig = plt.figure(figsize=(10,7))
        ax = Axes3D(fig)
        
        # Loop on the pos dictionary to extract the x,y,z coordinates of each node
        for key, value in pos.items():
            xi = value[0]
            yi = value[1]
            zi = value[2]

            # Scatter plot
            ax.scatter(xi, yi, zi, alpha=0.7)
            ax.text(xi, yi, zi, '%s' % (str(key)))
        
        # Loop on the list of edges to get the x,y,z, coordinates of the connected nodes
        # Those two points are the extrema of the line to be plotted
        for i,j in enumerate(G.edges()):
            x = np.array((pos[str(j[0])][0], pos[str(j[1])][0]))
            y = np.array((pos[str(j[0])][1], pos[str(j[1])][1]))
            z = np.array((pos[str(j[0])][2], pos[str(j[1])][2]))
        
        # Plot the connecting lines
            ax.plot(x, y, z, c='black', alpha=0.5)
            ax.set_axis_off()

    plt.show()
    return



def get_elm_neighbors(elm):
    dict_vecini={}
    for key,values in dict_cquads.items():
        if all(elem in values for elem in obj_cquads[elm].edge1) and key!=elm:
            dict_vecini[key]=obj_cquads[elm].edge1
        elif all(elem in values for elem in obj_cquads[elm].edge2)and key!=elm:
            dict_vecini[key]=obj_cquads[elm].edge2
        elif all(elem in values for elem in obj_cquads[elm].edge3)and key!=elm:
            dict_vecini[key]=obj_cquads[elm].edge3
        elif all(elem in values for elem in obj_cquads[elm].edge4)and key!=elm:
            dict_vecini[key]=obj_cquads[elm].edge4
        else:
            continue
    return dict_vecini


def get_edge_not_bar(edge):
    check=[]
    if edge not in dict_cbeams.values():
        return edge

def merge_elms(elem1,elem2,no):
    nodes1=dict_cquads[elem1]
    nodes2=dict_cquads[elem2]
    merge=[]
    for i,item in enumerate(nodes1):
        if item in no:
            merge.append(nodes2[i])
        else:
            merge.append(item)
    return merge

def get_angle(a1,b1,c1,d1,a2,b2,c2,d2):
    d = (a1*a2+b1*b2+c1*c2)
    e1 = math.sqrt(a1*a1+b1*b1+c1*c1);
    e2 = math.sqrt(a2*a2+b2*b2+c2*c2);
    d = d/(e1*e2)
    radian = math.acos(d)
    return math.degrees(radian)

def check_angle(quad1,quad2,tol):
    (a1,b1,c1,d1) = obj_cquads[quad1].normal_vector()
    (a2,b2,c2,d2) = obj_cquads[quad2].normal_vector() 
    if get_angle(a1,b1,c1,d1,a2,b2,c2,d2) < tol:
        return True
    else:
        return False


def main():
    dict_cquads_merged=dict_cquads.copy()
    delete_keys=[]
    for e in dict_cquads:
        if e not in delete_keys:
            neighbors=get_elm_neighbors(e)
            for key,value in neighbors.items():
                check = check_angle(e,key,0.5)
                if check:
                    nodes = get_edge_not_bar(value)
                    if nodes:
                        new_values=merge_elms(e,key,nodes)
                        d = {e: new_values}
                        dict_cquads_merged.update(d)
                        dict_cquads.update(d)
                        delete_keys.append(key)
        else:
            pass

   
    for ytem in delete_keys:
        if ytem in dict_cquads_merged.keys():
            del dict_cquads_merged[ytem]
        
    with open('modifiedSoft.dat','w') as f:
        for i,j in dict_cquads_merged.items():
            f.write('{0:7} {1:8} {2:6} {3:7} {4:7} {5:7} {6:7} \n'.\
                    format('CQUAD4',i,'100',j[0],j[1],j[2],j[3]))
    
##    M=nx.Graph()
##    M.clear()
##    N=nx.Graph()
##
##    M.add_nodes_from(grid_pos.keys())
##    N.add_nodes_from(grid_pos.keys())
##
##    for item in dict_cquads.values():
##        M.add_edges_from([(item[0],item[1])])
##        M.add_edges_from([(item[1],item[2])])
##        M.add_edges_from([(item[2],item[3])])
##        #M.add_edges_from([(item[3],item[0])])
##        
##    for item in dict_cquads_merged.values():
##        N.add_edges_from([(item[0],item[1]),(item[1],item[2]),(item[2],item[3]),(item[3],item[0])])
##
##
##    for n, p in grid_pos.items():
##        M.nodes[n]['pos'] = p
##    for n, p in grid_pos.items():
##        N.nodes[n]['pos'] = p    
##    #pos = nx.get_node_attributes(N,'pos')
##
##
##    #for value in G.nodes:
##    #    print(type(value),value)
##        
##    #nx.draw_networkx_nodes(M,pos)
##
##
##    network_plot_3D(M,0)
##    plt.show()


main()

