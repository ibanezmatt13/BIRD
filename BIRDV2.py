#!/usr/bin/env python
# coding: utf-8

# # Basic Integrated Rigid-Origami Demonstrator (B.I.R.D)
# 
# ### The following is a code developed as part of a Group Design Project at the University of Southampton to model parameters of rigid origami folds and study their kinematics.
# 
# #### The code takes the inputs of the global geomtery of the fold, the desired number of rows and columns within the Miura Ori fold as well as the expected differnetial in angle between the sides of the fold under deformation.
# 

# Import the relevant external python libraries

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
get_ipython().run_line_magic('matplotlib', 'qt5')


# Define Input variables

# In[2]:


n_columns = 5
n_rows = 5
global_width = 200
global_depth = 100
global_height = 20
differential_angle = 5
V_L_ratio = 0.8
K_fold_K_facet_ratio = 1
E = 10.5


# #### Calculate Unit Cell parameters (in folded form)
# 
# H is the height of the unit cell in the Z dimension
# 
# S is the width of the unit cell in the X direction
# 
# L is the depth of the unit cell in the Y direction
# 
# V is the depth of the extruded section in the Y direction

# In[3]:


H = global_height
S = global_width/(2*(n_rows))
L = global_depth/(2*(n_columns)) 
V = L*V_L_ratio


# Calculate Unit Cell Net Geometry

# In[4]:


L0 = (H**2 + L**2)**0.5
S0 = ((((H**2)*(V**2))/(H**2 + L**2))+S**2)**0.5
V0 = (L*V)/((H**2+L**2)**0.5)
theta = np.arctan(H/L)
gamma = np.arctan(S0/V0)


# Calculate Poisson's Ratio for given specific input

# In[5]:


v_sl = (- S/L)*((1-np.cos(differential_angle)/np.sin(differential_angle)))


# Cycle through different combinations of number of rows 

# In[6]:


H_i = []
S_i = []
L_i = []
V_i = []
L0_i = [] 
S0_i = []
V0_i = []
theta_i = []
gamma_i = []
v_sl_i = []

for i in range(10):
    H_i.append(global_height)
    S_i.append(global_width/(2*(n_columns)))
    L_i.append(global_depth/(2*(i+1)))
    V_i.append(L*V_L_ratio)
    L0_i.append((H_i[i]**2 + (L_i[i])**2)**0.5)
    S0_i.append(((H_i[i]**2*V_i[i]**2)/(H_i[i]**2 + L_i[i]**2)+S_i[i]**2)**0.5)
    V0_i.append((L_i[i]*V_i[i])/((H_i[i]**2+L_i[i]**2)**0.5))
    theta_i.append(np.arctan(H_i[i]/L_i[i]))
    gamma_i.append(np.arctan(S0_i[i]/V0_i[i]))

    v_sl_i.append((- S_i[i]/L_i[i])*((1-np.cos((differential_angle))/np.sin((differential_angle)))))


# In[7]:


#plt.scatter( np.degrees(theta_i), v_sl_i)
plt.scatter( np.degrees(gamma_i), v_sl_i)


# Define Cell Coordinate Matrix

# In[8]:


n_unit_cells = n_rows*n_columns

cell_coord = np.zeros([n_columns, n_rows])

i=0
for j in range(n_columns):
    for k in range(n_rows):
        cell_coord[j][k] = i
        i = i+1


# Define Unit Cell Class

# In[9]:


class unit_cell:
    def __init__(self, cell_number):
        """Define cell number"""
        self.cell_n = cell_number
        
        """Define cell location"""
        self.x_coord = np.where(cell_coord == self.cell_n)[1][0]
        self.y_coord = np.where(cell_coord == self.cell_n)[0][0]
        
        """ Define unit cell node locations """
        self.node_coord = np.zeros([9,3])
        self.node_coord[0][0] = 0 + self.x_coord*2*S
        self.node_coord[0][1] = 0 + self.y_coord*2*L
        self.node_coord[0][2] = 0
        self.node_coord[1][0] = self.node_coord[0][0] + S
        self.node_coord[1][1] = self.node_coord[0][1] + V
        self.node_coord[1][2] = self.node_coord[0][2] + 0
        self.node_coord[2][0] = self.node_coord[0][0] + 2*S
        self.node_coord[2][1] = self.node_coord[0][1] + 0
        self.node_coord[2][2] = self.node_coord[0][2] + 0 
        self.node_coord[3][0] = self.node_coord[0][0] + 0 
        self.node_coord[3][1] = self.node_coord[0][1] + L
        self.node_coord[3][2] = self.node_coord[0][2] + H
        self.node_coord[4][0] = self.node_coord[0][0] + S
        self.node_coord[4][1] = self.node_coord[0][1] + L+V#-(L0**2-H**2)**0.5
        self.node_coord[4][2] = self.node_coord[0][2] + H
        self.node_coord[5][0] = self.node_coord[0][0] + 2*S
        self.node_coord[5][1] = self.node_coord[0][1] + L
        self.node_coord[5][2] = self.node_coord[0][2] + H
        self.node_coord[6][0] = self.node_coord[0][0] + 0 
        self.node_coord[6][1] = self.node_coord[0][1] + 2*L
        self.node_coord[6][2] = self.node_coord[0][2] + 0 
        self.node_coord[7][0] = self.node_coord[0][0] + S
        self.node_coord[7][1] = self.node_coord[0][1] + 2*L+V
        self.node_coord[7][2] = self.node_coord[0][2] + 0 
        self.node_coord[8][0] = self.node_coord[0][0] + 2*S
        self.node_coord[8][1] = self.node_coord[0][1] + 2*L
        self.node_coord[8][2] = self.node_coord[0][2] + 0 
    
        """ Define unit cell connectivity """
        self.node_fold_elements = np.zeros([12, 2])
        self.node_fold_elements[0][0] = 0
        self.node_fold_elements[0][1] = 1
        self.node_fold_elements[1][0] = 1
        self.node_fold_elements[1][1] = 2
        self.node_fold_elements[2][0] = 0
        self.node_fold_elements[2][1] = 3
        self.node_fold_elements[3][0] = 1
        self.node_fold_elements[3][1] = 4
        self.node_fold_elements[4][0] = 2
        self.node_fold_elements[4][1] = 5
        self.node_fold_elements[5][0] = 3
        self.node_fold_elements[5][1] = 4
        self.node_fold_elements[6][0] = 4
        self.node_fold_elements[6][1] = 5
        self.node_fold_elements[7][0] = 3
        self.node_fold_elements[7][1] = 6
        self.node_fold_elements[8][0] = 4
        self.node_fold_elements[8][1] = 7
        self.node_fold_elements[9][0] = 5
        self.node_fold_elements[9][1] = 8
        self.node_fold_elements[10][0] = 6
        self.node_fold_elements[10][1] = 7
        self.node_fold_elements[11][0] = 7
        self.node_fold_elements[11][1] = 8

        """ Define node facet elements """
        self.node_facet_elements = np.zeros([4, 2])
        self.node_facet_elements[0][0]  = 0
        self.node_facet_elements[0][1]  = 4
        self.node_facet_elements[1][0]  = 2
        self.node_facet_elements[1][1]  = 4
        self.node_facet_elements[2][0]  = 4
        self.node_facet_elements[2][1]  = 6
        self.node_facet_elements[3][0]  = 4
        self.node_facet_elements[3][1]  = 8
        
        """Define facet element stiffness"""
        self.K_facet = (E*1)/(((((self.node_coord[int(self.node_facet_elements[0][0])][0]-self.node_coord[int(self.node_facet_elements[0][1])][0])**2)+
                                        (self.node_coord[int(self.node_facet_elements[0][0])][1]-self.node_coord[int(self.node_facet_elements[0][1])][1])**2)+
                                        (self.node_coord[int(self.node_facet_elements[0][0])][2]-self.node_coord[int(self.node_facet_elements[0][1])][2])**2))**0.5

        """Define fold element stiffness"""
        self.K_fold = self.K_facet*K_fold_K_facet_ratio

        """Define node displacement"""
        self.node_disp = np.zeros([9,3])
        
        """Define unit cell net"""
        self.net_coord = np.zeros([9,3])
        self.net_coord[0][0] = 0 + self.x_coord*(2*S0)
        self.net_coord[0][1] = 0 + self.y_coord*(2*L0)
        self.net_coord[0][2] = 0
        self.net_coord[1][0] = self.net_coord[0][0] + S0
        self.net_coord[1][1] = self.net_coord[0][1] + V0
        self.net_coord[1][2] = self.net_coord[0][2] + 0
        self.net_coord[2][0] = self.net_coord[0][0] + 2*S0
        self.net_coord[2][1] = self.net_coord[0][1] + 0
        self.net_coord[2][2] = self.net_coord[0][2] + 0 
        self.net_coord[3][0] = self.net_coord[0][0] + 0 
        self.net_coord[3][1] = self.net_coord[0][1] + L0
        self.net_coord[3][2] = self.net_coord[0][2] + 0
        self.net_coord[4][0] = self.net_coord[0][0] + S0
        self.net_coord[4][1] = self.net_coord[0][1] + L0+V0
        self.net_coord[4][2] = self.net_coord[0][2] + 0
        self.net_coord[5][0] = self.net_coord[0][0] + 2*S0
        self.net_coord[5][1] = self.net_coord[0][1] + L0
        self.net_coord[5][2] = self.net_coord[0][2] + 0
        self.net_coord[6][0] = self.net_coord[0][0] + 0 
        self.net_coord[6][1] = self.net_coord[0][1] + 2*L0
        self.net_coord[6][2] = self.net_coord[0][2] + 0 
        self.net_coord[7][0] = self.net_coord[0][0] + S0
        self.net_coord[7][1] = self.net_coord[0][1] + 2*L0+V0
        self.net_coord[7][2] = self.net_coord[0][2] + 0 
        self.net_coord[8][0] = self.net_coord[0][0] + 2*S0
        self.net_coord[8][1] = self.net_coord[0][1] + 2*L0
        self.net_coord[8][2] = self.net_coord[0][2] + 0 


# Create the dictionary of unit cells corrisponding to the overall texture

# In[10]:


cells_dict = {}
for k in range(n_unit_cells):
    cells_dict['unit_cell_{}'.format(k)] = unit_cell(k)


# Plot the nodes, fold elements and facet elements

# In[11]:


fig1 = plt.figure()
ax = fig1.add_subplot(111, projection='3d')
ax.set_xlabel("Depth of the Origami Texture /mm")
ax.set_ylabel("Width of the Origami Texture /mm")
ax.set_zlabel("height of the Origami Texture /mm")
ax.set_title("Graphical Representation of the undeformed Muira Ori Texture")
#ax.set_xlim(0,200)
#ax.set_ylim(0,200)

for j in range(n_unit_cells):
    for i in range(len( cells_dict['unit_cell_{}'.format(j)].node_coord)):
        x = cells_dict['unit_cell_{}'.format(j)].node_coord[i][0]
        y = cells_dict['unit_cell_{}'.format(j)].node_coord[i][1]
        z = cells_dict['unit_cell_{}'.format(j)].node_coord[i][2]
        ax.scatter(x, y, z, s=50, c='b', depthshade=False)
        

    for i in range(len(cells_dict['unit_cell_{}'.format(j)].node_fold_elements)):
        node_0 = int(cells_dict['unit_cell_{}'.format(j)].node_fold_elements[i][0])
        node_1 = int(cells_dict['unit_cell_{}'.format(j)].node_fold_elements[i][1])
        xe = [cells_dict['unit_cell_{}'.format(j)].node_coord[node_0][0], cells_dict['unit_cell_{}'.format(j)].node_coord[node_1][0]]
        ye = [cells_dict['unit_cell_{}'.format(j)].node_coord[node_0][1], cells_dict['unit_cell_{}'.format(j)].node_coord[node_1][1]]
        ze = [cells_dict['unit_cell_{}'.format(j)].node_coord[node_0][2], cells_dict['unit_cell_{}'.format(j)].node_coord[node_1][2]]
        ax.plot(xe, ye, ze, 'k', linewidth=2)

    for i in range(len(cells_dict['unit_cell_{}'.format(j)].node_facet_elements)):
        node_0 = int(cells_dict['unit_cell_{}'.format(j)].node_facet_elements[i][0])
        node_1 = int(cells_dict['unit_cell_{}'.format(j)].node_facet_elements[i][1])
        xe = [cells_dict['unit_cell_{}'.format(j)].node_coord[node_0][0], cells_dict['unit_cell_{}'.format(j)].node_coord[node_1][0]]
        ye = [cells_dict['unit_cell_{}'.format(j)].node_coord[node_0][1], cells_dict['unit_cell_{}'.format(j)].node_coord[node_1][1]]
        ze = [cells_dict['unit_cell_{}'.format(j)].node_coord[node_0][2], cells_dict['unit_cell_{}'.format(j)].node_coord[node_1][2]]
        ax.plot(xe, ye, ze, 'r', linewidth=2)


# Plot the Muira ori net

# In[12]:


#fig2 = plt.figure()

plt.xlabel("Depth of the Origami Net /mm")
plt.ylabel("Width of the Origami Net /mm")
plt.title("Graphical Representation of the Muira Ori Net")
#fig2.set_xlim(0,200)
#fig2.set_ylim(0,200)

for j in range(n_unit_cells):
    for i in range(len( cells_dict['unit_cell_{}'.format(j)].net_coord)):
        x = cells_dict['unit_cell_{}'.format(j)].net_coord[i][0]
        y = cells_dict['unit_cell_{}'.format(j)].net_coord[i][1]
        plt.scatter(x, y, s=50, c='b')
        

    for i in range(len(cells_dict['unit_cell_{}'.format(j)].node_fold_elements)):
        node_0 = int(cells_dict['unit_cell_{}'.format(j)].node_fold_elements[i][0])
        node_1 = int(cells_dict['unit_cell_{}'.format(j)].node_fold_elements[i][1])
        xe = [cells_dict['unit_cell_{}'.format(j)].net_coord[node_0][0], cells_dict['unit_cell_{}'.format(j)].net_coord[node_1][0]]
        ye = [cells_dict['unit_cell_{}'.format(j)].net_coord[node_0][1], cells_dict['unit_cell_{}'.format(j)].net_coord[node_1][1]]
        plt.plot(xe, ye, 'k', linewidth=2)


# In[13]:


cells_dict['unit_cell_1'].K_facet


# In[ ]:


print(cells_dict['unit_cell_0'].node_coord)
