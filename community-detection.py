import itertools

import pandas
import pandas as pd
import numpy as np
import random
import networkx as nx
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from networkx.algorithms import community as comm
from networkx.algorithms import centrality as cntr
import os
import scipy as sp # WARNING networkx crashes for many nodes and asks to import this ex. L,2102 -> P

############################### FG COLOR DEFINITIONS ###############################
class bcolors:
    HEADER      = '\033[95m'
    OKBLUE      = '\033[94m'
    OKCYAN      = '\033[96m'
    OKGREEN     = '\033[92m'
    WARNING     = '\033[93m'
    FAIL        = '\033[91m'
    ENDC        = '\033[0m'    # RECOVERS DEFAULT TEXT COLOR
    BOLD        = '\033[1m'
    UNDERLINE   = '\033[4m'

    def disable(self):
        self.HEADER     = ''
        self.OKBLUE     = ''
        self.OKGREEN    = ''
        self.WARNING    = ''
        self.FAIL       = ''
        self.ENDC       = ''

########################################################################################
############################## MY ROUTINES LIBRARY STARTS ##############################
########################################################################################

# SIMPLE ROUTINE TO CLEAR SCREEN BEFORE SHOWING A MENU...
def my_clear_screen():

    os.system('cls' if os.name == 'nt' else 'clear')

# CREATE A LIST OF RANDOMLY CHOSEN COLORS...
def my_random_color_list_generator(REQUIRED_NUM_COLORS):

    my_color_list = [   'red',
                        'green',
                        'cyan',
                        'brown',
                        'olive',
                        'orange',
                        'darkblue',
                        'purple',
                        'yellow',
                        'hotpink',
                        'teal',
                        'gold']

    my_used_colors_dict = { c:0 for c in my_color_list }     # DICTIONARY OF FLAGS FOR COLOR USAGE. Initially no color is used...
    constructed_color_list = []

    if REQUIRED_NUM_COLORS <= len(my_color_list):
        for i in range(REQUIRED_NUM_COLORS):
            constructed_color_list.append(my_color_list[i])
        
    else: # REQUIRED_NUM_COLORS > len(my_color_list)   
        constructed_color_list = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(REQUIRED_NUM_COLORS)]
 
    return(constructed_color_list)


# VISUALISE A GRAPH WITH COLORED NODES AND LINKS
def my_graph_plot_routine(G,fb_nodes_colors,fb_links_colors,fb_links_styles,graph_layout,node_positions):
    plt.figure(figsize=(10,10))
    
    if len(node_positions) == 0:
        if graph_layout == 'circular':
            node_positions = nx.circular_layout(G)
        elif graph_layout == 'random':
            node_positions = nx.random_layout(G, seed=50)
        elif graph_layout == 'planar':
            node_positions = nx.planar_layout(G)
        elif graph_layout == 'shell':
            node_positions = nx.shell_layout(G)
        else:   #DEFAULT VALUE == spring
            node_positions = nx.spring_layout(G)

    nx.draw(G, 
        with_labels=True,           # indicator variable for showing the nodes' ID-labels
        style=fb_links_styles,      # edge-list of link styles, or a single default style for all edges
        edge_color=fb_links_colors, # edge-list of link colors, or a single default color for all edges
        pos = node_positions,       # node-indexed dictionary, with position-values of the nodes in the plane
        node_color=fb_nodes_colors, # either a node-list of colors, or a single default color for all nodes
        node_size = 100,            # node-circle radius
        alpha = 0.9,                # fill-transparency 
        width = 0.5                 # edge-width
        )
    plt.show()

    return(node_positions)


########################################################################################
# MENU 1 STARTS: creation of input graph ### 
########################################################################################
def my_menu_graph_construction(G,node_names_list,node_positions):

    my_clear_screen()

    breakWhileLoop  = False

    while not breakWhileLoop:
        print(bcolors.OKGREEN 
        + '''
========================================
(1.1) Create graph from fb-food data set (fb-pages-food.nodes and fb-pages-food.nodes)\t[format: L,<NUM_LINKS>]
(1.2) Create RANDOM Erdos-Renyi graph G(n,p).\t\t\t\t\t\t[format: R,<number of nodes>,<edge probability>]
(1.3) Print graph\t\t\t\t\t\t\t\t\t[format: P,<GRAPH LAYOUT in {spring,random,circular,shell }>]    
(1.4) Continue with detection of communities.\t\t\t\t\t\t[format: N]
(1.5) EXIT\t\t\t\t\t\t\t\t\t\t[format: E]
----------------------------------------
        ''' + bcolors.ENDC)
        
        my_option_list = str(input('\tProvide your (case-sensitive) option: ')).split(',')
        
        if my_option_list[0] == 'L':
            MAX_NUM_LINKS = 2102    # this is the maximum number of links in the fb-food-graph data set...

            if len(my_option_list) > 2:
                print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                print(bcolors.WARNING + "\tERROR MESSAGE: Too many parameters. Try again..." + bcolors.ENDC)
                print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)            

            else:
                if len(my_option_list) == 1:
                    NUM_LINKS = MAX_NUM_LINKS
                else: #...len(my_option_list) == 2...
                    NUM_LINKS = int(my_option_list[1])

                if NUM_LINKS > MAX_NUM_LINKS or NUM_LINKS < 1:
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                    print(bcolors.WARNING + "\tERROR Invalid number of links to read from data set. It should be in {1,2,...,2102}. Try again..." + bcolors.ENDC)
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)            
                else:
                    # LOAD GRAPH FROM DATA SET...
                    G,node_names_list = read_graph_from_csv(NUM_LINKS)
                    print(  "\tConstructing the FB-FOOD graph with n =",G.number_of_nodes(),
                            "vertices and m =",G.number_of_edges(),"edges (after removal of loops).")

        elif my_option_list[0] == 'R':

            if len(my_option_list) > 3:
                print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                print(bcolors.WARNING + "\tERROR MESSAGE: Wrong number of parameters. Try again..." + bcolors.ENDC)
                print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
            
            else: # ...len(my_option_list) <= 3...
                if len(my_option_list) == 1:
                    NUM_NODES = 100                     # DEFAULT NUMBER OF NODES FOR THE RANDOM GRAPH...
                    ER_EDGE_PROBABILITY = 2 / NUM_NODES # DEFAULT VALUE FOR ER_EDGE_PROBABILITY...

                elif len(my_option_list) == 2:
                    NUM_NODES = int(my_option_list[1])
                    ER_EDGE_PROBABILITY = 2 / max(1,NUM_NODES) # AVOID DIVISION WITH ZERO...

                else: # ...NUM_NODES == 3...
                    NUM_NODES = int(my_option_list[1])
                    ER_EDGE_PROBABILITY = float(my_option_list[2])

                if ER_EDGE_PROBABILITY < 0 or ER_EDGE_PROBABILITY > 1 or NUM_NODES < 2:
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                    print(bcolors.WARNING + "\tERROR MESSAGE: Invalid probability mass or number of nodes of G(n,p). Try again..." + bcolors.ENDC)
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

                else:
                    G = nx.erdos_renyi_graph(NUM_NODES, ER_EDGE_PROBABILITY)
                    print(bcolors.ENDC +    "\tConstructing random Erdos-Renyi graph with n =",G.number_of_nodes(),
                                            "vertices and edge probability p =",ER_EDGE_PROBABILITY,
                                            "which resulted in m =",G.number_of_edges(),"edges.")

                    node_names_list = [ x for x in range(NUM_NODES) ]

        elif my_option_list[0] == 'P':                  # PLOT G...
            print("Printing graph G with",G.number_of_nodes(),"vertices,",G.number_of_edges(),"edges and",nx.number_connected_components(G),"connected components." )
                
            if len(my_option_list) > 3:
                print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                print(bcolors.WARNING + "\tERROR MESSAGE: Wrong number of parameters. Try again..." + bcolors.ENDC)
                print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                
            else:
                if len(my_option_list) <= 1:
                    graph_layout = 'spring'     # ...DEFAULT graph_layout value...
                    reset_node_positions = 'Y'  # ...DEFAULT decision: erase node_positions...

                elif len(my_option_list) == 2: 
                    graph_layout = str(my_option_list[1])
                    reset_node_positions = 'Y'  # ...DEFAULT decision: erase node_positions...

                else: # ...len(my_option_list) == 3...
                    graph_layout = str(my_option_list[1])
                    reset_node_positions = str(my_option_list[2])

                if graph_layout not in ['spring','random','circular','shell']:
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                        print(bcolors.WARNING + "\tERROR MESSAGE: Incomprehensible choice for graph layout. Try again..." + bcolors.ENDC)
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

                elif reset_node_positions not in ['Y','y','N','n']:
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                        print(bcolors.WARNING + "\tERROR MESSAGE: Incomprehensible decision for resetting node positions. Try again..." + bcolors.ENDC)
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

                else:
                    if reset_node_positions in ['y','Y']:
                        node_positions = []         # ...ERASE previous node positions...

                    node_positions = my_graph_plot_routine(G,'grey','blue','solid',graph_layout,node_positions)

        elif my_option_list[0] == 'N':
            NUM_NODES = G.number_of_nodes()
            if NUM_NODES == 0:
                print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                print(bcolors.WARNING + "\tERROR MESSAGE: You have not yet constructed a graph to work with. Try again..." + bcolors.ENDC)
                print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
            else:
                my_clear_screen()
                breakWhileLoop = True
            
        elif my_option_list[0] == 'E':
            quit()

        else:
            print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
            print(bcolors.WARNING + "\tERROR MESSAGE: Incomprehensible input was provided. Try again..." + bcolors.ENDC)
            print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

    return(G,node_names_list,node_positions)

########################################################################################
# MENU 2: detect communities in the constructed graph 
########################################################################################
def my_menu_community_detection(G,node_names_list,node_positions,hierarchy_of_community_tuples):

    breakWhileLoop = False

    while not breakWhileLoop:
            print(bcolors.OKGREEN 
                + '''
========================================
(2.1) Add random edges from each node\t\t\t[format: RE,<NUM_RANDOM_EDGES_PER_NODE>,<EDGE_ADDITION_PROBABILITY in [0,1]>]
(2.2) Add hamilton cycle (if graph is not connected)\t[format: H]
(2.3) Print graph\t\t\t\t\t[format: P,<GRAPH LAYOUT in { spring, random, circular, shell }>,<ERASE NODE POSITIONS in {Y,N}>]
(2.4) Compute communities with GIRVAN-NEWMAN\t\t[format: C,<ALG CHOICE in { O(wn),N(etworkx) }>,<GRAPH LAYOUT in {spring,random,circular,shell }>]
(2.5) Compute a binary hierarchy of communities\t\t[format: D,<NUM_DIVISIONS>,<GRAPH LAYOUT in {spring,random,circular,shell }>]
(2.6) Compute modularity-values for all community partitions\t[format: M]
(2.7) Visualize the communities of the graph\t\t[format: V,<GRAPH LAYOUT in {spring,random,circular,shell}>]
(2.8) EXIT\t\t\t\t\t\t[format: E]
----------------------------------------
            ''' + bcolors.ENDC)

            my_option_list = str(input('\tProvide your (case-sensitive) option: ')).split(',')

            if my_option_list[0] == 'RE':                    # 2.1: ADD RANDOM EDGES TO NODES...

                if len(my_option_list) > 3:
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                    print(bcolors.WARNING + "\tERROR MESSAGE: Wrong number of parameters. [format: D,<NUM_RANDOM_EDGES>,<EDGE_ADDITION_PROBABILITY>]. Try again..." + bcolors.ENDC)
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

                else:
                    if len(my_option_list) == 1:
                        NUM_RANDOM_EDGES = 1                # DEFAULT NUMBER OF RANDOM EDGES TO ADD (per node)
                        EDGE_ADDITION_PROBABILITY = 0.25    # DEFAULT PROBABILITY FOR ADDING EACH RANDOM EDGE (independently of other edges) FROM EAC NODE (independently from other nodes)...

                    elif len(my_option_list) == 2:
                        NUM_RANDOM_EDGES = int(my_option_list[1])
                        EDGE_ADDITION_PROBABILITY = 0.25    # DEFAULT PROBABILITY FOR ADDING EACH RANDOM EDGE (independently of other edges) FROM EAC NODE (independently from other nodes)...

                    else:
                        NUM_RANDOM_EDGES = int(my_option_list[1])
                        EDGE_ADDITION_PROBABILITY = float(my_option_list[2])
            
                    # CHECK APPROPIATENESS OF INPUT AND RUN THE ROUTINE...
                    if NUM_RANDOM_EDGES-1 not in range(5):
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                        print(bcolors.WARNING + "\tERROR MESSAGE: Too many random edges requested. Should be from {1,2,...,5}. Try again..." + bcolors.ENDC) 
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

                    elif EDGE_ADDITION_PROBABILITY < 0 or EDGE_ADDITION_PROBABILITY > 1:
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                        print(bcolors.WARNING + "\tERROR MESSAGE: Not appropriate value was given for EDGE_ADDITION PROBABILITY. Try again..." + bcolors.ENDC)
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

                    else: 
                        G = add_random_edges_to_graph(G, node_names_list, NUM_RANDOM_EDGES, EDGE_ADDITION_PROBABILITY)

            elif my_option_list[0] == 'H':                  #2.2: ADD HAMILTON CYCLE...
                    G = add_hamilton_cycle_to_graph(G, node_names_list)

            elif my_option_list[0] == 'P':                  # 2.3: PLOT G...
                print("Printing graph G with",G.number_of_nodes(),"vertices,",G.number_of_edges(),"edges and",nx.number_connected_components(G),"connected components." )
                
                if len(my_option_list) > 2:
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                    print(bcolors.WARNING + "\tERROR MESSAGE: Wrong number of parameters. Try again..." + bcolors.ENDC)
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                
                else:
                    if len(my_option_list) <= 1:
                        graph_layout = 'spring'     # ...DEFAULT graph_layout value...

                    else: # ...len(my_option_list) == 2... 
                        graph_layout = str(my_option_list[1])

                    if graph_layout not in ['spring','random','circular','shell']:
                            print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                            print(bcolors.WARNING + "\tERROR MESSAGE: Incomprehensible choice for graph layout. Try again..." + bcolors.ENDC)
                            print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

                    else:
                        if len(my_option_list) == 2:
                            node_positions = []         # ...ERASE previous node positions...

                        node_positions = my_graph_plot_routine(G,'grey','blue','solid',graph_layout,node_positions)

            elif my_option_list[0] == 'C':      # 2.4: COMPUTE ONE-SHOT GN-COMMUNITIES
                NUM_OPTIONS = len(my_option_list)

                if NUM_OPTIONS > 3:
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                    print(bcolors.WARNING + "\tERROR MESSAGE: Wrong number of parameters. Try again..." + bcolors.ENDC)
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

                else:
                    if NUM_OPTIONS == 1:
                        alg_choice  = 'N'            # DEFAULT COMM-DETECTION ALGORITHM == NX_GN
                        graph_layout = 'spring'     # DEFAULT graph layout == spring

                    elif NUM_OPTIONS == 2:
                        alg_choice  = str(my_option_list[1])
                        graph_layout = 'spring'     # DEFAULT graph layout == spring
                
                    else: # ...NUM_OPTIONS == 3...
                        alg_choice      = str(my_option_list[1])
                        graph_layout    = str(my_option_list[2])

                    # CHECKING CORRECTNESS OF GIVWEN PARAMETERS...
                    if alg_choice == 'N' and graph_layout in ['spring','circular','random','shell']:#changed this to return community tuples
                        community_tuples = use_nx_girvan_newman_for_communities(G, graph_layout, node_positions)

                    elif alg_choice == 'O'and graph_layout in ['spring','circular','random','shell']:
                        community_tuples = one_shot_girvan_newman_for_communities(G, graph_layout, node_positions)

                    else:
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                        print(bcolors.WARNING + "\tERROR MESSAGE: Incomprehensible parameters for executing the GN-algorithm. Try again..." + bcolors.ENDC)
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

            elif my_option_list[0] == 'D':          # 2.5: COMUTE A BINARY HIERARCHY OF COMMUNITY PARRTITIONS
                NUM_OPTIONS = len(my_option_list)
                NUM_NODES = G.number_of_nodes()
                NUM_COMPONENTS = nx.number_connected_components(G)
                MAX_NUM_DIVISIONS = min( 8*NUM_COMPONENTS , np.floor(NUM_NODES/4) )

                if NUM_OPTIONS > 3:
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                    print(bcolors.WARNING + "\tERROR MESSAGE: Wrong number of parameters. Try again..." + bcolors.ENDC)
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

                else:
                    if NUM_OPTIONS == 1:
                        number_of_divisions = 2*NUM_COMPONENTS      # DEFAULT number of communities to look for 
                        graph_layout = 'spring'                     # DEFAULT graph layout == spring
                        
                    elif NUM_OPTIONS == 2:
                        number_of_divisions = int(my_option_list[1])
                        graph_layout = 'spring'                     # DEFAULT graph layout == spring
                    
                    else: #...NUM_OPTIONS == 3...
                        number_of_divisions = int(my_option_list[1])
                        graph_layout = str(my_option_list[2])

                    # CHECKING SYNTAX OF GIVEN PARAMETERS...
                    if number_of_divisions < NUM_COMPONENTS or number_of_divisions > MAX_NUM_DIVISIONS:
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                        print(bcolors.WARNING + "\tERROR MESSAGE: The graph has already",NUM_COMPONENTS,"connected components." + bcolors.ENDC)
                        print(bcolors.WARNING + "\tProvide a number of divisions in { ",NUM_COMPONENTS,",",MAX_NUM_DIVISIONS,"}. Try again..." + bcolors.ENDC)
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

                    elif graph_layout not in ['spring','random','circular','shell']:
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                        print(bcolors.WARNING + "\tERROR MESSAGE: Incomprehensible choice of a graph layout. Try again..." + bcolors.ENDC)
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

                    else:#changed this to return the hierarchy
                        hierarchy_of_community_tuples = divisive_community_detection(G, number_of_divisions, graph_layout, node_positions)

            elif my_option_list[0] == 'M':      # 2.6: DETERMINE PARTITION OF MIN-MODULARITY, FOR A GIVEN BINARY HIERARCHY OF COMMUNITY PARTITIONS
                community_tuples = determine_opt_community_structure(G, hierarchy_of_community_tuples)


            elif my_option_list[0] == 'V':      # 2.7: VISUALIZE COMMUNITIES WITHIN GRAPH

                NUM_OPTIONS = len(my_option_list)

                if NUM_OPTIONS > 2:
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                    print(bcolors.WARNING + "\tERROR MESSAGE: Wrong number of parameters. Try again..." + bcolors.ENDC)
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                
                else:

                    if NUM_OPTIONS == 1:
                        graph_layout = 'spring'                     # DEFAULT graph layout == spring
                    
                    else: # ...NUM_OPTIONS == 2...
                        graph_layout = str(my_option_list[1])

                    if graph_layout not in ['spring','random','circular','shell']:
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                        print(bcolors.WARNING + "\tERROR MESSAGE: Incomprehensible choice of a graph layout. Try again..." + bcolors.ENDC)
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

                    else:
                        visualize_communities(G, community_tuples, graph_layout, node_positions)

            elif my_option_list[0] == 'E':
                #EXIT the program execution...
                quit()

            else:
                print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                print(bcolors.WARNING + "\tERROR MESSAGE: Incomprehensible input was provided. Try again..." + bcolors.ENDC)
                print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
    ### MENU 2 ENDS: detect communities in the constructed graph ### 

########################################################################################
############################### MY ROUTINES LIBRARY ENDS ############################### 
########################################################################################

########################################################################################
##########################  ROUTINES LIBRARY STARTS ##########################
# FILL IN THE REQUIRED ROUTINES FROM THAT POINT ON...
########################################################################################

########################################################################################
def read_graph_from_csv(NUM_LINKS):

    print(bcolors.ENDC + "\t" + '''
        ########################################################################################
        # CREATE GRAPH FROM EDGE_CSV DATA 
        # ...(if needed) load all details for the nodes to the fb_nodes DATAFRAME
        # ...create DATAFRAME edges_df with the first MAX_NUM_LINKS, from the edges dataset
        # ...edges_df has one row per edge, and two columns, named 'node_1' and 'node_2
        # ...CLEANUP: remove each link (from the loaded ones) which is a LOOP
        # ...create node_names_list of NODE IDs that appear as terminal points of the loaded edges
        # ...create graph from the edges_df dataframe
        # ...return the constructed graph
        ########################################################################################
''')
    nodes_dataframe = pandas.read_csv('fb-pages-food.nodes')
    # create edges dataframe with the first MAX_NUM_LINKS, from the edges dataset
    fb_links_df = pandas.read_csv('fb-pages-food.edges', nrows=NUM_LINKS)
    # remove each edge which is a loop
    fb_links_loopless_df = fb_links_df[fb_links_df.node_1 != fb_links_df.node_2]
    # create node_names_list from edges_dataframe
    node_names_list = fb_links_loopless_df.node_1.tolist() + fb_links_loopless_df.node_2.tolist()
    # remove duplicates from node_names_list
    node_names_list = list(set(node_names_list))
    # create graph from edges_dataframe
    G = nx.from_pandas_edgelist(fb_links_loopless_df, 'node_1', 'node_2', create_using=nx.Graph())
    # display the graph
    # use the command line since we return it from this function
    # my_graph_plot_routine(G, "r", "r", [], 'spring',[])

    return G, node_names_list
    # rest is not necessary
    print(bcolors.ENDC + "\tUSEFUL FUNCTIONS:")

    print(  bcolors.ENDC + "\t\t The routine " 
            + bcolors.OKCYAN + "nx.from_pandas_edgelist(...) "
            + bcolors.ENDC 
            + '''creates the graph from a dataframe representing its edges,
                 one per edge, in two columns representing the tail-node 
                 (node_1) and the head-node (node_2) of the edge.\n''')

######################################################################################################################
# ...(a)  IMPLEMENTATION OF ONE-SHOT GN-ALGORITHM...
######################################################################################################################
def one_shot_girvan_newman_for_communities(G, graph_layout, node_positions):

    print(  bcolors.ENDC 
            + "\tCalling routine " 
            + bcolors.HEADER + "_one_shot_girvan_newman_for_communities(G,graph_layout,node_positions)\n" 
            + bcolors.ENDC)

    print(bcolors.ENDC + "\t" + '''
        ######################################################################################################################
        # PROVIDE YOUR OWN ROUTINE WHICH CREATES K+1 COMMUNITIES FOR A GRAPH WITH K CONNECTED COMPONENTS, AS FOLLOWS:
        # 
        # INPUT: 
        #   G = the constructed graph to work with. 
        #   graph_layout = a string determining how to position the graph nodes in the plane.
        #   node_positions = a list of previously computed node positions in the plane (for next call of my_graph_plot_routine)
        # OUTPUT: 
        #    community_tuples = a list of K+1 tuples of node_IDs, each tuple determining a different community
        # PSEUDOCODE:    
        # ...THE K CONNECTED COMPONENTS OF G ARE COMPUTED. 
        # ...THE LIST community_tuples IS INITIALIZED, WITH ONE NODES-TUPLE FOR EACH CONNECTED COMPONENT DEFINING A DIFFERENT COMMUNITY.
        # ...GCC = THE SUBGRAPH OF G INDUCED BY THE NODES OF THE LARGEST COMMUNITY, LC.
        # ...SPLIT LC IN TWO SUBCOMMUNITIES, BY REPEATEDLY REMOVING MAX_BETWEENNESS EDGES FROM GCC, UNTIL ITS DISCONNECTION.
        # ...THE NODE-LISTS OF THE TWO COMPONENTS OF GCC (AFTER THE EDGE REMOVALS) ARE THE NEW SUBCOMMUNITIES THAT SUBSTITUTE LC 
        # IN THE LIST community_tuples, AS SUGGESTED BY THE GIRVAN-NEWMAN ALGORITHM...
        ######################################################################################################################
''')

    start_time = time.time()
    # find the connected components of G
    connected_components = list(nx.connected_components(G))
    print(connected_components)
    # find the largest connected component
    largest_connected_component = max(connected_components, key=len)
    # girvan-newman algorithm for largest connected component
    # find max_betweenness
    # create subgraph of largest connected component

    subG = G.subgraph(largest_connected_component).copy()
    while True:
        # find max_betweenness
        max_betweenness = nx.edge_betweenness_centrality(subG)
        max_betweenness_edge = max(max_betweenness.items(), key=lambda x: x[1])
        # remove edge with max_betweenness typically only one edge has max_betweenness
        subG.remove_edge(max_betweenness_edge[0][0], max_betweenness_edge[0][1])

        # check if subG is disconnected
        if nx.is_connected(subG) == False:
            # find connected components of subG
            new_connected_components = list(nx.connected_components(subG))
            break

    # remove largest connected component from connected components and add new connected components to connected components
    connected_components.remove(largest_connected_component)
    connected_components.extend(new_connected_components)


    print(bcolors.ENDC  + "\tUSEFUL FUNCTIONS:")
 
    print(bcolors.ENDC  + "\t\t"    + bcolors.OKCYAN    + "sorted(nx.connected_components(G), key=len, reverse=True) " + bcolors.ENDC 
                        + "initiates the community_tuples with the node-sets of the connected components of the graph G, sorted by their size")
 
    print(bcolors.ENDC  + "\t\t"    + bcolors.OKCYAN    + "G.subgraph(X).copy() " 
                                    + bcolors.ENDC      + "creates the subgraph of G induced by a subset (even in list format) X of nodes.")
 
    print(bcolors.ENDC  + "\t\t"    + bcolors.OKCYAN    + "edge_betweenness_centrality(...) " 
                                    + bcolors.ENDC      + "of networkx.algorithms.centrality computes edge-betweenness values.")
 
    print(bcolors.ENDC  + "\t\t"    + bcolors.OKCYAN    + "is_connected(G) " 
                                    + bcolors.ENDC      + "of networkx checks connectedness of G.\n")
   
    end_time = time.time()

    print(bcolors.ENDC  + "\t===================================================")
    print(bcolors.ENDC  + "\tYOUR OWN computation of ONE-SHOT Girvan-Newman clustering for a graph with",G.number_of_nodes(),"nodes and",G.number_of_edges(),"links. "
                        + "Computation time =", end_time - start_time,"\n")
    #turn inside sets into tuples
    comm_tuples = [tuple(x) for x in connected_components]
    print(comm_tuples)
    print(f"number of connected components: {len(comm_tuples)}")
    return comm_tuples

######################################################################################################################
# ...(b) USE NETWORKX IMPLEMENTATION OF ONE-SHOT GN-ALGORITHM...
######################################################################################################################
def use_nx_girvan_newman_for_communities(G, graph_layout, node_positions):

    print(bcolors.ENDC + "\tCalling routine " + bcolors.HEADER + "_use_nx_girvan_newman_for_communities(G,graph_layout,node_positions)" + bcolors.ENDC +"\n")

    print(bcolors.ENDC + "\t" + '''
        ######################################################################################################################
        # USE THE BUILT-IN ROUTINE OF NETWORKX FOR CREATING K+1 COMMUNITIES FOR A GRAPH WITH K CONNECTED COMPONENTS, WHERE 
        # THE GIANT COMPONENT IS SPLIT IN TO COMMUNITIES, BY REPEATEDLY REMOVING MAX_BETWEENNESS EDGES, AS SUGGESTED BY THE 
        # GIRVAN-NEWMAN ALGORITHM
        # 
        # INPUT: 
        #   G = the constructed graph to work with. 
        #   graph_layout = a string determining how to position the graph nodes in the plane.
        #   node_positions = a list of previously computed node positions in the plane (for next call of my_graph_plot_routine)
        # OUTPUT: 
        #    community_tuples = a list of K+1 tuples of node_IDs, each tuple determining a different community
        # PSEUDOCODE:    
        #   ...simple...
        ######################################################################################################################
''')

    start_time = time.time()

    # girvan_newman algorithm for communities
    comm = nx.algorithms.community.girvan_newman(G)
    # returns iterator that we slice to get the communities
    # in this case we only want the first set of communities therefore islice(comm, 1)
    iter_comm = itertools.islice(comm, 1)

    for communities in iter_comm:
        # get the community tuples from the iterator
        comm_tuples = list(tuple(sorted(c)) for c in communities)


    print(bcolors.ENDC  + "USEFUL FUNCTIONS:")
    print(  bcolors.ENDC    + "\t\t" + bcolors.OKCYAN + "girvan_newman(...) " 
            + bcolors.ENDC  + "from networkx.algorithms.community, provides the requiured list of K+1 community-tuples.\n")

    end_time = time.time()

    print(bcolors.ENDC  + "\t===================================================")
    print(bcolors.ENDC  + "\tBUILT-IN computation of ONE-SHOT Girvan-Newman clustering for a graph with",G.number_of_nodes(),"nodes and",G.number_of_edges(),"links. "
                        + "Computation time =", end_time - start_time,"\n")

    return comm_tuples

######################################################################################################################
def divisive_community_detection(G, number_of_divisions, graph_layout, node_positions):

    print(bcolors.ENDC + "\tCalling routine " + bcolors.HEADER + "_divisive_community_detection(G,number_of_divisions,graph_layout,node_positions)" + bcolors.ENDC +"\n")

    print(bcolors.ENDC + "\t" + '''
        ######################################################################################################################
        # CREATE HIERARCHY OF num_divisions COMMUNITIES FOR A GRAPH WITH num_components CONNECTED COMPONENTS, WHERE 
        # MIN(num_nodes / 4, 10*num_components) >= num_divisions >= K. 
        #
        # INPUT: 
        #   G = the constructed graph to work with. 
        #   graph_layout = a string determining how to position the graph nodes in the plane.
        #   node_positions = a list of previously computed node positions in the plane (for next call of my_graph_plot_routine)
        # OUTPUT: 
        #   A list, hierarchy_of_community_tuples, whose first item is the list community_tuples returned by the 
        #   ONE-SHOT GN algorithm, and each subsequent item is a triple of node-tuples: 
        #    * the tuple of the community to be removed next, and 
        #    * the two tuples of the subcommunities to be added for its substitution
        #
        # PSEUDOCODE:   
        #   HIERARCHY = LIST WITH ONE ITEM, THE community_tuple (LIST OF TUPLES) WITH K+1 COMMUNITIES DETERMINED BY THE 
        # ONE-SHOT GIRVAN-NEWMAN ALGORITHM
        #   REPEAT:
        #       ADD TO THE HIERARCHY A TRIPLE: THE COMMUNITY TO BE REMOVED AND THE TWO SUB-COMMUNITIES THAT MUST SUBSTITUTE IT 
        #       * THE COMMUNITY TO BE REMOVED IS THE LARGEST ONE IN THE PREVIOUS STEP.
        #       * THE TWO SUBCOMMUNITIES TO BE ADDED ARE DETERMINED BY REMOVING MAX_BC EDGES FROM THE SUBGRAPH INDUCED BY THE 
        #       COMMUNITY TO BE REMOVED, UNTIL DISCONNECTION. 
        #   UNTIL num_communities REACHES THE REQUIRED NUMBER num_divisions OF COMMUNITIES
        ######################################################################################################################
''')

    start_time = time.time()
    # repeatedly ask the user for input
    while True:
        try:
            percentage = input("input a number from 0.1 to 1.0> ")
            percentage = float(percentage)
            if percentage < 0.1 or percentage > 1.0:
                raise ValueError("input a number from 0.1 to 1.0")
            break
        except ValueError:
            continue

    # select a percentage of nodes from the graph to use in the computation.
    num_nodes = int(percentage * G.number_of_nodes())

    #  HIERARCHY = LIST WITH ONE ITEM, THE community_tuple (LIST OF TUPLES) WITH K+1 COMMUNITIES DETERMINED BY THE
    #  ONE-SHOT GIRVAN-NEWMAN ALGORITHM
    # the hierarchy to be returned.
    hierarchy_of_tuples = []
    # connected components of the graph initialize with girvan_newman algorithm for the first K+1 communities
    connected_components = one_shot_girvan_newman_for_communities(G, graph_layout, node_positions)
    # append a copy of the connected components to the hierarchy as the first item
    hierarchy_of_tuples.append(connected_components.copy())

    while True:
        # check if the number of communities in the hierarchy is equal to the number of divisions
        if len(connected_components) >= number_of_divisions:
            print("broke the outer while loop")
            break

        # find the largest connected component
        largest_connected_component = max(connected_components, key=len)

        # create the subgraph of the largest connected component
        subG = G.subgraph(largest_connected_component).copy()

        while True:
            # find max_betweenness
            big_list_of_nodes = list(subG.nodes) # list of nodes in the subgraph
            random.shuffle(big_list_of_nodes) # shuffle the list of nodes to get a random order
            small_list_of_nodes = big_list_of_nodes[:num_nodes] # get the first num_nodes nodes
            # calculate the betweenness centrality of the nodes in the subgraph
            max_betweenness = nx.edge_betweenness_centrality_subset(subG, small_list_of_nodes, small_list_of_nodes)
            # find the node with the max betweenness
            max_betweenness_edge = max(max_betweenness.items(), key=lambda x: x[1])
            # remove edge with max_betweenness
            subG.remove_edge(max_betweenness_edge[0][0], max_betweenness_edge[0][1])

            # check if subG is disconnected
            if nx.is_connected(subG) == False:
                # find connected components of subG
                new_connected_components = list(nx.connected_components(subG))
                break

        # remove largest connected component from connected components
        # and add new connected components to connected components
        connected_components.remove(largest_connected_component)
        connected_components.append(tuple(new_connected_components[0]))
        connected_components.append(tuple(new_connected_components[1]))
        hierarchy_of_tuples.append([tuple(largest_connected_component), tuple(new_connected_components[0]), tuple(new_connected_components[1])])


    print(bcolors.ENDC  + "\tUSEFUL FUNCTIONS:")

    print(bcolors.ENDC  + "\t\t" 
                        + bcolors.OKCYAN    + "girvan_newman(...) " 
                        + bcolors.ENDC      + "from networkx.algorithms.community, provides TWO communities for a given CONNECTED graph.")

    print(bcolors.ENDC  + "\t\t"
                        + bcolors.OKCYAN    + "G.subgraph(X) " 
                        + bcolors.ENDC      + "extracts the induced subgraph of G, for a given subset X of nodes.") 

    print(bcolors.ENDC  + "\t\t" 
                        + bcolors.OKCYAN    + "community_tuples.pop(GiantCommunityIndex) " 
                        + bcolors.ENDC      + "removes the giant community's tuple (to be split) from the community_tuples data structure.")
    print(bcolors.ENDC  + "\t\t"    
                        + bcolors.OKCYAN    + "community_tuples.append(...) " 
                        + bcolors.ENDC      + "can add the computed subcommunities' tuples, in substitution of the giant component.")    

    end_time = time.time()
    print(bcolors.ENDC  + "\t===================================================")
    print(bcolors.ENDC  + "\tComputation of HIERARCHICAL BIPARTITION of G in communities, "
                        + "using the BUILT-IN girvan-newman algorithm, for a graph with",G.number_of_nodes(),"nodes and",G.number_of_edges(),"links. "
                        + "Computation time =", end_time - start_time,"\n")
    return hierarchy_of_tuples

######################################################################################################################
def determine_opt_community_structure(G, hierarchy_of_community_tuples):
    print(bcolors.ENDC + "\tCalling routine " + bcolors.HEADER + "_determine_opt_community_structure(G,hierarchy_of_community_tuples)" + bcolors.ENDC +"\n")

    print(bcolors.ENDC + "\t" + '''
        ######################################################################################################################
        # GIVEN A HIERARCHY OF COMMUNITY PARTITIONS FOR A GRAPH, COMPUTE THE MODULARITY OF EAC COMMUNITY PARTITION. 
        # RETURN THE COMMUNITY PARTITION OF MINIMUM MODULARITY VALUE
        # INPUT: 
        #   G = the constructed graph to work with. 
        #   hierarchy_of_community_tuples = the output of the DIVISIVE_COMMUNITY_DETECTION routine
        # OUTPUT: 
        #   The partition which achieves the minimum modularity value, within the hierarchy 
        #        #
        # PSEUDOCODE:
        #   Iterate over the HIERARCHY, to construct (sequentially) all the partitions of the graph into communities
        #       For the current_partition, compute its own current_modularity value
        #   IF      current_modularity_vale < min_modularity_value (so far)
        #   THEN    min_partition = current_partition
        #           min_modularity_value = current_modularity_value
        #   RETURN  (min_partition, min_modularity_value)   
        ######################################################################################################################
        ''')
    # iterate over the hierarchy of community tuples, to construct (sequentially)
    # all the partitions of the graph into communities

    # current partition is first entry [K+1 tuples]
    current_partition = hierarchy_of_community_tuples[0].copy()
    # initialize max_modularity_value to the first partition's modularity value
    max_modularity_value = comm.modularity(G,current_partition)
    # initialize the first partition as the optimal partition
    max_partition = current_partition

    # a list for the bar chart of modularity values
    modularity_value_list = []

    # append the first modularity value to the list
    modularity_value_list.append(max_modularity_value)

    # this is not passed in the function this is why we call it a glabal variable
    global node_positions

    for i in range(1,len(hierarchy_of_community_tuples)):
        # remove LC
        current_partition.remove(hierarchy_of_community_tuples[i][0])
        # add LC1 to current partition
        current_partition.append(hierarchy_of_community_tuples[i][1])
        # add LC2 to current partition
        current_partition.append(hierarchy_of_community_tuples[i][2])
        # compute the modularity value of the current partition
        current_modularity_value = comm.modularity(G,current_partition)
        # append the modularity value to the list
        modularity_value_list.append(current_modularity_value)

        # if the current modularity value is greater than the max modularity value,
        # then update the max modularity value
        if current_modularity_value > max_modularity_value:
            max_partition = current_partition
            max_modularity_value = current_modularity_value

    print(f"max_modularity_value: {max_modularity_value}")

    # visualize the communities
    visualize_communities(G, max_partition, "spring", node_positions)
    # create bar diagram of modularity values
    start_of_plot = len(hierarchy_of_community_tuples[0])
    end_of_plot = len(hierarchy_of_community_tuples[0]) + len(modularity_value_list)
    x_list = [value for value in range(start_of_plot, end_of_plot)]
    plt.bar(x_list, modularity_value_list)
    plt.title("Bar plot Modularity Values")
    plt.ylabel("Modularity Value")
    plt.xlabel("Number of Communities (NOTE: starts from K+1)")
    plt.show()
    return max_partition

######################################################################################################################
def add_hamilton_cycle_to_graph(G, node_names_list):

    print(bcolors.ENDC + "\tCalling routine " + bcolors.HEADER + "_add_hamilton_cycle_to_graph(G,node_names_list)" + bcolors.ENDC +"\n")

    print(bcolors.ENDC + "\t" + '''
        ######################################################################################################################
        # GIVEN AN INPUT GRAPH WHICH IS DISCONNECTED, ADD A (HAMILTON) CYCLE CONTAINING ALL THE NODES IN THE GRAPH
        # INPUT: A graph G, and a list of node-IDs.
        # OUTPUT: The augmented graph of G, with a (MAMILTON) cycle containing all the nodes from the node_names_list (in that order).
        #
        # COMMENT: The role of this routine is to add a HAMILTON cycle, i.e., a cycle that contains all the nodes in the graph.
        # Such an operation will guarantee that the graph is connected and, moreover, there is no bridge in the new graph.
        ######################################################################################################################
    ''')
    #add hamilton cycle to graph
    nx.add_cycle(G,node_names_list)
    #my_graph_plot_routine(G, "r", "r", [], 'spring', [])
    return G
    print(  bcolors.ENDC        + "\tUSEFUL FUNCTIONS:") 

    print("\t\t"
            + bcolors.OKCYAN    + "my_graph_plot_routine(G,'grey','blue','solid',graph_layout,node_positions) " 
            + bcolors.ENDC      + "plots the graph with the grey-color for nodes, and (blue color,solid style) for the edges.\n")
    
######################################################################################################################
# ADD RANDOM EDGES TO A GRAPH...
######################################################################################################################
def add_random_edges_to_graph(G, node_names_list, NUM_RANDOM_EDGES, EDGE_ADDITION_PROBABILITY):

    print(  bcolors.ENDC     + "\tCalling routine " 
            + bcolors.HEADER + "_add_random_edges_to_graph(G,node_names_list,NUM_RANDOM_EDGES,EDGE_ADDITION_PROBABILITY)" 
            + bcolors.ENDC   + "\n")

    print(bcolors.ENDC + "\t" + '''
        ######################################################################################################################
        # GIVEN AN INPUT GRAPH WHICH IS DISCONNECTED, ADD A (HAMILTON) CYCLE CONTAINING ALL THE NODES IN THE GRAPH
        # INPUT: A graph G, an integer indicating the max number of random edges to be added per node, and an edge addition 
        # probability, for each random edge considered. 
        # 
        # OUTPUT: The augmented graph of G, containing also  all the newly added random edges.
        #
        # COMMENT: The role of this routine is to add, per node, a number of random edges to other nodes.
        # Create a for-loop, per node X, and then make NUM_RANDOM_EDGES attempts to add a new random edge from X to a 
        # randomly chosen destination (outside its neighborhood). Each attempt will be successful with probability 
        # EDGE_ADDITION_PROBABILITY (i.e., only when a random coin-flip in [0,1] returns a value < EDGE_ADDITION_PROBABILITY).")
        ######################################################################################################################
    ''')
    for node_name in node_names_list:
        # find the neightbors of node_name
        neighbors_of_node_name = set(G.neighbors(node_name))
        # find non-neighbors of node_name
        non_neighbors_of_node_name = set(node_names_list) - neighbors_of_node_name # [x for x in node_names_list if x not in neighbors_of_node_name]
        # remove the node_name from the set of non-neighbors
        non_neighbors_of_node_name = non_neighbors_of_node_name - {node_name}
        for i in range(NUM_RANDOM_EDGES): # NUM_RANDOM_EDGES attempts
            # select a non-neighbor at random with probability EDGE_ADDITION_PROBABILITY to add an edge to it
            if random.random() <= EDGE_ADDITION_PROBABILITY:
                # select a random non-neighbor of node_name
                random_non_neighbor = random.choice(list(non_neighbors_of_node_name))
                # add an edge from node_name to random_non_neighbor
                G.add_edge(node_name,random_non_neighbor)


    print(bcolors.ENDC          + "\tUSEFUL FUNCTIONS:")
    print("\t\t" 
            + bcolors.OKCYAN    + "my_graph_plot_routine(G,'grey','blue','solid',graph_layout,node_positions) " 
            + bcolors.ENDC      + "plots the graph with the grey-color for nodes, and (blue color,solid style) for the edges.\n")

    return G

######################################################################################################################
# VISUALISE COMMUNITIES WITHIN A GRAPH
######################################################################################################################
def visualize_communities(G, community_tuples, graph_layout, node_positions):

    print(bcolors.ENDC      + "\tCalling routine " 
                            + bcolors.HEADER + "_visualize_communities(G,community_tuples,graph_layout,node_positions)" + bcolors.ENDC +"\n")

    print(bcolors.ENDC      + "\tINPUT: A graph G, and a list of lists/tuples each of which contains the nodes of a different community.")
    print(bcolors.ENDC      + "\t\t The graph_layout parameter determines how the " + bcolors.OKCYAN + "my_graph_plot_routine will position the nodes in the plane.")
    print(bcolors.ENDC      + "\t\t The node_positions list contains an existent placement of the nodes in the plane. If empty, a new positioning will be created by the " + bcolors.OKCYAN + "my_graph_plot_routine" + bcolors.ENDC +".\n")
 
    print(bcolors.ENDC      + "\tOUTPUT: Plot the graph using a different color per community.\n")

    print(bcolors.ENDC      + "\tUSEFUL FUNCTIONS:")
    print(bcolors.OKCYAN    + "\t\tmy_random_color_list_generator(number_of_communities)" + bcolors.ENDC + " initiates a list of random colors for the communities.")
    print(bcolors.OKCYAN    + "\t\tmy_graph_plot_routine(G,node_colors,'blue','solid',graph_layout,node_positions)" + bcolors.ENDC + " plots the graph with the chosen node colors, and (blue color,solid style) for the edges.")
    number_of_colors = len(community_tuples)
    random_colors = my_random_color_list_generator(number_of_colors)
    color_dict = {}
    node_colors = []
    # we assign a color to each node in the dictionary (SAME color for every community)
    for i in range(0, number_of_colors):
        for node_name in community_tuples[i]:
            # add to dictionary
            color_dict[node_name] = random_colors[i]

    # just use the dictionary to assign colors for every node in the graph
    for node in G.nodes():
        if node not in color_dict:
            print("Error found a node not in color_dict")
            color_dict[node] = 'grey' # this will never happen but just in case

        node_colors.append(color_dict[node]) # assign color to node
    # plot the graph
    my_graph_plot_routine(G, node_colors, 'blue' , 'solid', graph_layout, node_positions)
########################################################################################
###########################  ROUTINES LIBRARY ENDS ###########################
########################################################################################


########################################################################################
############################# MAIN MENUS OF USER CHOICES ############################### 
########################################################################################

############################### GLOBAL INITIALIZATIONS #################################
G = nx.Graph()                      # INITIALIZATION OF THE GRAPH TO BE CONSTRUCTED
node_names_list = []
node_positions = []                 # INITIAL POSITIONS OF NODES ON THE PLANE ARE UNDEFINED...
community_tuples = []               # INITIALIZATION OF LIST OF COMMUNITY TUPLES...
hierarchy_of_community_tuples = []  # INITIALIZATION OF HIERARCHY OF COMMUNITY TUPLES

G,node_names_list,node_positions = my_menu_graph_construction(G,node_names_list,node_positions)

my_menu_community_detection(G,node_names_list,node_positions,hierarchy_of_community_tuples)