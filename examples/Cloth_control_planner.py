import ompl.control as oc
import numpy as np
import ompl.base as ob
from ompl.base import State
from math import sin, cos
from functools import partial
# import boost.python
import matplotlib.pyplot as plt

try:
    # graph-tool and py-OMPL have some minor issues coexisting with each other.  Both modules
    # define conversions to C++ STL containers (i.e. std::vector), and the module that is imported
    # first will have its conversions used.  Order doesn't seem to matter on Linux,
    # but on Apple, graph_tool will not be imported properly if OMPL comes first.
    import graph_tool.all as gt
    graphtool = True
except ImportError:
    print('Failed to import graph-tool.  PlannerData will not be analyzed or plotted')
    graphtool = False

class CustomControlSampler(oc.ControlSampler):
    def __init__(self, control_space, filename):
        super().__init__(control_space)
        self.filename = filename
        self.controls = self.load_controls()
        
    def sample(self, control):
        # randomly sample a control from the loaded controls
        idx = np.random.randint(len(self.controls))
        control[0] = self.controls[idx][0]
        control[1] = self.controls[idx][1]
        control[2] = self.controls[idx][2]
        control[3] = self.controls[idx][3]
        control[4] = self.controls[idx][4]
        control[5] = self.controls[idx][5]
        control[6] = self.controls[idx][6]
        control[7] = self.controls[idx][7]
        control[8] = self.controls[idx][8]
        # print("the sampled control: ", control)
    
    
    # def sampleNext(self, control, current, duration):
    #     # sample a control based on the current state and duration
    #     print("current state: ", current.getX(), current.getY(), current.getYaw())
    #     self.sample(control)
    #     # print("the sampled control: ", control)
    
    def canSampleNext(self):
        # the sampler can always sample the next control
        return True
    
    def load_controls(self):
        # load the controls from the file
        with open(self.filename, 'r') as f:
            lines = f.readlines()
        
        controls = []
        for line in lines:
            values = [float(x) for x in line.strip().split()]
            controls.append(values)
        
        return controls

class MyDecomposition(oc.GridDecomposition):
    def __init__(self, length, bounds):
        super(MyDecomposition, self).__init__(length, 2, bounds)
    def project(self, s, coord):
        coord[0] = s.getX()
        coord[1] = s.getY()
    def sampleFullState(self, sampler, coord, s):
        sampler.sampleUniform(s)
        s.setXY(coord[0], coord[1]) 

class MyControlSampler(oc.ControlSampler):
    def __init__(self, space, filename):
        super(MyControlSampler, self).__init__(space)
        self.space_ = space
        self.filename = filename
        self.controls = self.load_controls()

    def sample(self, control, state):
        # randomly sample a control from the loaded controls
        idx = np.random.randint(len(self.controls))
        control.setValues(self.controls[idx])

    def load_controls(self):
        # load the controls from the file
        with open(self.filename, 'r') as f:
            lines = f.readlines()
        
        controls = []
        for line in lines:
            values = [float(x) for x in line.strip().split()]
            controls.append(values)
        
        return controls
    
def myControlSamplerAllocator(space):
    # Create and return a new instance of a MyControlSampler object
    return CustomControlSampler(space, "controls.txt")

# Wrap the custom control sampler allocator in a std::function object
# sampler_alloc = boost.python.make_function(myControlSamplerAllocator)

def isStateValid(spaceInformation, state):
    # perform collision checking or check if other constraints are
    # satisfied
    return spaceInformation.satisfiesBounds(state)

# control: delta_x, delta_y, delta_theta, push_x, push_y, push_ori, trans_x, trans_y, rot
def propagate(start, control, duration, state):
    # duration is always 1
    # theta = np.deg2rad(start.getYaw())
    theta = start.getYaw()
    ctl_x = control[0]*cos(theta) - control[1]*sin(theta)
    ctl_y = control[0]*sin(theta) + control[1]*cos(theta)
    state.setX(start.getX() + ctl_x)
    state.setY(start.getY() + ctl_y)
    state.setYaw(start.getYaw() + np.deg2rad(control[2]))


def useGraphTool(pd):
    # Extract the graphml representation of the planner data
    graphml = pd.printGraphML()
    f = open("graph.graphml", 'w')
    f.write(graphml)
    f.close()

    # Load the graphml data using graph-tool
    graph = gt.load_graph("graph.graphml", fmt="xml")
    edgeweights = graph.edge_properties["weight"]

    # Write some interesting statistics
    avgdeg, stddevdeg = gt.vertex_average(graph, "total")
    avgwt, stddevwt = gt.edge_average(graph, edgeweights)

    print("---- PLANNER DATA STATISTICS ----")
    print(str(graph.num_vertices()) + " vertices and " + str(graph.num_edges()) + " edges")
    print("Average vertex degree (in+out) = " + str(avgdeg) + "  St. Dev = " + str(stddevdeg))
    print("Average edge weight = " + str(avgwt)  + "  St. Dev = " + str(stddevwt))

    _, hist = gt.label_components(graph)
    print("Strongly connected components: " + str(len(hist)))

    # Make the graph undirected (for weak components, and a simpler drawing)
    graph.set_directed(False)
    _, hist = gt.label_components(graph)
    print("Weakly connected components: " + str(len(hist)))

    # Plotting the graph
    gt.remove_parallel_edges(graph) # Removing any superfluous edges

    edgeweights = graph.edge_properties["weight"]
    colorprops = graph.new_vertex_property("string")
    vertexsize = graph.new_vertex_property("double")

    start = -1
    goal = -1

    for v in range(graph.num_vertices()):

        # Color and size vertices by type: start, goal, other
        if pd.isStartVertex(v):
            start = v
            colorprops[graph.vertex(v)] = "cyan"
            vertexsize[graph.vertex(v)] = 10
        elif pd.isGoalVertex(v):
            goal = v
            colorprops[graph.vertex(v)] = "green"
            vertexsize[graph.vertex(v)] = 10
        else:
            colorprops[graph.vertex(v)] = "yellow"
            vertexsize[graph.vertex(v)] = 5

    # default edge color is black with size 0.5:
    edgecolor = graph.new_edge_property("string")
    edgesize = graph.new_edge_property("double")
    for e in graph.edges():
        edgecolor[e] = "black"
        edgesize[e] = 0.5

    # using A* to find shortest path in planner data
    if start != -1 and goal != -1:
        _, pred = gt.astar_search(graph, graph.vertex(start), edgeweights)

        # Color edges along shortest path red with size 3.0
        v = graph.vertex(goal)
        while v != graph.vertex(start):
            p = graph.vertex(pred[v])
            for e in p.out_edges():
                if e.target() == v:
                    edgecolor[e] = "red"
                    edgesize[e] = 2.0
            v = p

    # Writing graph to file:
    # pos indicates the desired vertex positions, and pin=True says that we
    # really REALLY want the vertices at those positions
    gt.graph_draw(graph, vertex_size=vertexsize, vertex_fill_color=colorprops,
                  edge_pen_width=edgesize, edge_color=edgecolor,
                  output="graph.png")
    print('\nGraph written to graph.png')

    
def visualize_path(path):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the nodes as points
    x = []
    y = []
    z = []
    for node, _ in path:
        x.append(node['index_x'])
        y.append(node['index_y'])
        z.append(node['layer'])

    ax.scatter(x, y, z)

    colors = ["red", "green", "blue", "yellow", "purple", "orange", "pink", "brown", "gray", "olive", "cyan", "magenta"]
    c_id = 0

    last_push_pose = None
    # Plot edges
    for i in range(len(path) - 1):
        node, _ = path[i]
        next_node, _ = path[i + 1]

        # current_push_pose = (action['push_x'], action['push_y'], action['push_ori'])

        # Change color if push_pose has changed
        # if last_push_pose is not None and current_push_pose != last_push_pose:
        #     c_id += 1

        ax.plot([node['index_x'], next_node['index_x']], [node['index_y'], next_node['index_y']], [node['layer'], next_node['layer']], color=colors[c_id])
        # last_push_pose = current_push_pose

    # ax.view_init(elev=15., azim=-75)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Layer (Orientation)')

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-180, 180)

    plt.show()


def plot_planner_data(planner_data, start, goal):
    # plot the vertices
    for i in range(planner_data.numVertices()):
        vertex = planner_data.getVertex(i)
        state = vertex.getState()
        plt.plot(state.getX(), state.getY(), 'bo')

    # plot the edges
    for i in range(planner_data.numEdges()):
        edge = planner_data.getEdge(i)
        start_state = planner_data.getVertex(edge.getSource()).getState()
        end_state = planner_data.getVertex(edge.getTarget()).getState()
        plt.plot([start_state.getX(), end_state.getX()], [start_state.getY(), end_state.getY()], 'b')

    # plot the start and goal states
    plt.plot(start.getX(), start.getY(), 'go')
    plt.plot(goal.getX(), goal.getY(), 'ro')

    # show the plot
    plt.show()

def plan(node_s, node_t, planner="RRT"):
    # construct the state space we are planning in
    space = ob.SE2StateSpace()

    # set the bounds for the R^2 part of SE(2)
    bounds = ob.RealVectorBounds(2)
    bounds.setLow(-0)
    bounds.setHigh(1)
    space.setBounds(bounds)

    # create a control space
    cspace = oc.RealVectorControlSpace(space, 9)

    # set the bounds for the control space
    cbounds = ob.RealVectorBounds(9)
    cbounds.setLow(-1000.0)
    cbounds.setHigh(1000.0)
    cspace.setBounds(cbounds)

    cspace.setControlSamplerAllocator(oc.ControlSamplerAllocator(myControlSamplerAllocator))

    # define a simple setup class
    ss = oc.SimpleSetup(cspace)
    ss.setStateValidityChecker(ob.StateValidityCheckerFn( \
        partial(isStateValid, ss.getSpaceInformation())))
    ss.setStatePropagator(oc.StatePropagatorFn(propagate))

    # create a start state
    start = ob.State(space)
    start().setX(node_s[0])
    start().setY(node_s[1])
    start().setYaw(np.deg2rad(node_s[2]))

    # create a goal state
    goal = ob.State(space)
    goal().setX(node_t[0])
    goal().setY(node_t[1])
    goal().setYaw(np.deg2rad(node_t[2]))

    # set the start and goal states
    ss.setStartAndGoalStates(start, goal, 0.1)

    # (optionally) set planner
    si = ss.getSpaceInformation()
    si.setMinMaxControlDuration(1, 1)
    planner = oc.SST(si)
    # if planner == "RRT":
    # planner = oc.RRT(si)
    # elif planner == "EST":
    # planner = oc.EST(si)
    # planner = oc.KPIECE1(si) # this is the default
    # planner = oc.KPIECE1(si)
    # SyclopEST and SyclopRRT require a decomposition to guide the search
    # decomp = MyDecomposition(32, bounds)
    # planner = oc.SyclopEST(si, decomp)
    # planner = oc.SyclopRRT(si, decomp)
    ss.setPlanner(planner)
    # (optionally) set propagation step size
    si.setPropagationStepSize(1)

    # attempt to solve the problem
    solved = ss.solve(3)

    if solved:
        # print the path to screen
        print("Found solution:\n%s" % ss.getSolutionPath().printAsMatrix())

        # get the planner data
        planner_data = oc.PlannerData(si)
        ss.getPlannerData(planner_data)

        # plot the planner data
        # plot_planner_data(planner_data, start, goal)


        states = ss.getSolutionPath().getStates()
        controls = ss.getSolutionPath().getControls()
        # waypoints = []
        # actions = []
        path = []

        

        for i, state in enumerate(states):
            # print("x:", state.getX())
            # print("y:", state.getY())
            # print("yaw:", state.getYaw())
            # if i != 0:
            #     print("Control: ", controls[i-1][0], controls[i-1][1], controls[i-1][2],
            #                         controls[i-1][3], controls[i-1][4], controls[i-1][5],
            #                         controls[i-1][6], controls[i-1][7], controls[i-1][8])
            # rs = State(state)
            # print("length of first dim: ", len(rs))
            # st = [state.getX(), state.getY(), np.rad2deg(state.getYaw())]
            # ac = [controls[i-1][0], controls[i-1][1], controls[i-1][2],
            #         controls[i-1][3], controls[i-1][4], controls[i-1][5],
            #         controls[i-1][6], controls[i-1][7], controls[i-1][8]]
            # print("State: ", st)
            # waypoints.append(st)
            # actions.append(ac)

            node = {
                "index_x": state.getX(),
                "index_y": state.getY(),
                "layer": np.rad2deg(state.getYaw())
            }
            
            if i != 0:
                action = {
                    'push_x': controls[i-1][3],
                    'push_y': controls[i-1][4],
                    'push_ori': controls[i-1][5],
                    'trans_x': controls[i-1][6],
                    'trans_y': controls[i-1][7],
                    'rot': controls[i-1][8],
                    'delta_x': controls[i-1][0],
                    'delta_y': controls[i-1][1],
                    'delta_r': controls[i-1][2]
                }
            else:
                action = None

            path.append((node, action))
        

        # Extracting planner data from most recent solve attempt
        # pd = ob.PlannerData(ss.getSpaceInformation())
        # ss.getPlannerData(pd)

        # # Computing weights of all edges based on state space distance
        # pd.computeEdgeWeights()

        # if graphtool:
        #     useGraphTool(pd)

        # visualize_path(waypoints, 2, 2)

        return path
    
    return None

        


if __name__ == "__main__":
    start = [0, 0, 0]
    goal = [0, 0, 180]
    # control: delta_x, delta_y, delta_theta, push_x, push_y, push_ori, trans_x, trans_y, rot
    path = plan(start, goal)