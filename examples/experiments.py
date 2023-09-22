import importlib
from Control_ClothPush import main as experiment

# Replace 'your_module_name' with the name of your module without the '.py' extension
module_name = 'Control_ClothPush'

def main():
    # num_runs = 5  # Change this to the number of times you want to run your function
    
    # for run in range(num_runs):
        # Reload the module to start fresh
        
        
        # Call your function here
    tasks = 4
    demos = 2
    methods = ['rrt_star', 'rrt_star_refine', 'p2p']
    for di in range(demos):
        for ti in range(tasks):
            for mi in methods:
                importlib.reload(importlib.import_module(module_name))
                if mi == 'rrt_star':
                    experiment(demo_id=di, task_id=ti, use_rrt_star=True, refinemment=False)
                elif mi == 'rrt_star_refine':
                    experiment(demo_id=di, task_id=ti, use_rrt_star=True, refinemment=True)
                elif mi == 'p2p':
                    experiment(demo_id=di, task_id=ti, use_rrt_star=False, refinemment=False)
                # pyflex.clean()
        
if __name__ == "__main__":
    main()
