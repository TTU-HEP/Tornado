<div align="center">
  <h1>Tornado Plots for AIgantry Module</h1>
  <p>This guide will walk you through the steps to generate Tornado plots for a given module in the AIgantry repository.</p>
</div>

---

### Step 1: Clone the repository
Start by cloning the AIgantry repository to your local machine.

```
git clone https://github.com/TTU-HEP/Tornado
cd AIgantry
```

### Step 2: Create and activate a Conda environment
Next, create a new Conda environment for this project with Python 3.9 and install the required dependencies.

```
conda create -n aienv python=3.9 -y
conda activate aienv
pip install -r requirements.txt
```

### Step 3: Add Modules and Images
Create a directory for the modules, then add the modules with their associated images.

```
mkdir Modules
cd Modules
# Add your module files and images here!
cd ..
```

### Step 4: Configure the ArrowPlot Script
Before generating the Tornado plots, open the `ArrowPlotScript.py` file and set the path to the Modules directory.

```
nano ArrowPlotScript.py
```

Inside the script, locate the section where the path is defined and modify it as needed:

```
base_dir = os.path.join('/home/akshriva/AIgantry/Tornado/github_copy', 'Modules')
```

Save the changes and exit the editor.

### Step 5: Generate the Tornado Plots
To generate the Tornado plots, follow these steps:

1. **Run the detection script**:  
   This script will ask for the path to the directory. Enter the name of the module directory you created in Step 3.

```
python3 run_detect.py
```

2. **Run the Tornado plot script**:

```
python3 ArrowPlotScript.py
```

---

