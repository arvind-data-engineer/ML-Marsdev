import subprocess

def run_scripts():
    # Define the list of Python files to execute
    python_files = [
        'Models/EDA.py',
        'Models/Final_Models.py',
        'Models/app.py'
    ]
    
    # Iterate over each Python file and execute it using subprocess
    for file in python_files:
        subprocess.run(['python', file])

if __name__ == "__main__":
    run_scripts()
