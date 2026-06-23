import subprocess
import sys


def run_scripts():
    python_files = [
        "Models/EDA.py",
        "Models/Final_Models.py",
    ]

    for file in python_files:
        subprocess.run([sys.executable, file], check=True)

    subprocess.run([sys.executable, "Models/app.py"], check=True)


if __name__ == "__main__":
    run_scripts()
