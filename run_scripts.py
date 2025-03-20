import subprocess
scripts = ["create_datasets.py", "kfold_test.py", "sklearn_trainer.py"]

for script in scripts:
    try:
        print(f"running {script}")
        subprocess.run(["python", script], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error while runnign {script}: {e}")