import subprocess
import sys
import time

run_q2_1 = True
run_q2_2 = True

# Save the current standard output
original_stdout = sys.stdout

# Specify the file to redirect the output
with open('hw2-q2-stdout.txt', 'w') as f:
    # Redirect standard output to the file
    sys.stdout = f

    if run_q2_1:
        learning_rates = [0.1, 0.01, 0.001]

        print("Running Question 2.1 ..." + "\n")

        for lr in learning_rates:
            command = [
                'python',
                'hw2-q2.py',
                '-epochs', '15',
                '-learning_rate', str(lr),
                '-optimizer', 'sgd'
            ]

            try:
                start = time.time()
                subprocess.run(command, check=True, stdout=f)
                end = time.time()
                print(f"Time taken for learning rate {lr}: {end - start} seconds")
            except subprocess.CalledProcessError as e:
                print(f"Error running command: {e}")

    if run_q2_2:
        learning_rates = [0.1, 0.01, 0.001]

        print("Running Question 2.2 ..." + "\n")

        for lr in learning_rates:
            command = [
                'python',
                'hw2-q2.py',
                '-epochs', '15',
                '-learning_rate', str(lr),
                '-optimizer', 'sgd',
                '-no_maxpool'
            ]

            try:
                start = time.time()
                subprocess.run(command, check=True, stdout=f)
                end = time.time()
                print(f"Time taken for learning rate {lr}: {end - start} seconds")
            except subprocess.CalledProcessError as e:
                print(f"Error running command: {e}")

    # Restore the standard output
    sys.stdout = original_stdout
