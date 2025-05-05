import time
import subprocess
current_time = time.time()
subprocess.run(["python", "inference_script.py"])
total_time = time.time() - current_time
print(f"Total time taken: {total_time:.3f} seconds")
with open("time.txt", "w") as f:
    f.write(f"{total_time:.3f}")