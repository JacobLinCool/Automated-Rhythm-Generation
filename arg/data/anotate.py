import os
import sys
import subprocess
from glob import glob

data_dir = sys.argv[1]
if not os.path.exists(data_dir):
    print(f"Data directory {data_dir} does not exist.")
    sys.exit(1)

exe = sys.argv[2] if sys.argv[2] else "ryan"

tjas = glob(data_dir + "/**/*.tja", recursive=True)
print("Number of TJAs:", len(tjas))

for i, tja in enumerate(tjas):
    print(i, tja)
    json = tja.replace(".tja", ".json")
    subprocess.run([exe, tja, json])

print("Done.")
