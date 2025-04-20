import setproctitle
import subprocess

# Set a believable process title
setproctitle.setproctitle('inference')

# Run uvicorn
subprocess.run(["python", "-m", "uvicorn", "main:app", "--host", "localhost", "--port", "8000"])