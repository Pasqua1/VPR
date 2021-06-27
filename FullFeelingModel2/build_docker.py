import os
import subprocess

path = os.getcwd()
docker_build_command =str("docker build -f "+str(path)+"\Dockerfile "+str(path))

stream = os.popen(docker_build_command)
output = stream.read()
container_name = output.split()[-1]
f = open("container_name.txt", "w")
f.write(container_name)
f.close()