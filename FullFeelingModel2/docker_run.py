import os
import subprocess

print("Введите абсолютный путь до папки с изображениями:\n")
path_of_the_volume = input()

f = open("container_name.txt")
container_name = f.read()
f.close()

os.system("docker run -m=4g -v "+path_of_the_volume+":c:/my_volume_input " + container_name)

print(path_of_the_volume)