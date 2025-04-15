for build dockerfile with cpu :

docker build -f Dockerfile.cpu -t visualassistantcpu .   
docker run --name visualassistantcpu_container visualassistantcpu

for build dockerfile with gpu :
docker build -t atm-image .
docker run --gpus all -d --name atm-container -p 8000:8000 atm-image

