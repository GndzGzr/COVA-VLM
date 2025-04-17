for build dockerfile with cpu :

docker build --platform linux/amd64 -f Dockerfile.cpu -t visualassistantcpu .   
docker run -p 8000:8000 --name visualassistancpucontainer visualassistantcpu 

for build dockerfile with gpu :
docker build --platform linux/amd64 -f Dockerfile.gpu -t atm-image .
docker run --gpus all -d --name atm-container -p 8000:8000 atm-image

