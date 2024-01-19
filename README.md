# InfraredProduct-API

# How to Deploy Streamlit app on EC2 instance

## 1. Login with your AWS console and launch an EC2 instance

## 2. Run the following commands

### Note: Do the port mapping to this port:- 8000

```bash
sudo apt update
```

```bash
sudo apt-get update
```

```bash
sudo apt upgrade -y
```

```bash
sudo apt install git curl unzip tar make sudo vim wget -y
```



```bash
git clone "Your-repository"
```

```bash
sudo apt install python3-pip
```

```bash
pip3 install -r requirements.txt
```

```bash
#Temporary running
uvicorn main:app --host 0.0.0.0 --reload
```

```bash
#Permanent running
nohup uvicorn main:app --host 0.0.0.0 --reload
```

Note: fastapi runs on this port: 8000
#### Enjoy Coding!
