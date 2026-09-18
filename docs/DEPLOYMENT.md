# EC2 Deployment Guide

## Manual Deployment to EC2 Server

### Prerequisites

-   EC2 instance running (3.108.59.104)
-   SSH/PuTTY access as ec2-user
-   Python 3.8+ installed on EC2
-   Required system packages

### Step 1: Connect to EC2 Server

```bash
# Using PuTTY or SSH
ssh ec2-user@3.108.59.104
```

### Step 2: Create Directory Structure

```bash
# Create the revival/agent directory
sudo mkdir -p /home/ec2-user/revival/agent
cd /home/ec2-user/revival/agent

# Set proper ownership
sudo chown -R ec2-user:ec2-user /home/ec2-user/revival
```

### Step 3: Manual File Transfer ✅ COMPLETED with WinSCP

**✅ Files transferred successfully using WinSCP**, excluding:

-   `__pycache__/` folders
-   `.venv/` folder
-   Any `.pyc` files

**For future reference - Transfer options:**

#### Option A: Using SCP (if available)

**Note:** SCP doesn't support exclude patterns directly, so use one of these approaches:

**Option A1: Create temporary clean copy and transfer**

```bash
# From your local Windows machine (using WSL or Git Bash)
# First create a clean copy without excluded folders
mkdir temp_deploy
cp -r "d:\Workspace\Python\SchoolChatApp\code\Python\*" temp_deploy/
find temp_deploy -name "__pycache__" -type d -exec rm -rf {} +
rm -rf temp_deploy/.venv temp_deploy/venv

# Then transfer the clean copy
scp -r temp_deploy/* ec2-user@3.108.59.104:/home/ec2-user/revival/agent/

# Clean up
rm -rf temp_deploy
```

#### Using WinSCP or FileZilla

1. Connect to `3.108.59.104` with username `ec2-user`
2. Navigate to `/home/ec2-user/revival/agent/`
3. Upload all files from `d:\Workspace\Python\SchoolChatApp\code\Python\` except:
    - `__pycache__` folders
    - `.venv` folder

#### Option B: Using rsync (RECOMMENDED - Best for exclusions)

### Step 4: Setup Python Environment on EC2

```bash
# Navigate to the application directory
cd /home/ec2-user/revival/agent

# Install Python pip if not available
sudo yum update -y
sudo yum install python3-pip -y

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Test the application
python start.py
```

### Step 5: Create Systemd Service

Create a service file for auto-start:

```bash
# Create service file
sudo nano /etc/systemd/system/revival-agent.service
```

Add the following content to the service file:

```ini
[Unit]
Description=Revival AI Agent Service
After=network.target

[Service]
Type=simple
User=ec2-user
Group=ec2-user
WorkingDirectory=/home/ec2-user/revival/agent
Environment=PATH=/home/ec2-user/revival/agent/.venv/bin
ExecStart=/home/ec2-user/revival/agent/.venv/bin/python /home/ec2-user/revival/agent/start.py
Restart=always
RestartSec=3
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

### Step 6: Enable and Start Service

```bash
# Reload systemd daemon
sudo systemctl daemon-reload

# Enable service to start on boot
sudo systemctl enable revival-agent.service

# Start the service
sudo systemctl start revival-agent.service

# Check service status
sudo systemctl status revival-agent.service
```

### Step 7: Service Management Commands

#### Start the Service

```bash
sudo systemctl start revival-agent.service
```

#### Stop the Service

```bash
sudo systemctl stop revival-agent.service
```

#### Restart the Service

```bash
sudo systemctl restart revival-agent.service
```

#### Check Service Status

```bash
sudo systemctl status revival-agent.service
```

#### View Service Logs

```bash
# View recent logs
sudo journalctl -u revival-agent.service -f

# View logs from today
sudo journalctl -u revival-agent.service --since today

# View last 100 lines
sudo journalctl -u revival-agent.service -n 100
```

#### Disable Auto-start (if needed)

```bash
sudo systemctl disable revival-agent.service
```

### Step 8: Firewall Configuration (if needed)

```bash
# If your app runs on a specific port (e.g., 8000), open it
sudo firewall-cmd --permanent --add-port=8000/tcp
sudo firewall-cmd --reload

# Or disable firewall temporarily for testing
sudo systemctl stop firewalld
```

### Step 9: Environment Variables (if needed)

If your application needs environment variables, create a `.env` file:

```bash
cd /home/ec2-user/revival/agent
nano .env
```

Add your environment variables:

```bash
OPENAI_API_KEY=your_key_here
PINECONE_API_KEY=your_key_here
DATABASE_URL=your_db_url_here
```

### Troubleshooting

#### Check if service is running

```bash
ps aux | grep start.py
```

#### Check service logs for errors

```bash
sudo journalctl -u revival-agent.service --no-pager
```

#### Check port usage

```bash
sudo netstat -tlnp | grep :8000
```

#### Manually test the application

```bash
cd /home/ec2-user/revival/agent
source .venv/bin/activate
python start.py
```

#### Update the application

1. Stop the service: `sudo systemctl stop revival-agent.service`

2. find . -name "**pycache**" -type d -exec rm -rf {} + 2>/dev/null || true
3. Replace files manually (excluding **pycache** and .venv)
4. Start the service: `sudo systemctl start revival-agent.service`

### Security Notes

-   Ensure your EC2 security group allows traffic on the required ports
-   Keep your API keys secure in environment variables
-   Regularly update the system packages: `sudo yum update -y`
-   Monitor service logs for any issues

### File Structure on EC2

```
/home/ec2-user/revival/agent/
├── start.py
├── app.py
├── requirements.txt
├── agents/
├── api/
├── dal/
├── documents/
├── lib/
├── services/
├── tools/
├── .venv/          # Created during setup
└── .env            # Optional environment file
```

## remove cache

find . -name "**pycache**" -type d -exec rm -rf {} + 2>/dev/null || true

## Quick Command Reference

| Action             | Command                                        |
| ------------------ | ---------------------------------------------- |
| Start Service      | `sudo systemctl start revival-agent.service`   |
| Stop Service       | `sudo systemctl stop revival-agent.service`    |
| Restart Service    | `sudo systemctl restart revival-agent.service` |
| Check Status       | `sudo systemctl status revival-agent.service`  |
| View Logs          | `sudo journalctl -u revival-agent.service -f`  |
| Enable Auto-start  | `sudo systemctl enable revival-agent.service`  |
| Disable Auto-start | `sudo systemctl disable revival-agent.service` |
