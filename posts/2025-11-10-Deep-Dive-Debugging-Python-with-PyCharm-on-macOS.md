---
title: "Deep-Dive: Debugging Python with PyCharm on macOS (Local & Docker) 🐍"
date: "2025-11-10T06:31:30+00:00"
author: "gp"
layout: "post"
image: "/content/2025/11/debugging-python1.png"
categories: [Python, debugger]
published: true
mermaid: true
---

If you're working with PyCharm, Docker, and Python daily, your debugger is either your best friend or that powerful tool you know exists but haven't fully mastered. This guide will change that.



This is a comprehensive technical guide to debugging Python applications with **PyCharm 2025.2 Professional** on **macOS** using **Docker Desktop**. We'll cover:

- ✅ Local debugging with conda environments
- ✅ Remote debugging inside Docker containers
- ✅ Docker Compose-based debugging workflows
- ✅ Attaching to running containers with Python Debug Server

We'll use a real-world FastAPI application (a DateTime API with timezone conversions and business hours logic) as our example throughout this guide.

## Prerequisites

- macOS (Sonoma or later)
- Docker Desktop installed and running
- PyCharm Professional 2025.2+ (Community Edition has limited Docker debugging support)
- Conda/Miniconda installed

> PyCharm Professional is required for Docker remote interpreters and Docker Compose integration. Community Edition doesn't support these features in the same way.
{: .prompt-info }

---

## 1. Understanding PyCharm's Debugging Architecture 🏗️

Before diving into configurations, let's understand the three core components:

### 1.1 Python Interpreter
The Python environment that runs your code. This can be:
- **Local**: conda environment, venv, or system Python
- **Remote**: Docker container, Docker Compose service, SSH remote machine, or WSL

### 1.2 Run/Debug Configuration
A named recipe that specifies:
- Which script or module to run
- Which interpreter to use
- Working directory, environment variables, parameters
- Port mappings and path mappings (for Docker)

### 1.3 Debugger
PyCharm uses `pydevd` (Python Debugger Daemon), a powerful debugging engine that:
- Suspends program execution at breakpoints
- Allows you to inspect variables and call stacks
- Supports step-through debugging, watches, and expression evaluation
- Works seamlessly with both local and remote interpreters

**In Docker scenarios**, PyCharm either:
1. Runs your code inside a container via a Docker-based interpreter with the debugger attached automatically, OR
2. Attaches to a running process inside a container via `pydevd-pycharm` and `settrace()`

---

## 2. Local Debugging Setup: Getting the Foundation Right 🎯
First thing you need to clone the sample repo we will use, you can find it on my [GitHub]
Let's start with local debugging using conda. If this isn't solid, Docker debugging will only amplify the pain.

### 2.1 Setting Up Your Conda Environment

Our example project includes an `environment.yml` file:

```yaml
name: debugging-python
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.11
  - pip
  - pip:
      - fastapi==0.104.1
      - uvicorn[standard]==0.24.0
      - pydantic==2.5.0
      - pydevd-pycharm~=242.23726  # For remote debugging
```

Create the environment from your terminal:

```bash
# Navigate to project root
cd /path/to/DebuggingPython 
```
```bash
# Create the conda environment
conda env create -f environment.yml
```
```bash
# Activate it
conda activate debugging-python
```

### 2.2 Configuring PyCharm to Use Conda

1. Open your project in PyCharm
2. Open **Settings** (`⌘,` on macOS)
3. Navigate to **Project: debugging-python → Python Interpreter**
4. Click the gear icon ⚙️ → **Add Interpreter** → **Add Local Interpreter**
5. Select **Conda Environment** tab
6. Choose **Existing environment**
7. Navigate to your conda environments folder:
   - Default location: `~/miniconda3/envs/debugging-python/bin/python` or `~/anaconda3/envs/debugging-python/bin/python`
8. Click **OK**

PyCharm will now use this interpreter for:
- Running and debugging
- Installing packages
- Code completion and inspections


### 2.3 Creating a Python Run/Debug Configuration

Never use "Run file in console" for serious debugging. Always create named configurations.

1. Go to **Run → Edit Configurations...**
2. Click **+** → **Python**
3. Configure as follows:

**Configuration Name**: `debugging-python-local`

| Field | Value |
|-------|-------|
| **Script path** | Leave empty (we'll use module) |
| **Module name** | `uvicorn` |
| **Parameters** | `main:app --reload --log-level debug` |
| **Working directory** | `/path/to/debugging-python/src` |
| **Python interpreter** | `debugging-python` (your conda env) |
| **Environment variables** | (empty for now, add as needed) |

4. Click **OK**

![Python Run/Debug Configuration](/content/2025/11/debugging-python2.png){: width="500" height="800" }
_Python Run/Debug Configuration_


### 2.4 Using the Debugger Like a Pro 🔍

Now for the fun part. Let's debug our DateTime API.

#### Setting Breakpoints

Open `src/main.py` and click in the gutter (left margin) next to line 74:

```python
local_time, timezone_name = _convert_to_timezone(now_utc, tz)
```

A red dot appears. This is your breakpoint.

#### Starting a Debug Session

1. Click the **Debug** icon next to your configuration dropdown
![Debug Icon](/content/2025/11/debugging-python3.png)
_Debug Icon_

2. Or: **Run → Debug 'datetime-api-local'**
3. The Debug tool window opens at the bottom

#### Making a Request

Open another terminal and make a request:

```bash
curl "http://localhost:8000/datetime?tz=EST"
```

**PyCharm immediately suspends execution at line 74!** 🎉

#### The Debug Window

You'll see several tabs:

**Debugger Tab**:
- **Frames**: Call stack showing how you got here
- **Variables**: All local variables in the current scope
  - Expand `tz` → you'll see `"EST"`
  - Expand `now_utc` → see the datetime object internals

**Console Tab**:
- Interactive Python console in the current debugging context
- Try: `now_utc.timestamp()` and press Enter

#### Stepping Through Code

Use the stepping toolbar at the top of the Debug window:

| Button | Shortcut | Action |
|--------|----------|--------|
| **Step Over** | `F8` | Execute current line, move to next line |
| **Step Into** | `F7` | Go into the function call |
| **Step Into My Code** | `⌥⇧F7` | Like Step Into, but skips library code |
| **Step Out** | `⇧F8` | Finish current function and return to caller |
| **Resume** | `⌘⌥R` | Continue until next breakpoint |

Let's practice:
1. Press `F7` (Step Into) at line 74
2. You'll jump into the `_convert_to_timezone()` function at line 199
3. Hover over variables to see their values inline
4. Press `F8` repeatedly to step through the conversion logic
5. Watch `offset_hours` get assigned from `TIMEZONE_OFFSETS`
6. Watch `local_dt` get calculated

> To use F1, F2, etc. keys as standard function keys in System Settings → Keyboard → Keyboard Shortcuts → Function Keys. 
> Temporarily invert behavior by holding fn / globe while pressing the keys.
> ![F-Keys](/content/2025/11/debugging-python4.png)
{: .prompt-tip }

#### Advanced Debugging Features

**Watches**: Add expressions to track across frames
1. Right-click in the Variables pane → **New Watch**
2. Enter: `offset_hours * 3600` (offset in seconds)
3. This watch persists across debugging sessions

**Evaluate Expression** (`⌥F8`):
1. While paused at line 213, press `⌥F8`
2. In the debugger switch to the `console` console tab
3. Type: `local_dt.strftime('%Y-%m-%d %H:%M:%S')`
4. Press Enter to see the formatted result

![result](/content/2025/11/debugging-python5.png)
_result_

**Inline Debugging**:
Variables appear as gray italic text next to code lines. Over on it and select `Add as inline watch`
![inline watch](/content/2025/11/debugging-python6.png)
_inline watch_

### 2.5 PyCharm Debugger Settings

Go to **Settings → Build, Execution, Deployment → Debugger → Python**:

**Useful settings**:
- ☑️ **Attach to subprocess automatically**: Essential if your code spawns child processes
- ☑️ **Gevent compatible**: Required for gevent-monkeypatched async code
- ☑️ **PyQt compatible**: If using PyQt/PySide

For pytest debugging:
- ☑️ **Drop into debugger on failed tests**: Automatically breaks at failing assertions

---

## 3. Advanced Local Debugging Patterns 🚀

### 3.1 Debugging Async Code (FastAPI/Uvicorn)

Our FastAPI endpoints are `async`. Set a breakpoint inside an async function:

```python
@app.get("/datetime/convert", response_model=TimezoneConversion)
async def convert_timezones(
    time_str: str = Query(..., description="ISO format datetime string"),
    timezones: str = Query("EST,PST,CET", description="Comma-separated timezone codes"),
):
    logger.debug(f"Converting time {time_str} to timezones: {timezones}")
    # Set breakpoint here ⬅️
```

Test it:
```bash
curl "http://localhost:8000/datetime/convert?time_str=2024-11-21T10:00:00Z&timezones=EST,PST,JST"
```

The debugger works perfectly with async/await. You can step through:
- The await points
- Helper function calls
- Exception handlers

### 3.2 Debugging Complex Logic

The `_check_if_business_hours()` function has interesting branching logic. Set a breakpoint at line 240:

```python
is_business_hours = start_hour <= current_decimal_hour < end_hour
```

Make a request:
```bash
curl "http://localhost:8000/business-hours?tz=EST&start_hour=9&end_hour=17"
```

Now step through and observe:
- How `current_decimal_hour` is calculated
- Which branch executes (before, during, or after business hours)
- How `hours_until_open` or `hours_until_close` is computed

This is where PyCharm's debugger shines—you see exactly how your logic flows.

### 3.3 Conditional Breakpoints

Right-click on a breakpoint → **Edit Breakpoint**:

Add a condition: `tz == "EST"`

Now the breakpoint only triggers for EST timezone requests. Perfect for hunting down timezone-specific bugs.

### 3.4 Exception Breakpoints

Go to **Run → View Breakpoints** (`⌘⇧F8`)

Click **+** → **Python Exception Breakpoints**

Add: `ValueError`

Now if any `ValueError` occurs anywhere in your code, the debugger will break at the exact line that raised it.

Try triggering it:
```bash
curl "http://localhost:8000/datetime/convert?time_str=invalid-date&timezones=EST"
```

PyCharm breaks at line 98 where `ValueError` is raised! 🎯


---

## 4. Docker Integration: Bridging Local and Container 🐳

Now let's move to Docker. First, ensure Docker Desktop is running.

### 4.1 Enable Docker Support in PyCharm

1. **Settings** (`⌘,`) → **Plugins** → Verify **Docker** plugin is enabled (it's bundled with Pro)
2. **Settings → Build, Execution, Deployment → Docker**
3. Click **+** to add a Docker configuration
4. Select **Docker for Mac**
5. Verify **Connection successful** appears

### 4.2 Understanding Virtual Machine Path Mappings

On macOS, Docker Desktop runs in a lightweight VM. By default, it can mount:
- `/Users` (your home directory)
- `/Volumes`
- `/private`
- `/tmp`

Check this at: **Settings → Build, Execution, Deployment → Docker → [your config] → Virtual machine path mappings**

As long as your project is under `/Users/your-username/`, you're good to go.

---

## 5. Method 1: Docker-Based Python Interpreter 🎯

This is the most "PyCharm-native" approach. PyCharm manages the container lifecycle and automatically attaches the debugger.

### 5.1 Build Your Docker Image First

From terminal:
```bash
cd /path/to/debugging-python
docker build -t debugging-python:latest -f src/Dockerfile src/
```

Or use the provided build script:
```bash
chmod +x deployment/build.sh
./deployment/build.sh
```

### 5.2 Create a Docker-Based Interpreter

1. **Settings** (`⌘,`) → **Project: debugging-python → Python Interpreter**
2. Click the gear icon ⚙️ → **Add Interpreter** → **On Docker...**
3. Choose **Pull or use existing**
4. Select your Docker configuration (Docker for Mac)
5. Image name: `debugging-python:latest`
6. Click **Next**

PyCharm will:
- Inspect the image
- Detect Python location
- Create a "target" that represents this Docker environment

7. Click **Create**

### 5.3 Configure Run/Debug for Docker Interpreter

1. **Run → Edit Configurations...**
2. Click **+** → **Python**
3. Name: `datetime-api-docker`

| Field | Value |
|-------|-------|
| **Module name** | `uvicorn` |
| **Parameters** | `main:app --host 0.0.0.0 --port 8000 --reload --log-level debug` |
| **Working directory** | `/app` |
| **Python interpreter** | Select your Docker interpreter (`datetime-api:latest`) |
| **Path mappings** | `<Project root>/src=/app` (PyCharm usually auto-detects this) |
| **Docker container settings** | Click "Modify options" → "Docker container settings" |
| **Port bindings** | `8000:8000` |
| **Volume bindings** | `<Project root>/src:/app` |

4. Click **OK**

### 5.4 Debugging in Docker

Now the magic happens:

1. Set a breakpoint at line 106 in `src/main.py`:
```python
for tz in tz_list:  # Set breakpoint here
```

2. Click **Debug** 🐛 with the `datetime-api-docker` configuration

PyCharm will:
- Start a container from your image
- Mount your source code as a volume
- Run uvicorn inside the container
- Attach the debugger automatically

3. Make a request:
```bash
curl "http://localhost:8000/datetime/convert?time_str=2024-11-21T10:00:00Z&timezones=EST,PST,JST"
```

**PyCharm breaks at your breakpoint inside the Docker container!** 🎉

From your perspective, it looks identical to local debugging, but:
- Your code is running in a Linux container
- The environment matches your production setup
- Dependencies are isolated

### 5.5 Debugging Tips for Docker Interpreter

**Logs**: All container stdout/stderr appears in PyCharm's Debug console

**Environment Variables**: Add them in the Run Configuration's "Environment variables" field

**Volume Sync**: Changes you make in PyCharm are immediately reflected in the container (hot reload works!)

**Performance**: First run is slow (container startup), subsequent debug sessions are faster

---

## 6. Method 2: Docker Compose as Interpreter 🎼

For multi-service applications, Docker Compose is the way to go.

### 6.1 Understanding Our docker-compose.yml

Our project includes `docker-compose.yml`:

```yaml
version: '3.8'

services:
  datetime-api:
    build:
      context: ./src
      dockerfile: Dockerfile
    container_name: datetime-api
    ports:
      - "8000:8000"
    volumes:
      - ./src:/app
    environment:
      - PYTHONUNBUFFERED=1
    command: ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--log-level", "debug"]
```

Key points:
- **volumes**: Maps `./src` → `/app` in the container (live code sync)
- **ports**: Exposes port 8000
- **command**: Explicit uvicorn command with `--reload`

### 6.2 Create Docker Compose Interpreter

1. **Settings** → **Project: codechallenge → Python Interpreter**
2. Click ⚙️ → **Add Interpreter** → **On Docker Compose...**
3. Choose:
   - Server: Docker for Mac
   - Configuration files: `./docker-compose.yml`
   - Service: `datetime-api`
   - Python interpreter path: `/usr/local/bin/python` (or auto-detected)
4. Click **Create**

PyCharm will:
- Parse your `docker-compose.yml`
- Use the `datetime-api` service as the interpreter
- Understand volume mappings automatically

### 6.3 Create Run/Debug Configuration for Compose

1. **Run → Edit Configurations...**
2. Click **+** → **Python**
3. Name: `datetime-api-compose`

| Field | Value |
|-------|-------|
| **Module name** | `uvicorn` |
| **Parameters** | `main:app --host 0.0.0.0 --port 8000 --reload --log-level debug` |
| **Working directory** | `/app` |
| **Python interpreter** | Select the Docker Compose interpreter (`datetime-api` service) |

4. Click **OK**

### 6.4 Debugging with Docker Compose

1. Set a breakpoint at line 140 in the `check_business_hours()` function
2. Click **Debug** 🐛 with `datetime-api-compose`

PyCharm:
- Runs `docker-compose up` with your service
- Starts any dependent services (if you had them)
- Attaches debugger to the `datetime-api` container

3. Test:
```bash
curl "http://localhost:8000/business-hours?tz=PST&start_hour=9&end_hour=17"
```

Debugging works exactly like before, but now:
- Your entire multi-service stack is running
- Dependencies (databases, caches, etc.) are available
- It's closer to your real deployment environment

### 6.5 Debugging Multi-Service Stacks

If you later add a database or message queue:

```yaml
services:
  datetime-api:
    # ... existing config ...
    depends_on:
      - postgres
  
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_USER: dev
      POSTGRES_PASSWORD: devpass
    ports:
      - "5432:5432"
```

PyCharm will bring up both services when you debug. The API can connect to `postgres:5432` inside the Docker network.


---

## 7. Method 3: Attaching to Running Containers with Debug Server 🔌

Sometimes you can't let PyCharm control the container:
- You start containers via Makefile/scripts
- You're debugging in a CI/CD environment
- You need to attach to an already-running production-like setup

Enter: **Python Debug Server** + `pydevd-pycharm`.

### 7.1 How It Works

1. PyCharm starts a **Debug Server** listening on a port (e.g., 5678)
2. Your Python code inside the container connects back to PyCharm via `pydevd_pycharm.settrace()`
3. Once connected, debugging works normally

### 7.2 Update Dockerfile for Remote Debugging

We need `pydevd-pycharm` installed in the image. Update `src/Dockerfile`:

```dockerfile
# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install PyCharm remote debugger (match your PyCharm version)
RUN pip install pydevd-pycharm~=251.28774

# Copy application code
COPY main.py .

# Expose port 8000 for API and 5678 for debugger
EXPOSE 8000 5678

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "debug"]
```

> **Important**: The `pydevd-pycharm` version should match your PyCharm build. Check **PyCharm → About PyCharm** for your build number.

Rebuild:
```bash
docker build -t debugging-python:debug -f src/Dockerfile src/
```

### 7.3 Add Debug Hook to Your Code

At the top of `src/main.py`, add after imports:

```python
import os

# Remote debugging hook for PyCharm
if os.getenv("PYCHARM_DEBUG", "0") == "1":
    import pydevd_pycharm
    pydevd_pycharm.settrace(
        'host.docker.internal',  # macOS Docker Desktop magic hostname
        port=5678,
        stdoutToServer=True,
        stderrToServer=True,
        suspend=False,  # Set to True to break immediately on connection
    )
```

**Explanation**:
- `host.docker.internal`: Docker Desktop's special hostname that resolves to your Mac (the host)
- `port=5678`: Must match PyCharm's Debug Server port
- `suspend=False`: Don't pause immediately; wait for breakpoints

### 7.4 Create Python Debug Server Configuration

1. **Run → Edit Configurations...**
2. Click **+** → **Python Debug Server**
3. Name: `docker-remote-debug`

| Field | Value |
|-------|-------|
| **IDE host name** | `localhost` (PyCharm's perspective) |
| **Port** | `5678` |
| **Path mappings** | Click "+" and add:<br>Local: `/Users/your-username/.../codechallenge/src`<br>Remote: `/app` |

4. Click **OK**

### 7.5 Debugging Workflow

**Step 1: Start PyCharm Debug Server**

1. Select `docker-remote-debug` configuration
2. Click **Debug** 🐛
3. PyCharm shows: `Waiting for process connection...`

**Step 2: Start Your Docker Container**

```bash
docker run -d \
  --name datetime-api-debug \
  -p 8000:8000 \
  -p 5678:5678 \
  -v "$(pwd)/src:/app" \
  -e PYCHARM_DEBUG=1 \
  datetime-api:debug
```

Or use the debug compose file:
```bash
docker-compose -f docker-compose.debug.yml up -d
```

**Step 3: Watch the Connection**

PyCharm console shows:
```
Connected to pydev debugger (build 242.23726)
```

Success! 🎉

**Step 4: Set Breakpoints and Debug**

1. Set a breakpoint anywhere in `main.py`
2. Make an API request:
```bash
curl "http://localhost:8000/datetime?tz=CET"
```

PyCharm breaks at your breakpoint, even though the container was started manually!

### 7.6 When to Use This Method

✅ **Use Debug Server when**:
- You have complex container orchestration (Kubernetes, docker-compose stacks)
- Containers are started by CI/CD or external scripts
- You're debugging in staging/production-like environments
- You need to attach to a specific long-running process

❌ **Don't use it when**:
- Simple single-container apps (use Docker interpreter instead)
- You have full control over container lifecycle (use Compose interpreter)

---

## 8. Troubleshooting Common Issues 🔧

### 8.1 Breakpoints Don't Hit in Docker

**Symptom**: You set breakpoints, but execution never pauses.

**Checklist**:

✅ **Wrong interpreter**: Ensure your Run/Debug config uses the Docker/Compose interpreter, not the local conda env

✅ **Path mismatch**: Verify path mappings. Your local `/Users/.../codechallenge/src` must map to `/app` in the container

Check: **Run → Edit Configurations → [your config] → Path mappings**

✅ **Code mismatch**: If you baked code into the image but also mount it as a volume, ensure `WORKDIR` and `PYTHONPATH` point to the mounted path, not the baked one

✅ **Debugger not attached**: For Debug Server method, verify PyCharm shows "Connected to pydev debugger"

**Fix**: Add `print("Startup OK")` in `main.py` startup code. If you see it in logs, code is running but debugger isn't attached.

### 8.2 "Could not connect to host.docker.internal:5678"

**Symptom** (in container logs):
```
ConnectionRefusedError: [Errno 61] Connection refused
```

**Causes & Fixes**:

1. **Debug Server not running**: Start the `docker-remote-debug` configuration in PyCharm first

2. **Port mismatch**: Verify `settrace(port=5678)` matches your Debug Server config

3. **Docker networking issue** (rare): Restart Docker Desktop

Test connectivity from inside container:
```bash
docker exec -it datetime-api-debug ping host.docker.internal
```

If ping fails, Docker Desktop's networking is broken. Restart it.

### 8.3 Debugging is Extremely Slow

**Causes**:

1. **Too many watches/complex evaluations**: Remove unnecessary watches. Evaluating large objects can slow stepping.

2. **Cython speedup missing**: PyCharm's debugger has optional Cython extensions for speed. If you see a warning on first debug run, follow the suggestion to compile them.

To disable Cython (useful if compilation fails):
- Add environment variable in Run/Debug config: `PYDEVD_USE_CYTHON=NO`

3. **Volume mount performance** (macOS): Docker Desktop's file sync can be slow. For huge projects, consider:
   - Using `:cached` flag: `-v "$(pwd)/src:/app:cached"`
   - Or sync files into the image (less convenient for debugging)

### 8.4 Import Errors / Module Not Found

**Symptom**: Code runs locally but fails in Docker with `ModuleNotFoundError`.

**Fix**:

1. Verify all dependencies are in `requirements.txt` and installed in the image

2. Check working directory in Run/Debug config matches `WORKDIR` in Dockerfile

3. Verify volume mounts are correct:
```bash
docker exec -it datetime-api ls /app
# Should show: main.py, requirements.txt, etc.
```

### 8.5 Environment Variables Not Available

**Symptom**: `os.getenv("SOME_VAR")` returns `None` in Docker but works locally.

**Fix**:

For **Docker Interpreter**:
- Add env vars in Run/Debug config: **Environment variables** field
- Format: `KEY1=value1;KEY2=value2`

For **Docker Compose**:
- Add to `docker-compose.yml` under `environment:`
```yaml
environment:
  - SOME_VAR=some_value
  - ANOTHER_VAR=another_value
```

For **Debug Server**:
- Add to `docker run -e` or docker-compose `environment:`

### 8.6 PyCharm Can't Find Docker

**Symptom**: "Cannot connect to Docker daemon"

**Fix**:

1. Ensure Docker Desktop is running (whale icon in menu bar)

2. Test from terminal:
```bash
docker ps
```

If it works in terminal but not in PyCharm:

3. Restart PyCharm

4. Re-add Docker connection: **Settings → Docker → + → Docker for Mac**


---

## 9. Best Practices & Workflow Recommendations 💡

### 9.1 Tiered Debugging Strategy

Use the right tool for the job:

**Tier 1: Local conda environment**
- ✅ Fast iteration on business logic
- ✅ No Docker overhead
- ✅ Use for: unit tests, algorithm development, quick feature work

**Tier 2: Docker Compose interpreter**
- ✅ Real container environment
- ✅ Multi-service integration
- ✅ Use for: API integration testing, debugging service interactions

**Tier 3: Debug Server (remote attach)**
- ✅ Complex orchestration scenarios
- ✅ Production-like environments
- ✅ Use for: "Why does this fail only in this specific environment?"

### 9.2 Conditional Debug Hooks

Instead of hardcoding `settrace()`, use environment variables:

```python
import os

if os.getenv("PYCHARM_DEBUG") == "1":
    try:
        import pydevd_pycharm
        pydevd_pycharm.settrace(
            os.getenv("DEBUG_HOST", "host.docker.internal"),
            port=int(os.getenv("DEBUG_PORT", "5678")),
            stdoutToServer=True,
            stderrToServer=True,
            suspend=False,
        )
        print("🐛 Connected to PyCharm debugger")
    except Exception as e:
        print(f"⚠️ Failed to connect to debugger: {e}")
```

This way:
- In local dev: `PYCHARM_DEBUG=0` (no debug server needed)
- In Docker debug mode: `PYCHARM_DEBUG=1`
- In production: Variable not set (no debug overhead)

### 9.3 Named Configurations for Everything

Create separate Run/Debug configs for each scenario:
- `datetime-api-local` (conda, local testing)
- `datetime-api-docker` (Docker interpreter)
- `datetime-api-compose` (Docker Compose)
- `docker-remote-debug` (Debug Server)
- `pytest-local` (local unit tests)
- `pytest-docker` (tests in Docker)

This makes switching contexts instant.

### 9.4 Use Logging Strategically

Don't rely solely on breakpoints. Add strategic logging:

```python
logger.debug(f"Processing timezone conversion: {tz_code}")
logger.info(f"Successful conversion to {tz_code}: {result}")
logger.warning(f"Unknown timezone requested: {tz_code}")
logger.error(f"Conversion failed: {e}", exc_info=True)
```

Benefits:
- Logs persist after debugging sessions
- See flow even when not debugging
- Easier to debug intermittent issues

### 9.5 Exception Breakpoints for "WTF" Moments

When you get an exception but don't know where it's coming from:

1. **Run → View Breakpoints** (`⌘⇧F8`)
2. Add Python Exception Breakpoint for the exception type
3. Run in debug mode
4. PyCharm breaks at the exact line that raises it

This is invaluable for:
- Third-party library exceptions
- Async code exceptions
- Deeply nested call stacks

### 9.6 Keep Path Mappings Consistent

Always use the same paths:
- **Docker WORKDIR**: `/app`
- **Volume mount**: `./src:/app`
- **Working directory in configs**: `/app`

This consistency prevents 90% of "breakpoints don't work" issues.

---

## 10. Real-World Debugging Scenarios with Our DateTime API 🎬

Let's walk through common debugging scenarios with our DateTime API.

### Scenario 1: "Why is my timezone conversion wrong?"

**Situation**: EST times seem off.

**Debug Process**:

1. Set breakpoint at line 206 in `_convert_to_timezone()`
2. Run: `curl "http://localhost:8000/datetime?tz=EST"`
3. Step through and inspect:
   - `offset_hours` from `TIMEZONE_OFFSETS` → should be `-5`
   - `offset = timedelta(hours=offset_hours)` → creates -5 hour offset
   - Watch `local_dt = dt + offset` calculation
   - Verify the result

**Common Issue**: Confusing EST (Eastern Standard Time) vs EDT (Eastern Daylight Time). Our simplified example doesn't handle DST. In production, use `pytz` or `zoneinfo`.

### Scenario 2: "Business hours calculation is broken"

**Situation**: The `/business-hours` endpoint says it's closed when it should be open.

**Debug Process**:

1. Set breakpoint at line 233 in `_check_if_business_hours()`
2. Make request with specific timezone:
```bash
curl "http://localhost:8000/business-hours?tz=EST&start_hour=9&end_hour=17"
```
3. In the debugger, check:
   - `current_hour` and `current_minute` values
   - `current_decimal_hour` calculation (e.g., 14:30 → 14.5)
   - The boolean expression: `start_hour <= current_decimal_hour < end_hour`

**Add a Watch**: `f"{start_hour} <= {current_decimal_hour} < {end_hour}"`

This helps you see the exact comparison being made.

### Scenario 3: "The loop only processes 2 of 3 timezones"

**Situation**: Converting to "EST,PST,INVALID" only returns EST and PST, not 3 results.

**Debug Process**:

1. Set breakpoint at line 106: `for tz in tz_list:`
2. Request:
```bash
curl "http://localhost:8000/datetime/convert?time_str=2024-11-21T15:00:00Z&timezones=EST,PST,INVALID"
```
3. Step through each loop iteration (F8)
4. Watch the Variables panel:
   - `tz_list` → `['EST', 'PST', 'INVALID']`
   - On iteration 3, `tz` = `'INVALID'`
   - The `if tz not in TIMEZONE_OFFSETS:` branch triggers
   - `continue` skips adding to results

**Insight**: The `logger.warning()` call at line 108 should log this. Check your logs!

### Scenario 4: "DateTime parsing fails with cryptic error"

**Situation**: Request with `time_str=2024-13-45T99:99:99` gives a vague 400 error.

**Debug with Exception Breakpoint**:

1. **Run → View Breakpoints** (`⌘⇧F8`)
2. Add Python Exception Breakpoint: `ValueError`
3. Click Debug and make the bad request

PyCharm breaks at line 95:
```python
original_dt = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
```

**Variables panel shows**:
- `time_str` = `"2024-13-45T99:99:99"`

Now you can see the exact invalid input. Add better error handling:

```python
except ValueError as e:
    logger.error(f"Invalid datetime '{time_str}': {e}")
    raise HTTPException(
        status_code=400,
        detail=f"Invalid datetime format: {time_str}. Use ISO format like 2024-11-21T15:00:00"
    )
```

---

## 11. Conda vs Docker: When to Use Which 🤔

### Use Local Conda Environment When:
- ✅ Developing new features (fast iteration)
- ✅ Running unit tests
- ✅ Algorithm exploration and data analysis
- ✅ Your app has no system-level dependencies

### Use Docker Debugging When:
- ✅ Testing system-level integrations (file systems, networking)
- ✅ Reproducing production-only bugs
- ✅ Working with multiple services (databases, queues)
- ✅ Verifying deployment behavior

### Pro Tip: Use Both

1. **Write code** with conda env (fast!)
2. **Run unit tests** with conda
3. **Integration test** with Docker Compose
4. **Final verification** with Docker before deploying

This workflow combines speed and accuracy.

---

## 12. Summary & Key Takeaways 🎯

You now have a complete debugging toolkit for Python development on macOS:

### Core Skills Learned:
1. ✅ **Local conda debugging** with PyCharm's powerful debugger
2. ✅ **Docker interpreter method** for seamless container debugging
3. ✅ **Docker Compose method** for multi-service stacks
4. ✅ **Remote debug server** for advanced scenarios

### Remember These Golden Rules:
- 🔑 **Path mappings must be correct** (local ↔ container)
- 🔑 **Use named Run/Debug configurations** for each scenario
- 🔑 **Combine logging with debugging** for best results
- 🔑 **Exception breakpoints** are your secret weapon
- 🔑 **Conditional breakpoints** save time on busy endpoints

### The Debugging Hierarchy:
```
Level 1: Breakpoints & step-through
    ↓
Level 2: Watches & expression evaluation
    ↓
Level 3: Conditional & exception breakpoints
    ↓
Level 4: Remote debugging in Docker
    ↓
Level 5: Multi-service Docker Compose debugging
```

Start simple, escalate as needed.

### Next Steps 🚀

1. **Set up your conda environment** for this project
2. **Create your first Run/Debug configurations**
3. **Practice with the DateTime API** endpoints
4. **Try Docker debugging** with the provided Dockerfile
5. **Experiment with Docker Compose** multi-service setups
6. **Add the remote debug hook** for advanced scenarios

---

## Resources & References 📚

- **PyCharm Debugging Guide**: [JetBrains Official Docs](https://www.jetbrains.com/help/pycharm/debugging-code.html?utm_source=genmind.ch)
- **Docker Integration**: [PyCharm Docker Support](https://www.jetbrains.com/help/pycharm/docker.html?utm_source=genmind.ch)
- **Remote Debugging**: [PyCharm Remote Debug](https://www.jetbrains.com/help/pycharm/remote-debugging-with-product.html?utm_source=genmind.ch)
- **FastAPI Documentation**: [FastAPI Official Docs](https://fastapi.tiangolo.com/?utm_source=genmind.ch)
- **Docker Desktop for Mac**: [Docker Docs](https://docs.docker.com/desktop/mac/?utm_source=genmind.ch)

---

## Feedback & Questions 💬

Mastering debugging is a journey, not a destination. The techniques in this guide will serve you well across any Python project, whether it's a small script or a large microservices architecture.

Remember: **The best debugger is the one you actually use.** Start with breakpoints, build from there, and soon you'll be hunting down bugs like a pro. 🐛🔫

Happy debugging! 🎉

---

*Written with ❤️ for the Python & PyCharm community*  
*Last updated: November 2024*
