## Logging (rosbag)

### Run Command for Logging
```
rosbag record -O performance_capture.bag /rosout_agg
```

Ctrl + C to exit

This captures a bag in the root directory `/catkin_ws/`, which you can then convert to a log file using `bagtolog.py`.

More robust than previous version, make sure you use `bagtolog.py` with `python3`.

You will need to install dependencies (atleast I ran into them):
```
pip install pycryptodomex
pip install python-gnupg
```

Then run `Logging.py`.

> ⚠️ Note: Make sure to point to the log file — update the code (path is hardcoded). IN BOTH CASES