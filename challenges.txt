## Challenges Faced

### Test Simulation Connection Issue
Initially, the test simulation was not working. Despite following the setup instructions, the simulator never connected to my Python server. As a result, no telemetry events were being sent, and no steering commands were emitted. This caused significant delays in testing and debugging my self-driving car model.

### How I Fixed It
After investigating, I identified that the issue was due to the server environment setup. The Python server was being run in a way that prevented proper WebSocket communication with the simulator. I resolved this by configuring the server to correctly handle WebSocket connections, ensuring the simulator could connect and exchange telemetry data. Once this was fixed, the simulator successfully communicated with the Python server, allowing me to receive real-time telemetry data and send steering commands, which enabled proper testing of the model.
