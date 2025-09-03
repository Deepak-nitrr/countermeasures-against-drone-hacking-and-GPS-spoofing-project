# countermeasures-against-drone-hacking-and-GPS-spoofing-project
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

OUTPUT_DIR = os.path.join(os.getcwd(), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

np.random.seed(42)
time = np.arange(100)
gps_x = np.cumsum(np.random.normal(0, 1, 100))
gps_y = np.cumsum(np.random.normal(0, 1, 100))

gps_x[50:55] += np.random.normal(20, 5, 5)
gps_y[70:75] += np.random.normal(-25, 5, 5)

gps_data = pd.DataFrame({"time": time, "x": gps_x, "y": gps_y})
model = IsolationForest(contamination=0.1, random_state=42)
gps_data["anomaly"] = model.fit_predict(gps_data[["x", "y"]])

plt.figure(figsize=(8, 6))
normal = gps_data[gps_data["anomaly"] == 1]
anomaly = gps_data[gps_data["anomaly"] == -1]
plt.plot(normal["x"], normal["y"], "bo-", label="Normal Path")
plt.plot(anomaly["x"], anomaly["y"], "rx", label="Spoofed Points", markersize=10)
plt.xlabel("X coordinate")
plt.ylabel("Y coordinate")
plt.title("GPS Spoofing Detection (Simulated)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "gps_spoofing.png"))
plt.close()

packets = np.random.normal(50, 5, 200)
packets[120:130] = np.random.normal(120, 5, 10)
network_data = pd.DataFrame({"packets": packets})
model_net = IsolationForest(contamination=0.05, random_state=42)
network_data["anomaly"] = model_net.fit_predict(network_data[["packets"]])

plt.figure(figsize=(8, 6))
plt.plot(network_data.index, network_data["packets"], label="Traffic", color="blue")
plt.scatter(
    network_data[network_data["anomaly"] == -1].index,
    network_data[network_data["anomaly"] == -1]["packets"],
    color="red", label="Detected Attack"
)
plt.xlabel("Time")
plt.ylabel("Packets/sec")
plt.title("Network Intrusion Detection (Simulated)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "network_intrusion.png"))
plt.close()

gps_data.to_csv(os.path.join(OUTPUT_DIR, "drone_data.csv"), index=False)
print("✅ Analysis complete! Results saved in 'output/' folder.")

!zip -r output_files.zip output
from google.colab import files
files.download("output_files.zip")
