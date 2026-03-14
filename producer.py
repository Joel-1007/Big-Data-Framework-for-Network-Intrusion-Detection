import time
import csv
import json
import glob
from kafka import KafkaProducer

# 1. Initialize Producer
# We connect to localhost:9092 where your Docker container is listening
try:
    producer = KafkaProducer(
        bootstrap_servers=['localhost:9092'],
        value_serializer=lambda x: json.dumps(x).encode('utf-8')
    )
    print("--- Kafka Producer Connected Successfully ---")
except Exception as e:
    print(f"Error connecting to Kafka: {e}")
    exit()

# 2. Get Data File
csv_files = glob.glob("*.csv")
if not csv_files:
    print("Error: No CSV files found!")
    exit()

filename = csv_files[0]
print(f"Streaming traffic from: {filename}")

# 3. Stream Data
try:
    with open(filename, 'r') as file:
        reader = csv.DictReader(file)
        count = 0
        print("Starting stream... (Press Ctrl+C to stop)")
        for row in reader:
            # Clean keys (strip spaces)
            clean_row = {k.strip(): v for k, v in row.items()}
            
            # Send to Kafka Topic 'network-flows'
            producer.send('network-flows', value=clean_row)
            
            count += 1
            if count % 1000 == 0:
                print(f"Sent {count} packets...")
                # Sleep briefly to simulate real-time speed
                time.sleep(0.1) 
except KeyboardInterrupt:
    print("\nStopping stream.")

print("Stream finished.")
