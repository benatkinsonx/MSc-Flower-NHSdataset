import os

# Check if CLIENT_ID exists
if 'CLIENT_ID' in os.environ:
    print(f"CLIENT_ID already set to: {os.environ['CLIENT_ID']}")
    client_id = int(os.environ['CLIENT_ID'])
else:
    print("CLIENT_ID not set, using default value 0")
    client_id = 0

print(f"Final client_id value: {client_id}")