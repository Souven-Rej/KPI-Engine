import json, urllib.request

resp = urllib.request.urlopen("http://localhost:8000/api/scenarios")
data = json.loads(resp.read())

# Print first scenario to see its keys
print("Keys:", list(data["scenarios"][0].keys()))
print()
for s in data["scenarios"]:
    print(json.dumps(s))
